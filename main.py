"""
LP Content Engine — Weekly social media draft generator.

Fetches artists from Airtable, searches for recent news via Claude web search,
deduplicates against a Google Sheet of already-used topics, generates
platform-native LinkedIn/Instagram/Facebook posts via Claude, and queues
them as Buffer drafts for human review.

Usage:
  python main.py               # normal weekly run
  python main.py --dry-run     # log everything, skip Sheets and Buffer writes
  python main.py --test-airtable  # print artist list and exit
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from anthropic.types import TextBlock
import gspread
import requests
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BUFFER_API_KEY = os.environ.get("BUFFER_API_KEY", "")
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY", "")
AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "appMMwX47V1g2Sv5u")
AIRTABLE_TABLE_ID = os.environ.get("AIRTABLE_ARTIST_TABLE", "tbloEhiPP4kyTTVDb")
SHEETS_ID = os.environ.get("FOUND_NEWS_STORIES_SHEETS_ID", "")

AIRTABLE_PRIORITY_ORDER = ["Top of Roster", "Exclusive", "Core Roster"]

SEARCH_MODEL = "claude-haiku-4-5"
CONTENT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

COST_CAP_USD = float(os.environ.get("COST_CAP_USD", "5.00"))
_estimated_cost_usd = 0.0
_HAIKU_INPUT = 1.00 / 1_000_000
_HAIKU_OUTPUT = 5.00 / 1_000_000
_SONNET_INPUT = 3.00 / 1_000_000
_SONNET_OUTPUT = 15.00 / 1_000_000
_SEARCH_COST = 0.01  # per web search use

_claude_call_count = 0
CLAUDE_CALL_LIMIT = 50
_THROTTLE_FILE = "/tmp/lp_content_throttle.txt"
_THROTTLE_BUFFER = 2  # extra seconds after API reset timestamp

BUFFER_API_URL = "https://api.buffer.com"
SKILL_GRAPH_DIR = Path(__file__).parent / "content-skill-graph"

# ── Rate limiting & cost tracking (pattern from love-automations) ─────────────


def _load_throttle() -> float:
    try:
        return float(Path(_THROTTLE_FILE).read_text().strip())
    except Exception:
        return 0.0


def _save_throttle(t: float) -> None:
    try:
        Path(_THROTTLE_FILE).write_text(str(t))
    except Exception:
        pass


def _claude_throttle() -> None:
    wait = _load_throttle() - time.time()
    if wait > 0:
        log.info("Rate limit: waiting %.0fs...", wait)
        time.sleep(wait)


def _claude_call_done(headers: dict) -> None:
    reset_str = headers.get("anthropic-ratelimit-input-tokens-reset") or headers.get(
        "anthropic-ratelimit-tokens-reset"
    )
    if reset_str:
        try:
            reset_dt = datetime.fromisoformat(reset_str.replace("Z", "+00:00"))
            next_at = reset_dt.timestamp() + _THROTTLE_BUFFER
            _save_throttle(next_at)
            log.info(
                "Token reset at %s — next call allowed in %.0fs",
                reset_str,
                max(0, next_at - time.time()),
            )
            return
        except Exception:
            pass
    _save_throttle(time.time() + 90)


def _track_cost(resp, model: str) -> None:
    global _estimated_cost_usd
    usage = getattr(resp, "usage", None)
    if usage:
        if "sonnet" in model:
            in_cost, out_cost = _SONNET_INPUT, _SONNET_OUTPUT
        else:
            in_cost, out_cost = _HAIKU_INPUT, _HAIKU_OUTPUT
        _estimated_cost_usd += getattr(usage, "input_tokens", 0) * in_cost
        _estimated_cost_usd += getattr(usage, "output_tokens", 0) * out_cost
    server_tool = getattr(getattr(resp, "usage", None), "server_tool_use", None)
    searches = getattr(server_tool, "web_search_requests", 0) if server_tool else 0
    _estimated_cost_usd += searches * _SEARCH_COST
    log.debug("Est. cost so far: $%.4f / $%.2f cap", _estimated_cost_usd, COST_CAP_USD)


def _under_cost_cap(label: str) -> bool:
    if _estimated_cost_usd >= COST_CAP_USD:
        log.warning(
            "Cost cap $%.2f reached (est. $%.4f) — skipping %s",
            COST_CAP_USD,
            _estimated_cost_usd,
            label,
        )
        return False
    return True


# ── Loading ───────────────────────────────────────────────────────────────────


def load_env() -> None:
    # In GitHub Actions, Google credentials come as a JSON string in an env var
    gc_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
    if gc_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(gc_json)
        tmp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        log.info("Wrote Google credentials from env to %s", tmp.name)

    missing = [
        k
        for k, v in {
            "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
            "BUFFER_API_KEY": BUFFER_API_KEY,
            "AIRTABLE_API_KEY": AIRTABLE_API_KEY,
            "FOUND_NEWS_STORIES_SHEETS_ID": SHEETS_ID,
        }.items()
        if not v
    ]
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        sys.exit(1)


def load_skill_graph() -> str:
    """Read all markdown files in content-skill-graph/ into one concatenated string."""
    parts = []
    for md_file in sorted(SKILL_GRAPH_DIR.rglob("*.md")):
        rel = md_file.relative_to(SKILL_GRAPH_DIR.parent)
        parts.append(f"## {rel}\n\n{md_file.read_text().strip()}")
    return "\n\n---\n\n".join(parts)


def load_artist_mappings() -> dict[str, str]:
    """Parse artists.md markdown table → {tribute_name: original_artist}."""
    path = SKILL_GRAPH_DIR / "engine" / "artists.md"
    mappings: dict[str, str] = {}
    if not path.exists():
        return mappings
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if (
            not stripped.startswith("|")
            or "---" in stripped
            or "Tribute Act" in stripped
        ):
            continue
        cols = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cols) >= 2 and cols[0]:
            mappings[cols[0]] = cols[1]
    return mappings


# ── Airtable ──────────────────────────────────────────────────────────────────


def fetch_airtable_artists() -> list[dict]:
    """Fetch artists filtered by Marketing Priority, sorted by priority order."""
    priority_filter = ", ".join(
        f"{{Marketing Priority}}='{p}'" for p in AIRTABLE_PRIORITY_ORDER
    )
    params = {
        "fields[]": ["Artist / Show Name", "Marketing Priority"],
        "filterByFormula": f"OR({priority_filter})",
    }
    try:
        resp = requests.get(
            f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}",
            headers={"Authorization": f"Bearer {AIRTABLE_API_KEY}"},
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.error("Airtable fetch error: %s", exc)
        return []

    def _priority_key(record: dict) -> int:
        p = record["fields"].get("Marketing Priority", "")
        try:
            return AIRTABLE_PRIORITY_ORDER.index(p)
        except ValueError:
            return len(AIRTABLE_PRIORITY_ORDER)

    records = sorted(resp.json().get("records", []), key=_priority_key)
    return [
        {
            "name": r["fields"].get("Artist / Show Name", ""),
            "priority": r["fields"].get("Marketing Priority", ""),
        }
        for r in records
        if r["fields"].get("Artist / Show Name")
    ]


# ── Google Sheets ─────────────────────────────────────────────────────────────

SHEETS_HEADER = ["artist", "original_artist", "headline", "url", "date_added"]


def _get_sheet():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        log.error("GOOGLE_APPLICATION_CREDENTIALS not set")
        return None
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(SHEETS_ID).sheet1


def read_used_topics() -> set[str]:
    """Return set of all URLs and headlines already in the sheet."""
    sheet = _get_sheet()
    if not sheet:
        return set()
    rows = sheet.get_all_values()
    used: set[str] = set()
    for row in rows[1:]:  # skip header
        if len(row) > 3 and row[3]:
            used.add(row[3].strip())  # url column
        if len(row) > 2 and row[2]:
            used.add(row[2].strip())  # headline column as fallback
    return used


def mark_topics_used(topics: list[dict], dry_run: bool = False) -> None:
    if not topics:
        return
    if dry_run:
        log.info("[dry-run] Would mark %d topics as used in Sheets", len(topics))
        return
    sheet = _get_sheet()
    if not sheet:
        return
    if not sheet.get_all_values():
        sheet.append_row(SHEETS_HEADER)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for t in topics:
        sheet.append_row(
            [
                t.get("artist", ""),
                t.get("original_artist", ""),
                t.get("headline", ""),
                t.get("url", ""),
                today,
            ]
        )
    log.info("Marked %d topics as used in Sheets", len(topics))


# ── News search ───────────────────────────────────────────────────────────────


def search_artist_news(tribute: str, original: str) -> list[dict]:
    """Search for recent news about a tribute act (and optionally the original artist)."""
    global _claude_call_count
    if _claude_call_count >= CLAUDE_CALL_LIMIT or not _under_cost_cap(tribute):
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    original_line = (
        f" Also search for recent news about these original artists (the acts this tribute "
        f"represents): {original}. Search for the original artists themselves — NOT tribute "
        f"bands or cover acts for them."
        if original
        else ""
    )

    prompt = (
        f"Search for news articles published in the last 14 days about '{tribute}'. "
        f"IMPORTANT: Search for the exact act name '{tribute}' only — ignore any results "
        f"about other tribute bands, cover acts, or similarly-named performers. "
        f"{original_line} "
        f"Today's date is {today}. "
        "Do 2-3 targeted searches. Then return ONLY a JSON array. "
        "Each object in the array must have these exact keys: "
        "headline (string), url (string), summary (1-2 sentence string), "
        "hook_type (one of: 'upcoming_show', 'tribute_news', 'original_artist_news'), "
        "artist (the exact name of the tribute act or original artist this news is about). "
        f"IMPORTANT: If the news is about a specific show or event, only include it if the "
        f"show date is in the future (after {today}). Do not include past shows. "
        "If no relevant news found in the last 14 days, return an empty array []. "
        "Do not include any text outside the JSON array."
    )

    _claude_throttle()
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        raw = client.messages.with_raw_response.create(
            model=SEARCH_MODEL,
            max_tokens=MAX_TOKENS,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        resp = raw.parse()
        _claude_call_count += 1
        _claude_call_done(dict(raw.headers))
        _track_cost(resp, SEARCH_MODEL)
    except Exception as exc:
        log.error("News search error for %s: %s", tribute, exc)
        return []

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        log.warning("No news items found for %s", tribute)
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.error("JSON parse error for %s news search: %s", tribute, exc)
        return []

    for item in items:
        item.setdefault("original_artist", original)
    log.info("Found %d news items for %s", len(items), tribute)
    return items


# ── Deduplication ─────────────────────────────────────────────────────────────


def filter_new_topics(found: list[dict], used: set[str]) -> list[dict]:
    new = []
    for item in found:
        key = item.get("url", "").strip() or item.get("headline", "").strip()
        if key and key not in used:
            new.append(item)
    return new


# ── Content generation ────────────────────────────────────────────────────────


def generate_posts(topic: dict, skill_graph: str) -> dict | None:
    """Generate LinkedIn, Instagram, and Facebook posts for a topic."""
    global _claude_call_count
    if _claude_call_count >= CLAUDE_CALL_LIMIT or not _under_cost_cap(
        topic.get("headline", "")
    ):
        return None

    user_prompt = (
        "Generate social media content for Love Productions based on this news topic:\n\n"
        f"Tribute Act: {topic.get('artist', '')}\n"
        f"Original Artist: {topic.get('original_artist', '') or 'N/A'}\n"
        f"Headline: {topic.get('headline', '')}\n"
        f"URL: {topic.get('url', '')}\n"
        f"Summary: {topic.get('summary', '')}\n"
        f"Suggested Hook Type: {topic.get('hook_type', '')}\n\n"
        "Follow the content skill graph instructions exactly. Write all three platform posts "
        "in the repurposing chain order (LinkedIn first, then Instagram, then Facebook). "
        "Each post must think about the topic differently — not just reformatted.\n\n"
        "Include the source URL in every post. For LinkedIn and Facebook, weave it naturally "
        "into the post body (e.g. 'Full story here: <url>' or 'Read more: <url>'). "
        "For Instagram, place it at the end of the caption before the hashtags.\n\n"
        "Return ONLY a JSON object with these exact keys: linkedin, instagram, facebook. "
        "Each value is the full post text, ready to publish. No other text outside the JSON."
    )

    _claude_throttle()
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        raw = client.messages.with_raw_response.create(
            model=CONTENT_MODEL,
            max_tokens=MAX_TOKENS,
            system=skill_graph,
            messages=[{"role": "user", "content": user_prompt}],
        )
        resp = raw.parse()
        _claude_call_count += 1
        _claude_call_done(dict(raw.headers))
        _track_cost(resp, CONTENT_MODEL)
    except Exception as exc:
        log.error("Content generation error for '%s': %s", topic.get("headline"), exc)
        return None

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        log.error(
            "No JSON in content generation response for '%s'", topic.get("headline")
        )
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.error("JSON parse error in content generation: %s", exc)
        return None


# ── Buffer (GraphQL API) ──────────────────────────────────────────────────────


def _buffer_gql(query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query or mutation against the Buffer API."""
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    for attempt in range(2):
        try:
            resp = requests.post(
                BUFFER_API_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {BUFFER_API_KEY}",
                },
                timeout=15,
            )
            if resp.status_code == 429:
                try:
                    wait = int(resp.json().get("retryAfter", 61))
                except Exception:
                    wait = 61
                log.warning("Buffer rate limited — retrying in %ds...", wait)
                time.sleep(wait)
                continue
            if not resp.ok:
                log.error("Buffer GraphQL %d: %s", resp.status_code, resp.text[:500])
                return {}
            return resp.json()
        except Exception as exc:
            log.error("Buffer GraphQL error: %s", exc)
            return {}
    log.error("Buffer rate limit persists after retry")
    return {}


def discover_buffer_profiles() -> dict[str, str]:
    """Return {platform: channel_id} for linkedin, instagram, facebook."""
    # Step 1: get the organization ID
    data = _buffer_gql("query { account { organizations { id name } } }")
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        log.error("No Buffer organizations found — check API key scopes")
        return {}
    org_id = orgs[0]["id"]
    log.info("Buffer org: %s (%s)", orgs[0].get("name", ""), org_id)

    # Step 2: get channels for that organization
    data = _buffer_gql(
        """
        query GetChannels($input: ChannelsInput!) {
          channels(input: $input) { id service displayName }
        }
        """,
        {"input": {"organizationId": org_id}},
    )
    channels = data.get("data", {}).get("channels", [])
    profiles: dict[str, str] = {}
    for c in channels:
        service = c.get("service", "").lower()
        if service in ("linkedin", "instagram", "facebook") and service not in profiles:
            profiles[service] = c["id"]
    log.info("Buffer profiles found: %s", list(profiles.keys()))
    return profiles


def post_draft_to_buffer(
    text: str,
    channel_id: str,
    platform: str = "",
    dry_run: bool = False,
    image: str | None = None,
) -> bool:
    if dry_run:
        log.info(
            "[dry-run] Buffer %s draft (%d chars):\n%.300s\n---",
            platform,
            len(text),
            text,
        )
        return True
    post_input: dict = {
        "text": text,
        "channelId": channel_id,
        "schedulingType": "automatic",
        "mode": "addToQueue",
        "saveToDraft": True,
    }
    if platform == "facebook":
        post_input["metadata"] = {"facebook": {"type": "post"}}
    elif platform == "instagram":
        post_input["metadata"] = {"instagram": {"type": "post", "shouldShareToFeed": True}}
        if image:
            post_input["assets"] = {"images": [{"url": image}]}
    data = _buffer_gql(
        """
        mutation CreateDraft($input: CreatePostInput!) {
          createPost(input: $input) {
            ... on PostActionSuccess { post { id } }
            ... on MutationError { message }
          }
        }
        """,
        {"input": post_input},
    )
    result = data.get("data", {}).get("createPost", {})
    if "message" in result:
        log.error("Buffer draft error: %s", result["message"])
        return False
    post_id = result.get("post", {}).get("id", "")
    log.info("Buffer draft created (id=%s)", post_id)
    return bool(post_id)


# ── Expired draft cleanup ─────────────────────────────────────────────────────

# Date patterns to detect show announcements, e.g. "January 15", "Jan 15, 2026",
# "15th January", "01/15/2026", "2026-01-15".
_DATE_PATTERNS = [
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",          # January 15 / Jan 15, 2026
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:,?\s+\d{4})?\b",                                      # 15th January / 15 Jan 2026
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                           # 01/15/2026
    r"\b\d{4}-\d{2}-\d{2}\b",                                  # 2026-01-15
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)

# Keywords that suggest a post is a show announcement rather than general news.
_SHOW_KEYWORDS = re.compile(
    r"\b(ticket|tickets|show|concert|performance|doors open|venue|live at|"
    r"on sale|book now|get your tickets|don't miss|link in bio)\b",
    re.IGNORECASE,
)


def _extract_earliest_date(text: str) -> datetime | None:
    """Return the earliest future-or-past date found in text, or None."""
    today = datetime.now(tz=timezone.utc).date()
    earliest = None
    for match in _DATE_RE.finditer(text):
        raw = match.group()
        for fmt in (
            "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
            "%B %d", "%b %d",
            "%d %B %Y", "%d %b %Y", "%d %B", "%d %b",
            "%m/%d/%Y", "%m/%d/%y",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(raw.strip(), fmt)
                # If no year was parsed, assume current year (or next if already past)
                if dt.year == 1900:
                    dt = dt.replace(year=today.year)
                    if dt.date() < today:
                        dt = dt.replace(year=today.year + 1)
                date = dt.date()
                if earliest is None or date < earliest:
                    earliest = date
                break
            except ValueError:
                continue
    return datetime(earliest.year, earliest.month, earliest.day, tzinfo=timezone.utc) if earliest else None


def _is_expired_show_announcement(text: str) -> bool:
    """Return True if the post looks like a show announcement with a past date."""
    if not _SHOW_KEYWORDS.search(text):
        return False
    dt = _extract_earliest_date(text)
    if dt is None:
        return False
    return dt.date() < datetime.now(tz=timezone.utc).date()


def purge_expired_show_drafts(dry_run: bool = False) -> None:
    """Fetch all Buffer drafts and delete any expired show announcements."""
    # Get org and channels
    data = _buffer_gql("query { account { organizations { id name } } }")
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        log.error("purge_expired_show_drafts: no Buffer org found")
        return
    org_id = orgs[0]["id"]

    data = _buffer_gql(
        "query GetChannels($input: ChannelsInput!) { channels(input: $input) { id service } }",
        {"input": {"organizationId": org_id}},
    )
    channels = data.get("data", {}).get("channels", [])
    channel_ids = [
        c["id"] for c in channels
        if c.get("service", "").lower() in ("linkedin", "instagram", "facebook")
    ]

    deleted = 0
    for channel_id in channel_ids:
        data = _buffer_gql(
            """
            query GetDrafts($input: PostsInput!) {
              posts(input: $input) { edges { node { id text } } }
            }
            """,
            {"input": {"channelId": channel_id, "status": ["draft"]}},
        )
        posts = [e["node"] for e in data.get("data", {}).get("posts", {}).get("edges", []) if e.get("node")]
        for post in posts:
            text = post.get("text", "")
            if not _is_expired_show_announcement(text):
                continue
            log.info("Expired show draft found (id=%s): %.80s...", post["id"], text)
            if dry_run:
                log.info("[dry-run] Would delete post %s", post["id"])
                deleted += 1
                continue
            result = _buffer_gql(
                """
                mutation DeletePost($input: DeletePostInput!) {
                  deletePost(input: $input) {
                    ... on PostActionSuccess { post { id } }
                    ... on MutationError { message }
                  }
                }
                """,
                {"input": {"id": post["id"]}},
            ).get("data", {}).get("deletePost", {})
            if "message" in result:
                log.error("Failed to delete post %s: %s", post["id"], result["message"])
            else:
                log.info("Deleted expired draft %s", post["id"])
                deleted += 1

    log.info("purge_expired_show_drafts: %d post(s) %s", deleted, "would be deleted" if dry_run else "deleted")


# ── Orchestration ─────────────────────────────────────────────────────────────


def main(dry_run: bool = False, single_artist: str = "") -> None:
    load_env()

    # Uncomment to delete expired show announcement drafts from Buffer at the start of each run:
    # purge_expired_show_drafts(dry_run=dry_run)

    log.info("Loading content skill graph...")
    skill_graph = load_skill_graph()
    mappings = load_artist_mappings()
    log.info("Loaded %d artist mappings", len(mappings))

    if single_artist:
        artists = [{"name": single_artist, "priority": "manual"}]
        log.info("Single-artist mode: %s", single_artist)
    else:
        artists = fetch_airtable_artists()
        if not artists:
            log.error("No artists fetched from Airtable — aborting")
            return
        log.info("Fetched %d artists from Airtable", len(artists))

    used = read_used_topics()
    log.info("Loaded %d already-used topics from Sheets", len(used))

    buffer_profiles = discover_buffer_profiles()
    if not buffer_profiles and not dry_run:
        log.error("No Buffer profiles found — aborting")
        return

    all_new_topics: list[dict] = []

    for artist in artists:
        name = artist["name"]
        original = mappings.get(name, "")
        log.info("--- Processing: %s [%s]", name, artist["priority"])

        found = search_artist_news(name, original)
        new_topics = filter_new_topics(found, used)

        if not new_topics:
            log.info("No new topics for %s", name)
            continue

        log.info("%d new topic(s) for %s", len(new_topics), name)

        for topic in new_topics:
            headline = topic.get("headline", "")[:80]
            log.info("Generating posts for: %s", headline)

            posts = generate_posts(topic, skill_graph)
            if not posts:
                continue

            for platform in ("linkedin", "instagram", "facebook"):
                text = posts.get(platform, "").replace(" — ", "—").replace(" – ", "–")
                if not text:
                    log.warning("No %s post generated for '%s'", platform, headline)
                    continue
                image = None
                if platform == "instagram":
                    image = "https://www.loveproductions.com/wp-content/uploads/2022/03/LPI_logo_RGB_Red_BLK.png"
                    # log.info("  instagram caption ready (%d chars) — add media before posting", len(text))
                    # log.info("  instagram caption:\n%s", text)
                    # continue
                profile_id = buffer_profiles.get(platform, "")
                if not profile_id and not dry_run:
                    log.warning("No Buffer profile for %s — skipping", platform)
                    continue
                ok = post_draft_to_buffer(
                    text,
                    profile_id,
                    platform=platform,
                    dry_run=dry_run,
                    image=image if platform == "instagram" else None,
                )
                if ok:
                    log.info("  %s draft queued (%d chars)", platform, len(text))
                else:
                    log.warning("  %s draft FAILED — see error above", platform)

            key = topic.get("url", "").strip() or topic.get("headline", "").strip()
            used.add(key)

        all_new_topics.extend(new_topics)
        mark_topics_used(new_topics, dry_run=dry_run)

    log.info(
        "=== Run complete. Topics processed: %d | Est. cost: $%.4f ===",
        len(all_new_topics),
        _estimated_cost_usd,
    )


def test_buffer() -> None:
    """Print all connected Buffer channels and verify the API key works."""
    load_dotenv()
    if not BUFFER_API_KEY:
        print("BUFFER_API_KEY not set in .env")
        sys.exit(1)

    # Get organization
    data = _buffer_gql("query { account { organizations { id name } } }")
    errors = data.get("errors")
    if errors:
        print(f"Buffer API error: {errors}")
        sys.exit(1)
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        print("No organizations found. Check that the API key has the correct scopes.")
        sys.exit(1)

    org_id = orgs[0]["id"]
    org_name = orgs[0].get("name", "")
    print(f"Connected — org: {org_name} ({org_id})\n")

    # Get channels
    data = _buffer_gql(
        "query GetChannels($input: ChannelsInput!) { channels(input: $input) { id service displayName } }",
        {"input": {"organizationId": org_id}},
    )
    channels = data.get("data", {}).get("channels", [])
    if not channels:
        print("No channels found for this organization.")
        sys.exit(1)

    matched_channels = []
    print(f"{len(channels)} channel(s) found:\n")
    for c in channels:
        service = c.get("service", "unknown").lower()
        name = c.get("displayName", "")
        cid = c.get("id", "")
        is_match = service in ("linkedin", "instagram", "facebook")
        marker = "✓" if is_match else " "
        print(f"  [{marker}] {service:12s}  {name:30s}  id={cid}")
        if is_match:
            matched_channels.append({"service": service, "id": cid, "name": name})

    print("\nChannels marked [✓] will be used by the content engine.")

    if not matched_channels:
        print("\nNo matched channels — skipping draft test.")
        return

    _TEST_IMAGE = "https://www.loveproductions.com/wp-content/uploads/2022/03/LPI_logo_RGB_Red_BLK.png"
    test_text = (
        "[LP Content Engine test] This is an automated test draft — safe to delete."
    )
    print(f"\nCreating test draft on {len(matched_channels)} channel(s)...\n")
    all_ok = True
    for c in matched_channels:
        img = _TEST_IMAGE if c["service"] == "instagram" else None
        ok = post_draft_to_buffer(test_text, c["id"], platform=c["service"], image=img)
        status = "OK" if ok else "FAILED"
        print(f"  {c['service']:12s}  {c['name']:30s}  {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print(
            "\nAll test drafts created successfully. Check Buffer to confirm, then delete them."
        )
    else:
        print("\nOne or more drafts failed — check the error log above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LP Content Engine — weekly social draft generator"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log outputs, skip Sheets and Buffer writes",
    )
    parser.add_argument(
        "--test-airtable",
        action="store_true",
        help="Print Airtable artist list and exit",
    )
    parser.add_argument(
        "--test-buffer",
        action="store_true",
        help="List Buffer channels and post a test draft to each matched one",
    )
    parser.add_argument(
        "--artist",
        metavar="NAME",
        help="Run the full pipeline for a single artist by name (skips Airtable fetch)",
    )
    args = parser.parse_args()

    if args.test_airtable:
        load_dotenv()
        artists = fetch_airtable_artists()
        if not artists:
            print(
                "No artists returned. Check AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_ARTIST_TABLE."
            )
        for a in artists:
            print(f"  [{a['priority']}] {a['name']}")
        sys.exit(0)

    if args.test_buffer:
        test_buffer()
        sys.exit(0)

    if args.artist:
        main(dry_run=args.dry_run, single_artist=args.artist)
        sys.exit(0)

    main(dry_run=args.dry_run)

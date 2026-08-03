import logging
import re
import sys
import time
from datetime import datetime, timezone

import requests

from .config import BUFFER_API_KEY, BUFFER_API_URL, _IG_PLACEHOLDER

log = logging.getLogger(__name__)

# ── Date/keyword patterns for expired show detection ─────────────────────────

_DATE_PATTERNS = [
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:,?\s+\d{4})?\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)
_SHOW_KEYWORDS = re.compile(
    r"\b(ticket|tickets|show|concert|performance|doors open|venue|live at|"
    r"on sale|book now|get your tickets|don't miss|link in bio)\b",
    re.IGNORECASE,
)


# ── GraphQL helper ────────────────────────────────────────────────────────────


def _buffer_gql(query: str, variables: dict | None = None) -> dict:
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    for _ in range(2):
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
                    body = resp.json()
                    window = body.get("errors", [{}])[0].get("extensions", {}).get("window", "")
                    retry_after = body.get("retryAfter") or body.get("retry_after")
                except Exception:
                    window, retry_after = "", None
                if window == "24h":
                    sys.exit("Buffer daily API limit reached. Try again tomorrow.")
                wait = int(retry_after) if retry_after else 61
                log.warning("Buffer rate limited, retrying in %ds...", wait)
                time.sleep(wait)
                continue
            if not resp.ok:
                log.error("Buffer GraphQL %d: %s", resp.status_code, resp.text[:500])
                return {}
            body = resp.json()
            if body.get("errors"):
                log.error("Buffer GraphQL errors: %s", body["errors"])
            return body
        except Exception as exc:
            log.error("Buffer GraphQL error: %s", exc)
            return {}
    log.error("Buffer rate limit persists after retry")
    return {}


# ── Channel discovery ─────────────────────────────────────────────────────────


def discover_buffer_profiles() -> dict[str, str]:
    """Return {platform: channel_id} for linkedin, instagram, facebook."""
    data = _buffer_gql("query { account { organizations { id name } } }")
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        log.error("No Buffer organizations found, check API key scopes")
        return {}
    org_id = orgs[0]["id"]
    log.info("Buffer org: %s (%s)", orgs[0].get("name", ""), org_id)

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


# ── Draft creation ────────────────────────────────────────────────────────────


def post_draft_to_buffer(
    text: str,
    channel_id: str,
    platform: str = "",
    dry_run: bool = False,
    image: str | None = None,
    scheduled_at: datetime | None = None,
    images: list[str] | None = None,
    video: str | None = None,
    thumbnail: str | None = None,
) -> bool:
    """Draft one post to a Buffer channel.

    ``image`` attaches a single asset. ``images`` attaches several, which
    Instagram renders as a swipeable carousel; it takes precedence over
    ``image`` when both are given. ``CreatePostInput.assets`` is typed
    ``[AssetInput!]!``, verified by schema introspection on 2026-07-31, so a
    list is the native shape (the long-standing single-dict form worked only
    because GraphQL coerces a lone value into a one-element list).

    ``video`` is a public mp4 URL and outranks every image: ``AssetInput`` has a
    ``video`` member (``VideoAssetInput``), so a clip needs nothing beyond a
    hosted file. ``thumbnail`` is its poster frame, optional but worth sending,
    since without one the platforms pick their own and often land on a black
    frame.
    """
    assets = [u for u in (images or ([image] if image else [])) if u]
    if dry_run:
        slot_str = f" @ {scheduled_at.strftime('%a %b %d %I:%M%p %Z')}" if scheduled_at else ""
        if video:
            log.info("[dry-run] with video %s", video)
        elif len(assets) > 1:
            log.info("[dry-run] with %d-slide carousel", len(assets))
        log.info(
            "[dry-run] Buffer %s draft%s (%d chars):\n%s\n---",
            platform,
            slot_str,
            len(text),
            text,
        )
        return True
    post_input: dict = {
        "text": text,
        "channelId": channel_id,
        "schedulingType": "automatic",
        "saveToDraft": True,
    }
    if scheduled_at:
        post_input["mode"] = "customScheduled"
        post_input["dueAt"] = scheduled_at.isoformat()
    else:
        post_input["mode"] = "addToQueue"
    video_asset = None
    if video:
        va: dict = {"url": video}
        if thumbnail:
            va["thumbnailUrl"] = thumbnail
        video_asset = [{"video": va}]

    if platform == "facebook":
        # Normally no image asset, so the URL in the body unfurls a native
        # link-preview card. A card post has no URL to unfurl, so it passes
        # assets explicitly and they are attached here.
        post_input["metadata"] = {"facebook": {"type": "video" if video else "post"}}
        if video_asset or assets:
            post_input["assets"] = video_asset or [{"image": {"url": u}} for u in assets]
    elif platform == "instagram":
        # A clip posts as a reel; Instagram has no other native video type, and
        # shouldShareToFeed puts it in the grid as well.
        post_input["metadata"] = {
            "instagram": {"type": "reel" if video else "post", "shouldShareToFeed": True}
        }
        if video_asset or assets:
            post_input["assets"] = video_asset or [{"image": {"url": u}} for u in assets]
    elif platform == "linkedin" and (video_asset or assets):
        # Attach the media natively. With no URL in the LinkedIn body, this
        # renders as an image or video post with no link-preview card.
        post_input["assets"] = video_asset or [{"image": {"url": u}} for u in assets]
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


# ── Occupied slot detection ───────────────────────────────────────────────────


def _get_org_id() -> str:
    data = _buffer_gql("query { account { organizations { id } } }")
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        raise RuntimeError("No Buffer org found")
    return orgs[0]["id"]


def get_occupied_slots() -> set[str]:
    """Return date strings (YYYY-MM-DD) that already have a scheduled post with a dueAt set."""
    org_id = _get_org_id()
    occupied: set[str] = set()
    data = _buffer_gql(
        """
        query GetDrafts($input: PostsInput!) {
          posts(input: $input) {
            edges { node { id dueAt } }
          }
        }
        """,
        {"input": {"organizationId": org_id}},
    )
    for edge in data.get("data", {}).get("posts", {}).get("edges", []):
        due = (edge.get("node") or {}).get("dueAt")
        if due:
            try:
                dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                occupied.add(dt.date().isoformat())
            except Exception:
                pass
    return occupied


def fetch_buffer_posts(
    services: set[str] | None = None,
    statuses: tuple[str, ...] = ("draft",),
    since: datetime | None = None,
) -> list[dict]:
    """Return Buffer posts as {id, text, service, status, dueAt, sentAt}.

    Filters by ``statuses`` (default: drafts only) and, when ``services`` is given,
    by channel service (e.g. {"facebook"}). When ``since`` is given, keeps only
    posts that were *sent* on or after that time (``sentAt >= since``), used to
    pull the past week's published posts. Fetches a single page of up to 100
    posts, Buffer removed cursor pagination, which comfortably covers the
    current queue and a recent date window.
    """
    org_id = _get_org_id()
    data = _buffer_gql(
        """
        query GetPosts($input: PostsInput!) {
          posts(input: $input, first: 100) {
            edges { node { id text status dueAt sentAt channel { service } } }
          }
        }
        """,
        {"input": {"organizationId": org_id}},
    )
    out: list[dict] = []
    for edge in data.get("data", {}).get("posts", {}).get("edges", []):
        node = edge.get("node") or {}
        service = (node.get("channel") or {}).get("service")
        if statuses and node.get("status") not in statuses:
            continue
        if services and service not in services:
            continue
        sent_at = node.get("sentAt")
        if since is not None:
            if not sent_at:
                continue
            try:
                if datetime.fromisoformat(sent_at.replace("Z", "+00:00")) < since:
                    continue
            except Exception:
                continue
        out.append({
            "id":      node.get("id", ""),
            "text":    node.get("text", "") or "",
            "service": service,
            "status":  node.get("status"),
            "dueAt":   node.get("dueAt"),
            "sentAt":  sent_at,
        })
    return out


def fetch_top_performers(n: int = 3) -> list[dict]:
    """Return top n posts by engagement across Meta (Facebook + Instagram) and LinkedIn."""
    from .meta import fetch_meta_top_performers
    from .linkedin import fetch_linkedin_top_performers
    posts = fetch_meta_top_performers(n=n) + fetch_linkedin_top_performers(n=n)
    posts.sort(key=lambda p: p["engagement_score"], reverse=True)
    return posts[:n]


# ── Expired draft cleanup ─────────────────────────────────────────────────────


def _extract_earliest_date(text: str) -> datetime | None:
    today = datetime.now(tz=timezone.utc).date()
    earliest = None
    for match in _DATE_RE.finditer(text):
        # Strip ordinal suffixes ("July 3rd" -> "July 3") so strptime can parse;
        # the date regex matches them but %d cannot.
        raw = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", match.group(), flags=re.IGNORECASE)
        for fmt in (
            "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
            "%B %d", "%b %d",
            "%d %B %Y", "%d %b %Y", "%d %B", "%d %b",
            "%m/%d/%Y", "%m/%d/%y",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(raw.strip(), fmt)
                if dt.year == 1900:
                    dt = dt.replace(year=today.year)
                date = dt.date()
                if earliest is None or date < earliest:
                    earliest = date
                break
            except ValueError:
                continue
    return datetime(earliest.year, earliest.month, earliest.day, tzinfo=timezone.utc) if earliest else None


def is_show_announcement(text: str) -> bool:
    """True if the post reads as a live-show/ticket announcement (show keyword + a date).

    Same signal used to detect show drafts for expiry, minus the past-date check.
    Used to keep show announcements out of the website news section, which the
    client does not want there.
    """
    return bool(_SHOW_KEYWORDS.search(text)) and _extract_earliest_date(text) is not None


def is_future_show_announcement(text: str) -> bool:
    """True if the copy pitches attendance at a live date that has not happened yet.

    Stricter than :func:`is_show_announcement`, and the difference matters: this
    is the last-line guard that keeps show copy off LinkedIn (client direction
    2026-08-03), and LinkedIn is the channel already starved of content, so a
    false positive there costs a post. Buyer-proof copy cites past dates ("this
    venue has booked them three times") and legitimately uses words like "venue"
    and "show"; only a date still ahead of us means the post is selling tickets.
    """
    if not is_show_announcement(text):
        return False
    dt = _extract_earliest_date(text)
    return dt is not None and dt.date() >= datetime.now(tz=timezone.utc).date()


def _is_expired_show_announcement(text: str) -> bool:
    if not is_show_announcement(text):
        return False
    dt = _extract_earliest_date(text)
    return dt.date() < datetime.now(tz=timezone.utc).date()


def purge_expired_show_drafts(dry_run: bool = False) -> None:
    """Fetch all Buffer drafts and delete any expired show announcements."""
    try:
        org_id = _get_org_id()
    except RuntimeError:
        log.error("purge_expired_show_drafts: no Buffer org found")
        return

    data = _buffer_gql(
        """
        query GetDrafts($input: PostsInput!) {
          posts(input: $input) { edges { node { id text } } }
        }
        """,
        {"input": {"organizationId": org_id}},
    )
    posts = [
        e["node"]
        for e in data.get("data", {}).get("posts", {}).get("edges", [])
        if e.get("node")
    ]

    deleted = 0
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
                ... on DeletePostSuccess { id }
                ... on VoidMutationError { message }
              }
            }
            """,
            {"input": {"id": post["id"]}},
        ).get("data", {}).get("deletePost") or {}
        if "id" in result:
            log.info("Deleted expired draft %s", post["id"])
            deleted += 1
        else:
            log.error(
                "Failed to delete post %s: %s",
                post["id"],
                result.get("message") or "no response",
            )

    log.info(
        "purge_expired_show_drafts: %d post(s) %s",
        deleted,
        "would be deleted" if dry_run else "deleted",
    )


# ── API test ──────────────────────────────────────────────────────────────────


def test_buffer() -> None:
    """Print all connected Buffer channels and verify the API key works."""
    if not BUFFER_API_KEY:
        print("BUFFER_API_KEY not set in .env")
        sys.exit(1)

    data = _buffer_gql("query { account { organizations { id name } } }")
    errors = data.get("errors")
    if errors:
        print(f"Buffer API error: {errors}")
        sys.exit(1)
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        print("No organizations found. Check that the API key has the correct scopes.")
        sys.exit(1)

    org_id   = orgs[0]["id"]
    org_name = orgs[0].get("name", "")
    print(f"Connected, org: {org_name} ({org_id})\n")

    data = _buffer_gql(
        "query GetChannels($input: ChannelsInput!) { channels(input: $input) { id service displayName } }",
        {"input": {"organizationId": org_id}},
    )
    channels = data.get("data", {}).get("channels", [])
    if not channels:
        print("No channels found for this organization.")
        sys.exit(1)

    matched = []
    print(f"{len(channels)} channel(s) found:\n")
    for c in channels:
        service = c.get("service", "unknown").lower()
        name    = c.get("displayName", "")
        cid     = c.get("id", "")
        is_match = service in ("linkedin", "instagram", "facebook")
        marker   = "✓" if is_match else " "
        print(f"  [{marker}] {service:12s}  {name:30s}  id={cid}")
        if is_match:
            matched.append({"service": service, "id": cid, "name": name})

    print("\nChannels marked [✓] will be used by the content engine.")
    if not matched:
        print("\nNo matched channels, skipping draft test.")
        return

    test_text = "[LP Content Engine test] This is an automated test draft, safe to delete."
    print(f"\nCreating test draft on {len(matched)} channel(s)...\n")
    all_ok = True
    for c in matched:
        img = _IG_PLACEHOLDER if c["service"] == "instagram" else None
        ok  = post_draft_to_buffer(test_text, c["id"], platform=c["service"], image=img)
        print(f"  {c['service']:12s}  {c['name']:30s}  {'OK' if ok else 'FAILED'}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll test drafts created successfully. Check Buffer to confirm, then delete them.")
    else:
        print("\nOne or more drafts failed, check the error log above.")

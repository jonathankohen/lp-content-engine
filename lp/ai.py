import json
import logging
import re
from datetime import datetime, timezone

import anthropic
from anthropic.types import TextBlock

from . import config

log = logging.getLogger(__name__)


def search_artist_news(tribute: str, original: str) -> list[dict]:
    """Search for recent news about a tribute act (and optionally the original artist)."""
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(tribute):
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    original_line = (
        f" Also search for recent news about these original artists (the acts this tribute "
        f"represents): {original}. Search for the original artists themselves — NOT tribute "
        f"bands, cover acts, or spin-off/successor acts performing under a related name "
        f"(e.g. a deceased artist's backing band continuing to tour independently). "
        f"Only include news about the original named artist, not about associated acts "
        f"that are now separate performing entities."
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
        f"IMPORTANT: If the news is primarily about a specific show, concert date, or live event, "
        f"only include it if that show date is strictly in the future (after {today}). "
        f"Exclude reviews, recaps, or coverage of shows that have already happened. "
        f"Tour announcements mentioning multiple dates are acceptable if at least one future date exists. "
        "If no relevant news found in the last 14 days, return an empty array []. "
        "Do not include any text outside the JSON array."
    )

    config.claude_throttle()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        raw = client.messages.with_raw_response.create(
            model=config.SEARCH_MODEL,
            max_tokens=config.MAX_TOKENS,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        resp = raw.parse()
        config.claude_call_done(dict(raw.headers))
        config.track_cost(resp, config.SEARCH_MODEL)
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


_EXCLUSIVE_PRIORITIES = {"Top of Roster", "Exclusive"}
_EXCLUSIVE_ACTS = {"Tony Danza", "The Rocket Man Show"}
_SCORE_THRESHOLD = 0.40


def score_and_rank_topics(topics: list[dict]) -> list[dict]:
    """Score all candidate topics in one Haiku call, apply exclusivity bonus, return sorted best-first."""
    if not topics:
        return []
    if not config.under_cost_cap("topic scoring"):
        return topics

    numbered = "\n".join(
        f"{i}. Artist: {t.get('artist', '')} | Hook: {t.get('hook_type', '')} | "
        f"Headline: {t.get('headline', '')} | Summary: {t.get('summary', '')}"
        for i, t in enumerate(topics)
    )
    prompt = (
        "Score these news topics for Love Productions social media. "
        "Use these weighted criteria (weights sum to 1.0):\n"
        "- relevance (0.30): tribute_news about an LP act = 1.0, original_artist_news = 0.8, other = lower\n"
        "- freshness (0.30): published today = 1.0, ~6 months ago = 0.5, ~1 year+ = 0.0\n"
        "- velocity (0.10): how many sources discuss it; social media sources score higher\n"
        "- virality (0.10): potential to spread — celebrity angle, nostalgia, trending moment\n"
        "- uniqueness (0.20): interesting/non-generic; trivia and surprises beat routine announcements\n\n"
        f"Topics (0-indexed):\n{numbered}\n\n"
        "Return ONLY a JSON array, one entry per topic: [{\"index\": 0, \"score\": 0.75}, ...]. "
        "No other text."
    )

    config.claude_throttle()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        raw = client.messages.with_raw_response.create(
            model=config.SEARCH_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        resp = raw.parse()
        config.claude_call_done(dict(raw.headers))
        config.track_cost(resp, config.SEARCH_MODEL)
    except Exception as exc:
        log.error("Topic scoring error: %s — using original order", exc)
        return topics

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        log.warning("Could not parse scoring response — using original order")
        return topics
    try:
        scores = json.loads(match.group())
    except json.JSONDecodeError:
        log.warning("Scoring JSON parse error — using original order")
        return topics

    score_map = {
        item["index"]: item["score"]
        for item in scores
        if "index" in item and "score" in item
    }

    scored = []
    for i, t in enumerate(topics):
        base = score_map.get(i, 0.5)
        if base < _SCORE_THRESHOLD:
            log.info("Dropping low-score topic (%.2f): %s", base, t.get("headline", "")[:60])
            continue
        priority = t.get("_priority", "")
        artist = t.get("artist", "")
        if priority in _EXCLUSIVE_PRIORITIES or artist in _EXCLUSIVE_ACTS:
            base += 1.0
        t["_score"] = base
        scored.append(t)

    scored.sort(key=lambda t: t["_score"], reverse=True)
    log.info("Scored %d topic(s), %d passed threshold", len(topics), len(scored))
    return scored


def filter_new_topics(found: list[dict], used: set[str]) -> list[dict]:
    new = []
    for item in found:
        key = item.get("url", "").strip() or item.get("headline", "").strip()
        if key and key not in used:
            new.append(item)
    return new


def search_historical_facts(tribute: str, original: str, slot_date: datetime) -> list[dict]:
    """Search for a pre-1990 historical fact about the original artist for a given date."""
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(tribute):
        return []

    month_day = f"{slot_date.strftime('%B')} {slot_date.day}"
    prompt = (
        f"Search for an interesting historical fact about '{original}' to use in a social media post. "
        f"Ideally find something that happened on {month_day} in any year before 1990 "
        f"(e.g., a recording session, album release, chart milestone, interview quote, or news story). "
        f"Prioritize archival sources: archive.org, old Rolling Stone, Billboard, NME, "
        f"Melody Maker, Guitar World, Cashbox magazine. "
        f"If no compelling {month_day} fact exists, return the single most interesting "
        f"lesser-known fact about {original} from before 1990. "
        f"Return ONLY a JSON array with at most one object (or [] if nothing compelling is found). "
        f"The object must have exactly these keys: "
        f"headline (a concise 'On this day in [year], ...' style headline if date-specific, "
        f"otherwise a compelling fact headline), "
        f"url (most direct archival source URL available), "
        f"summary (1-2 sentences about the fact), "
        f"hook_type (always the string 'historical_fact'), "
        f"artist (always '{tribute}'). "
        f"Do not include any text outside the JSON array."
    )

    config.claude_throttle()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        raw = client.messages.with_raw_response.create(
            model=config.SEARCH_MODEL,
            max_tokens=config.MAX_TOKENS,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        resp = raw.parse()
        config.claude_call_done(dict(raw.headers))
        config.track_cost(resp, config.SEARCH_MODEL)
    except Exception as exc:
        log.error("Historical fact search error for %s: %s", tribute, exc)
        return []

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    for item in items:
        item.setdefault("original_artist", original)
    log.info("Found %d historical fact(s) for %s", len(items), tribute)
    return items[:1]


def format_performance_context(top_posts: list[dict]) -> str:
    """Format top-performing Buffer posts as style examples for Claude."""
    if not top_posts:
        return ""
    lines = ["Recent posts that performed well (study what made them effective):"]
    for p in top_posts:
        platform = (p.get("serviceType") or "").capitalize() or "Post"
        score = p.get("engagement_score", 0)
        text = (p.get("text") or "")[:200]
        lines.append(f"\n[{platform}] (engagement: {score})\n\"{text}\"")
    return "\n".join(lines)


def generate_posts(topic: dict, skill_graph: str, performance_context: str = "") -> dict | None:
    """Generate LinkedIn, Instagram, and Facebook posts for a topic."""
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(
        topic.get("headline", "")
    ):
        return None

    ticket_url = topic.get("ticket_url") or ""
    ticket_line = (
        f"Ticket URL: {ticket_url}\n"
        if ticket_url
        else "Ticket URL: not available\n"
    )
    perf_section = (
        f"\n{performance_context}\n"
        if performance_context
        else ""
    )
    user_prompt = (
        "Generate social media content for Love Productions based on this news topic:\n\n"
        f"Tribute Act: {topic.get('artist', '')}\n"
        f"Original Artist: {topic.get('original_artist', '') or 'N/A'}\n"
        f"Headline: {topic.get('headline', '')}\n"
        f"URL: {topic.get('url', '')}\n"
        f"{ticket_line}"
        f"Summary: {topic.get('summary', '')}\n"
        f"Suggested Hook Type: {topic.get('hook_type', '')}\n"
        f"{perf_section}\n"
        "Follow the content skill graph instructions exactly. Write all three platform posts "
        "in the repurposing chain order (LinkedIn first, then Instagram, then Facebook). "
        "Each post must think about the topic differently — not just reformatted.\n\n"
        "If a Ticket URL is provided, include it prominently in every platform post as the "
        "call-to-action link (e.g., 'Get tickets: <url>'). If not available, do not invent "
        "a link — omit ticket link entirely.\n\n"
        "Include the source URL in every post. For LinkedIn and Facebook, weave it naturally "
        "into the post body (e.g. 'Full story here: <url>' or 'Read more: <url>'). "
        "For Instagram, place it at the end of the caption before the hashtags.\n\n"
        "IMPORTANT: If the source article references a specific show date, venue, or performance "
        "that has already happened, do not mention that specific date or venue in any post. "
        "Write about the artist and their broader story instead. Never direct audiences to a "
        "past show or imply they can attend something that already occurred.\n\n"
        "Return ONLY a JSON object with these exact keys: linkedin, instagram, facebook. "
        "Each value is the full post text, ready to publish. No other text outside the JSON."
    )

    config.claude_throttle()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        raw = client.messages.with_raw_response.create(
            model=config.CONTENT_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=skill_graph,
            messages=[{"role": "user", "content": user_prompt}],
        )
        resp = raw.parse()
        config.claude_call_done(dict(raw.headers))
        config.track_cost(resp, config.CONTENT_MODEL)
    except Exception as exc:
        log.error("Content generation error for '%s': %s", topic.get("headline"), exc)
        return None

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        log.error("No JSON in content generation response for '%s'", topic.get("headline"))
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.error("JSON parse error in content generation: %s", exc)
        return None

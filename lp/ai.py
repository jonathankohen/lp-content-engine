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
        f"that are now separate performing entities. "
        f"HARD RULE: NEVER return an original artist's own tour, concert, show, residency, "
        f"festival appearance, or any live-performance announcement. We never promote the "
        f"original artist's live dates — only the tribute act's. For the original artists, "
        f"only return non-live news (album/release, award, biopic, anniversary, milestone, "
        f"passing, etc.) and set is_live_event to false for those items."
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
        "is_live_event (boolean: true if the news is primarily a concert, tour, show, "
        "residency, festival appearance, or other live-performance announcement), "
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

    # HARD RULE: never announce an original artist's own shows/tours. Drop any
    # original-artist item flagged as a live event, regardless of what the model
    # classified hook_type as. We only ever promote the tribute act's live dates.
    kept = []
    for item in items:
        item.setdefault("original_artist", original)
        is_original = item.get("hook_type") == "original_artist_news"
        if is_original and item.get("is_live_event"):
            log.info(
                "Dropping original-artist live-event item (never news): %s",
                item.get("headline", "")[:60],
            )
            continue
        kept.append(item)

    log.info("Found %d news items for %s", len(kept), tribute)
    return kept


_EXCLUSIVE_PRIORITIES = {"Top of Roster", "Exclusive"}
_EXCLUSIVE_ACTS = {"Tony Danza", "The Rocket Man Show"}
_SCORE_THRESHOLD = 0.40


def exclusivity_bonus(priority: str, artist: str) -> float:
    """+1.0 for Top of Roster / Exclusive acts (plus two hardcoded exceptions), else 0.0.

    Single source of truth for the exclusivity boost, applied to both scored news
    topics (below) and upcoming-show candidates (main.py) so the two compete fairly.
    """
    if priority in _EXCLUSIVE_PRIORITIES or artist in _EXCLUSIVE_ACTS:
        return 1.0
    return 0.0


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
        base += exclusivity_bonus(priority, artist)
        t["_score"] = base
        scored.append(t)

    scored.sort(key=lambda t: t["_score"], reverse=True)
    log.info("Scored %d topic(s), %d passed threshold", len(topics), len(scored))
    return scored


def classify_show_announcements(texts: list[str]) -> list[bool]:
    """Flag which posts are live-event/show announcements (one Haiku call for the batch).

    Used by the website-news backfill, where reconstructed topics carry no
    hook_type. A post is a show announcement if its primary purpose is to promote
    attendance at a specific upcoming live event at a venue on a date — a concert,
    show, residency, festival, or personal/live appearance. General artist news
    that merely mentions the tribute act's name or a release date is NOT a show
    announcement. On any failure, returns all-False (nothing skipped) so the
    backfill degrades to its prior behaviour rather than silently dropping news.
    """
    if not texts:
        return []
    if not config.under_cost_cap("show classification"):
        return [False] * len(texts)

    numbered = "\n\n".join(f"[{i}]\n{t}" for i, t in enumerate(texts))
    prompt = (
        "Each item below is a social media post. Decide, for each, whether it is a "
        "SHOW ANNOUNCEMENT: a post whose primary purpose is to promote attendance at "
        "a specific upcoming live event at a venue on a date — a concert, show, "
        "residency, festival, or a personal/live appearance.\n"
        "A post is NOT a show announcement if it is general news (a new release, "
        "award, anniversary, interview, obituary, trivia, etc.), even if it mentions "
        "a tribute act's name, the word 'concert', or a release date.\n\n"
        f"Posts (0-indexed):\n{numbered}\n\n"
        "Return ONLY a JSON array of objects, one per post, like "
        "[{\"index\": 0, \"show\": true}, ...]. No other text."
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
        log.error("Show classification error: %s — treating all as non-shows", exc)
        return [False] * len(texts)

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        log.warning("Could not parse show classification — treating all as non-shows")
        return [False] * len(texts)
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        log.warning("Show classification JSON parse error — treating all as non-shows")
        return [False] * len(texts)

    flags = [False] * len(texts)
    for item in items:
        i = item.get("index")
        if isinstance(i, int) and 0 <= i < len(texts):
            flags[i] = bool(item.get("show"))
    return flags


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


def search_trivia(tribute: str, original: str) -> list[dict]:
    """Search for a surprising piece of trivia about the original artist (any era, not date-bound)."""
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(tribute):
        return []

    prompt = (
        f"Search for a surprising, lesser-known piece of trivia about '{original}' to use in a "
        f"social media post for a tribute-act booking agency. "
        f"This is NOT breaking news and NOT tied to any particular date — find a genuinely "
        f"interesting fact from any era of the artist's career that a casual fan would not know "
        f"(e.g. an unusual recording story, the hidden meaning behind a song, a record they hold, "
        f"a surprising collaboration, an odd job before fame, a quirky stage habit). "
        f"Prioritize facts that are fun, share-worthy, and spark an 'I didn't know that' reaction. "
        f"Avoid anything already widely repeated as a cliché. "
        f"Return ONLY a JSON array with at most one object (or [] if nothing compelling is found). "
        f"The object must have exactly these keys: "
        f"headline (a punchy, curiosity-driving headline for the trivia), "
        f"url (most direct source URL available), "
        f"summary (1-2 sentences stating the fact), "
        f"hook_type (always the string 'trivia'), "
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
        log.error("Trivia search error for %s: %s", tribute, exc)
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
    log.info("Found %d trivia item(s) for %s", len(items), tribute)
    return items[:1]


def format_performance_context(top_posts: list[dict]) -> str:
    """Format top-performing Buffer posts as style examples for Claude."""
    if not top_posts:
        return ""
    lines = ["Recent posts that performed well (study what made them effective):"]
    for p in top_posts:
        platform = p.get("platform") or (p.get("serviceType") or "").capitalize() or "Post"
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

    url = topic.get("url", "")
    url_line = f"URL: {url}\n" if url else ""
    source_url_instruction = (
        "Include the source URL in the Facebook post only — weave it naturally "
        "into the post body (e.g. 'Full story here: <url>' or 'Read more: <url>'). "
        "Do NOT include any raw URL in the LinkedIn or Instagram posts (LinkedIn is an "
        "image-only post and must not contain a link).\n\n"
    ) if url else ""
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
    # Tribute act name: prefer the Airtable act (_act) since for original-artist
    # news the `artist` field can be the original artist.
    tribute = (topic.get("_act") or topic.get("artist", "")).strip()
    tribute_mention_instruction = (
        f"IMPORTANT: Every platform post MUST mention the tribute act by name "
        f"('{tribute}') at least once — even for original-artist news, trivia, or "
        f"historical facts, tie the story back to {tribute}. (Social posts cannot "
        f"hyperlink a name; just name the act in the copy.)\n\n"
        if tribute
        else ""
    )
    user_prompt = (
        "Generate social media content for Love Productions based on this news topic:\n\n"
        f"Tribute Act: {tribute}\n"
        f"Original Artist: {topic.get('original_artist', '') or 'N/A'}\n"
        f"Headline: {topic.get('headline', '')}\n"
        f"{url_line}"
        f"{ticket_line}"
        f"Summary: {topic.get('summary', '')}\n"
        f"Suggested Hook Type: {topic.get('hook_type', '')}\n"
        f"{perf_section}\n"
        "Follow the content skill graph instructions exactly. Write all three platform posts "
        "in the repurposing chain order (LinkedIn first, then Instagram, then Facebook). "
        "Each post must think about the topic differently — not just reformatted.\n\n"
        f"{tribute_mention_instruction}"
        "If a Ticket URL is provided, include it prominently in the Facebook post only "
        "as the call-to-action link (e.g., 'Get tickets: <url>'). Do NOT include the ticket link "
        "in the LinkedIn or Instagram posts. If not available, do not invent a link — omit entirely.\n\n"
        f"{source_url_instruction}"
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


# ── Website news posts ────────────────────────────────────────────────────────

# The only categories the LP News WordPress plugin accepts. Claude must choose
# from this list; anything else is dropped server-side.
NEWS_CATEGORIES = [
    "Celebration",
    "Celebrity",
    "Theatre",
    "Tour",
    "Condolences",
    "Festival",
    "Interview",
    "Sold Out",
    "Tribute",
    "TV Show",
    "Uncategorized",
]

# Deterministic starting category per hook type. Claude then verifies and
# adds/removes to fit the actual story (e.g. an obituary → Condolences, an
# interview piece → Interview, a sold-out show → Sold Out).
_HOOK_CATEGORY_DEFAULTS = {
    "tribute_news": ["Tribute"],
    "original_artist_news": ["Celebrity"],
    "upcoming_show": ["Tour"],
    "trivia": ["Tribute"],
    "historical_fact": ["Celebrity"],
}


def default_categories(hook_type: str) -> list[str]:
    """Deterministic default category list for a topic's hook type."""
    return list(_HOOK_CATEGORY_DEFAULTS.get(hook_type, ["Uncategorized"]))


def generate_article(
    topic: dict,
    skill_graph: str,
    default_cats: list[str] | None = None,
    artist_url: str = "",
    appointment_url: str = "",
) -> dict | None:
    """Generate a website news article for a topic.

    Returns {"title": str, "body": str, "categories": list[str]} or None. The
    body is a few short paragraphs of prose in LP brand voice. The only links it
    may contain are two inline HTML anchors — the tribute act's name (linked to
    ``artist_url``, its loveproductions.com page) and "Steve Love" in the closing
    booking CTA (linked to ``appointment_url``) — no other links or raw URLs; the
    "Read more" button carries the source link. Categories are chosen from
    NEWS_CATEGORIES, seeded by ``default_cats`` and adjusted by Claude to fit the
    story. Gated by the same cost cap as generate_posts().
    """
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(
        topic.get("headline", "")
    ):
        return None

    # The tribute act name: prefer the Airtable act (_act) since for
    # original-artist news the `artist` field can be the original artist.
    tribute = (topic.get("_act") or topic.get("artist", "")).strip()
    tribute_anchor = (
        f'<a href="{artist_url}">{tribute}</a>' if (tribute and artist_url) else tribute
    )
    this_link_anchor = (
        f'<a href="{appointment_url}">this link</a>' if appointment_url else "this link"
    )
    tie_in_instruction = ""
    if tribute:
        link_rule = (
            f"EVERY time the act's name '{tribute}' appears in the body — including in "
            f"the booking sentence below — render it EXACTLY as this HTML hyperlink: "
            f"{tribute_anchor} (never as plain text).\n"
            if artist_url
            else f"Mention the act by name ('{tribute}').\n"
        )
        tie_in_instruction = (
            "\nTRIBUTE ACT TIE-IN AND BOOKING CTA (both are REQUIRED):\n"
            f"- The article MUST mention {tribute}, Love Productions' tribute act, at "
            "least once, connecting the news to the act naturally (e.g. how the act "
            "carries this artist's music/spirit to stages today). "
            f"{link_rule}"
            "- End the article with a short closing paragraph inviting bookings, "
            f'phrased like: "If you\'re interested in booking {tribute}, please follow '
            f'{this_link_anchor} to set up an appointment with Steve Love." Use that '
            'exact HTML hyperlink on the words "this link"; Steve Love stays plain text.\n'
            "- The act-name hyperlink(s) and the \"this link\" anchor are the ONLY links "
            "allowed anywhere in the body.\n"
        )

    seed = default_cats if default_cats is not None else default_categories(topic.get("hook_type", ""))
    user_prompt = (
        "Write a short website news article for the Love Productions site "
        "(loveproductions.com) based on this topic:\n\n"
        f"Tribute Act: {tribute}\n"
        f"Original Artist: {topic.get('original_artist', '') or 'N/A'}\n"
        f"Headline: {topic.get('headline', '')}\n"
        f"Summary: {topic.get('summary', '')}\n"
        f"Hook Type: {topic.get('hook_type', '')}\n\n"
        "Follow the content skill graph (voice, banned words, humanizer rules, "
        "one em dash maximum). This is a website article, NOT a social caption: "
        "write 2–4 short paragraphs of flowing prose. Do NOT include hashtags or "
        "emoji. Do NOT include any visible raw URL — the ONLY links permitted are "
        "the two inline HTML anchors described below; a 'Read more' button handles "
        "the source link separately.\n"
        f"{tie_in_instruction}"
        "If the story announces an upcoming show, performance, or event, you MUST "
        "state the show date explicitly in the article (and the city/venue when "
        "known). Do NOT, however, point readers to a specific show, date, or venue "
        "that has ALREADY occurred — for past events, write about the artist and "
        "their broader story instead.\n"
        "Aside from the required booking CTA above, do NOT add any other "
        "call-to-action (no 'visit loveproductions.com', 'click the button/see "
        "below', or 'full event details here' for tickets or the venue). The "
        "booking invitation is the only permitted closing directive.\n\n"
        "Also choose 1–3 categories that best fit this story. You MUST pick only "
        f"from this exact list: {', '.join(NEWS_CATEGORIES)}.\n"
        f"Suggested starting point (adjust as the story warrants): {', '.join(seed)}.\n\n"
        "Return ONLY a JSON object with these exact keys: title, body, categories. "
        "`title` is the article headline (concise, no clickbait). `body` is the full "
        "article text with paragraphs separated by blank lines. `categories` is an "
        "array of strings drawn only from the list above. No text outside the JSON."
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
        log.error("Article generation error for '%s': %s", topic.get("headline"), exc)
        return None

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        log.error("No JSON in article generation response for '%s'", topic.get("headline"))
        return None
    try:
        article = json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.error("JSON parse error in article generation: %s", exc)
        return None

    # Keep only allowed categories (case-insensitive); fall back to the seed.
    allowed = {c.lower(): c for c in NEWS_CATEGORIES}
    cats = []
    for c in article.get("categories", []) or []:
        canonical = allowed.get(str(c).strip().lower())
        if canonical and canonical not in cats:
            cats.append(canonical)
    if not cats:
        cats = seed or ["Uncategorized"]
    article["categories"] = cats
    article["title"] = (article.get("title") or topic.get("headline", "")).strip()
    article["body"] = (article.get("body") or topic.get("summary", "")).strip()
    return article

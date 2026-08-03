import json
import logging
import re
from datetime import datetime, timezone

import anthropic
from anthropic.types import TextBlock

from . import config
from .scrape import extract_page_quotes, fetch_page_prose, verify_quote_on_page

log = logging.getLogger(__name__)


_DASH_RE = re.compile(r"\s*[—–]\s*")


def strip_dashes(text: str) -> str:
    """Remove every em and en dash, replacing each with real punctuation.

    Em dashes are banned outright by the client, and asking the model not to use
    them is not enough: a dry run on 2026-07-31 produced seven across seven
    posts despite the rule appearing in both the skill graph and the prompt. So
    the guarantee is enforced here instead of hoped for.

    A dash joining a clause to a following capitalised word is doing the job of
    a full stop ("Winter Garden, FL—The Rocket Man Show opens..."), so it becomes
    one. Everywhere else it is standing in for a comma. This is the last thing
    that touches post copy, so nothing downstream can reintroduce one.
    """
    def replace(match: re.Match) -> str:
        after = text[match.end():match.end() + 1]
        return ". " if after.isupper() else ", "

    return _DASH_RE.sub(replace, text or "")


def _quarter_key(prefix: str) -> str:
    """A dedup key stamped with the current quarter, e.g. 'agency_2026Q3'.

    Topics keyed this way are automatically capped at once per quarter by the
    existing Google Sheets dedup, with no extra state to maintain.
    """
    now = datetime.now(timezone.utc)
    return f"{prefix}_{now.year}Q{(now.month - 1) // 3 + 1}"


def search_artist_news(tribute: str, original: str) -> list[dict]:
    """Search for recent news about a tribute act (and optionally the original artist)."""
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(tribute):
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    original_line = (
        f" Also search for recent news about these original artists (the acts this tribute "
        f"represents): {original}. Search for the original artists themselves, NOT tribute "
        f"bands, cover acts, or spin-off/successor acts performing under a related name "
        f"(e.g. a deceased artist's backing band continuing to tour independently). "
        f"Only include news about the original named artist, not about associated acts "
        f"that are now separate performing entities. "
        f"HARD RULE: NEVER return an original artist's own tour, concert, show, residency, "
        f"festival appearance, or any live-performance announcement. We never promote the "
        f"original artist's live dates, only the tribute act's. For the original artists, "
        f"only return non-live news (album/release, award, biopic, anniversary, milestone, "
        f"passing, etc.) and set is_live_event to false for those items."
        if original
        else ""
    )

    prompt = (
        f"Search for news articles published in the last 14 days about '{tribute}'. "
        f"IMPORTANT: Search for the exact act name '{tribute}' only, ignore any results "
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
        config.record_api_exception(exc, "News search")
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
        "- virality (0.10): potential to spread, celebrity angle, nostalgia, trending moment\n"
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
        config.record_api_exception(exc, "Topic scoring")
        log.error("Topic scoring error: %s, using original order", exc)
        return topics

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        log.warning("Could not parse scoring response, using original order")
        return topics
    try:
        scores = json.loads(match.group())
    except json.JSONDecodeError:
        log.warning("Scoring JSON parse error, using original order")
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
    attendance at a specific upcoming live event at a venue on a date, a concert,
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
        "a specific upcoming live event at a venue on a date, a concert, show, "
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
        config.record_api_exception(exc, "Show classification")
        log.error("Show classification error: %s, treating all as non-shows", exc)
        return [False] * len(texts)

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        log.warning("Could not parse show classification, treating all as non-shows")
        return [False] * len(texts)
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        log.warning("Show classification JSON parse error, treating all as non-shows")
        return [False] * len(texts)

    flags = [False] * len(texts)
    for item in items:
        i = item.get("index")
        if isinstance(i, int) and 0 <= i < len(texts):
            flags[i] = bool(item.get("show"))
    return flags


def topic_key(item: dict) -> str:
    """The dedup key for a topic, as stored in the url column of the Sheets log.

    An explicit ``sheet_key`` wins so synthesized topics can control their own
    cadence (a quarter-stamped spotlight key caps that act at one spotlight per
    quarter; a rebooking key fires once per act/venue pairing, ever). Searched
    topics have no sheet_key and fall back to their source URL, then headline.
    """
    return (
        (item.get("sheet_key") or "").strip()
        or (item.get("url") or "").strip()
        or (item.get("headline") or "").strip()
    )


def filter_new_topics(found: list[dict], used: set[str]) -> list[dict]:
    new = []
    for item in found:
        key = topic_key(item)
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
        config.record_api_exception(exc, "Historical fact search")
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
        f"This is NOT breaking news and NOT tied to any particular date, find a genuinely "
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
        config.record_api_exception(exc, "Trivia search")
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


def search_testimonials(tribute: str, original: str = "") -> list[dict]:
    """Search for a genuine, published quote about a tribute act.

    Testimonials are the highest-converting buyer content (see
    ``audience/buyers.md``), but only the small subset that has been published
    is findable this way. Anything without a verifiable source is dropped:
    every candidate must carry a quote, an attribution and a URL, and the quote
    must be confirmed present on the cited page by
    :func:`lp.scrape.verify_quote_on_page`. A model asked for a real quote will
    occasionally reconstruct a plausible one, so the check is code-level, not
    prompt-level.
    """
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(tribute):
        return []

    prompt = (
        f"Search for a genuine, publicly published quote praising the tribute act "
        f"'{tribute}', for use in a social media post by their booking agency. "
        f"Acceptable sources: a venue, promoter, talent buyer, festival, or a published "
        f"review in a newspaper, magazine or entertainment outlet. "
        f"The quote must be about '{tribute}' the tribute act itself, NOT about the "
        f"original artist they perform the music of. "
        f"CRITICAL: return the quote verbatim, exactly as it appears on the page you "
        f"found it on. Do NOT invent, paraphrase, translate, tidy up, or reconstruct a "
        f"quote from memory. If you cannot find a real published quote with a working "
        f"source URL, return an empty array. An empty array is the correct and expected "
        f"answer most of the time. Do not settle for a marketing blurb written by the act "
        f"or the agency itself. "
        f"Return ONLY a JSON array with at most one object (or [] if nothing is found). "
        f"The object must have exactly these keys: "
        f"headline (a short buyer-facing headline for the praise), "
        f"url (the exact page the quote appears on), "
        f"quote (the verbatim quote, no surrounding quotation marks), "
        f"attribution (who said or published it, e.g. 'The Boston Globe' or "
        f"'Ridgefield Playhouse'), "
        f"summary (1-2 sentences of context around the quote), "
        f"hook_type (always the string 'testimonial'), "
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
        config.record_api_exception(exc, "Testimonial search")
        log.error("Testimonial search error for %s: %s", tribute, exc)
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

    verified = []
    for item in items:
        quote = (item.get("quote") or "").strip()
        url = (item.get("url") or "").strip()
        attribution = (item.get("attribution") or "").strip()
        if not (quote and url and attribution):
            log.info("Testimonial for %s dropped: missing quote, source or attribution", tribute)
            continue
        if not verify_quote_on_page(url, quote):
            log.info("Testimonial for %s dropped: quote not found at %s", tribute, url)
            continue
        item["hook_type"] = "testimonial"
        item["artist"] = tribute
        item.setdefault("original_artist", original)
        # The verified quote and its source are what the post must be built on,
        # so put them where generate_posts() will see them.
        item["summary"] = (
            f'{attribution} on {tribute}: "{quote}". {item.get("summary", "")}'.strip()
        )
        verified.append(item)

    log.info(
        "Found %d verified testimonial(s) for %s (%d candidate(s) searched)",
        len(verified), tribute, len(items),
    )
    return verified[:1]


def build_act_spotlight_topic(artist: dict) -> dict | None:
    """Build an evergreen buyer-facing spotlight topic from an act's own LP page.

    No search call: the act's loveproductions.com page (``artist_url``, backed by
    the static map in ``lp/artist_links.py``) already carries the credentials a
    talent buyer wants. The dedup key is quarter-stamped so the existing Sheets
    dedup caps each act at one spotlight per quarter. Returns None when the page
    yields no usable copy.
    """
    name = (artist.get("name") or "").strip()
    url = (artist.get("artist_url") or "").strip()
    if not name or not url:
        return None

    prose = fetch_page_prose(url)
    if len(prose) < 200:
        log.info("Spotlight skipped for %s: no usable copy at %s", name, url)
        return None

    return {
        "artist":          name,
        "original_artist": artist.get("original_artist", ""),
        "headline":        f"Act spotlight: {name}",
        "url":             url,
        "sheet_key":       _quarter_key(f"spotlight_{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}"),
        "summary": (
            f"Buyer-facing spotlight on {name}, a Love Productions act. Credentials and "
            f"background from the act's own page:\n\n{prose}"
        ),
        "hook_type":  "act_spotlight",
        "ticket_url": None,
        "_act":       name,
    }


def build_page_testimonial_topics(artist: dict) -> list[dict]:
    """Build testimonial topics from praise already published on the act's LP page.

    Twelve of the roster's act pages carry attributed pull-quotes: buyer praise
    from venues and festivals, and press quotes from outlets. The agency
    published them itself, so they are pre-cleared, and because the page is the
    source they need no ``verify_quote_on_page()`` round trip. That makes this a
    strictly better testimonial source than :func:`search_testimonials`, which
    costs a web-search call and correctly returns nothing most of the time.
    Callers should try this first and fall back to the search.

    Buyer quotes are returned ahead of press quotes: a venue saying it went well
    answers the booking question directly, where a review answers it obliquely.
    The dedup key is per quote (not per act or per quarter), so each individual
    quote is used once, ever, and an act with several quotes can appear again in
    a later run with a different one.
    """
    name = (artist.get("name") or "").strip()
    url = (artist.get("artist_url") or "").strip()
    if not name or not url:
        return []

    topics = []
    for quote in extract_page_quotes(url):
        slug = re.sub(r"[^a-z0-9]+", "-", quote["quote"][:60].lower()).strip("-")
        topics.append({
            "artist":          name,
            "original_artist": artist.get("original_artist", ""),
            "headline":        f"What {quote['attribution']} said about {name}",
            "url":             url,
            "sheet_key":       f"pagequote_{slug}",
            "quote":           quote["quote"],
            "attribution":     quote["attribution"],
            "summary": (
                f"Published praise for {name}, a Love Productions act, quoted on the "
                f"act's own page. Said by {quote['attribution']}."
            ),
            "hook_type":    "testimonial",
            "ticket_url":   None,
            "_act":         name,
            "_source_type": quote["source_type"],
        })

    topics.sort(key=lambda t: t["_source_type"] != "buyer")
    return topics


def build_agency_topic() -> dict:
    """Build a Love Productions institutional credibility topic (LinkedIn only).

    No search and no scrape: the agency's verified credentials live in
    ``content-skill-graph/engine/agency-facts.md``, which is already part of the
    system prompt for every generation call. Quarter-stamped dedup key caps this
    at one per quarter.
    """
    return {
        "artist":          "Love Productions",
        "original_artist": "",
        "headline":        "Love Productions: who we are and what we book",
        "url":             config.LP_HOMEPAGE,
        "sheet_key":       _quarter_key("agency"),
        "summary": (
            "An institutional post about Love Productions itself, aimed at talent buyers. "
            "Use ONLY the verified credentials in the agency facts file of the skill graph. "
            "Do not invent numbers, dates, client names or claims."
        ),
        "hook_type":  "agency_proof",
        "ticket_url": None,
        "_act":       "",
    }


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
        "Include the source URL in the Facebook post only, weave it naturally "
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
    # A testimonial's whole value is the verified quote, so state it as its own
    # instruction rather than leaving it buried in the summary, where the model
    # tends to paraphrase it away or drop it entirely.
    quote = (topic.get("quote") or "").strip()
    quote_instruction = (
        f"THIS IS A TESTIMONIAL POST. The quote below has been verified word for word "
        f"against its source. Build every platform post around it.\n"
        f'  Quote: "{quote}"\n'
        f"  Said by: {topic.get('attribution', '')}\n"
        f"Reproduce the quote EXACTLY as written above, inside quotation marks, and name "
        f"who said it. Do not paraphrase it, shorten it, extend it, fix its grammar, or "
        f"re-attribute it. If it needs trimming to fit, cut from the end and nothing else. "
        f"Your own commentary is one sentence at most, the quote does the work.\n\n"
        if quote
        else ""
    )
    # Tribute act name: prefer the Airtable act (_act) since for original-artist
    # news the `artist` field can be the original artist.
    tribute = (topic.get("_act") or topic.get("artist", "")).strip()
    tribute_mention_instruction = (
        f"IMPORTANT: Every platform post MUST mention the tribute act by name "
        f"('{tribute}') at least once, even for original-artist news, trivia, or "
        f"historical facts, tie the story back to {tribute}. (Social posts cannot "
        f"hyperlink a name; just name the act in the copy.)\n\n"
        if tribute
        else ""
    )
    # LinkedIn is written to talent buyers, so its CTA is a booking appointment
    # rather than an email: a lower bar than composing a message, and the client
    # asked for the calendar link specifically (2026-07-31). The model cannot use
    # a URL it was never given, so it is passed in here rather than left to the
    # skill graph, which only describes the rule.
    booking_cta_instruction = (
        f"The LinkedIn booking CTA must be a single short line pointing at this "
        f"calendar link: {config.STEVE_CALENDAR_LINK}\n"
        f"Vary the wording ('Booking: <link>' / 'Availability: <link>' / "
        f"'Book a time with Steve Love: <link>'). Use the link exactly as given. "
        f"Do NOT also include an email address; the link replaces it. This is the "
        f"only URL allowed in the LinkedIn post.\n\n"
        if config.STEVE_CALENDAR_LINK
        else
        "End the LinkedIn post with a single short booking line using "
        "info@loveproductions.com. One line, never a paragraph.\n\n"
    )
    # Instagram cannot linkify a caption, so a URL there is dead text and an
    # email address asks for more effort than a DM. Client direction 2026-08-03:
    # keep Instagram to DM plus link in bio, nothing else.
    instagram_cta_instruction = (
        "End the Instagram caption with a booking CTA of exactly this shape: "
        "a DM ask, then link in bio. For example \"DM us for booking availability. "
        "Link in bio for more.\" Vary the wording slightly, but never put a URL or "
        "an email address in an Instagram caption: captions do not linkify, so a "
        "link is dead text, and the email is a higher bar than a DM.\n\n"
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
        "Each post must think about the topic differently, not just reformatted.\n\n"
        "KEEP IT SHORT AND HUMAN. This is the single most important instruction. The client "
        "rejected a 130-word LinkedIn post as sounding AI-written, and the post they asked us "
        "to imitate is 17 words long.\n"
        "LinkedIn: 200 to 300 characters of copy, 400 absolute maximum. The booking link does not count toward that budget. Two or three sentences plus a "
        "one-line booking CTA. That is the entire post. Do NOT write paragraphs. Do NOT write a "
        "closing sentence that summarises what you just said. Do NOT explain why a fact matters.\n"
        "Instagram: punchy, a few short lines. Facebook: warm and conversational, still tight.\n"
        "Lead with the single hardest fact you have (a venue, an award, a number, a year), in "
        "the first line. One fact per post, not a stack. Vary your sentence length. Cut every "
        "adjective and check the post still stands; if it collapses, it was fluff. "
        "If a post could describe any act, rewrite it so it could only describe this one.\n\n"
        f"{booking_cta_instruction}"
        f"{instagram_cta_instruction}"
        f"{tribute_mention_instruction}"
        f"{quote_instruction}"
        "If a Ticket URL is provided, include it prominently in the Facebook post only "
        "as the call-to-action link (e.g., 'Get tickets: <url>'). Do NOT include the ticket link "
        "in the LinkedIn or Instagram posts. If not available, do not invent a link, omit entirely.\n\n"
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
        config.record_api_exception(exc, "Content generation")
        log.error("Content generation error for '%s': %s", topic.get("headline"), exc)
        return None

    text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        log.error("No JSON in content generation response for '%s'", topic.get("headline"))
        return None
    try:
        posts = json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.error("JSON parse error in content generation: %s", exc)
        return None

    # Hashtags are guaranteed here rather than asked for in the prompt: the rule
    # has been in instagram.md all along and the model honoured it on 2 of 8
    # posts in a real run.
    if isinstance(posts, dict) and posts.get("instagram"):
        posts["instagram"] = ensure_hashtags(posts["instagram"], topic)
    return posts


# ── Instagram hashtags ────────────────────────────────────────────────────────

_MIN_HASHTAGS = 3
_MAX_HASHTAGS = 7
_HASHTAG_RE = re.compile(r"#\w+")

# Evergreen tags by hook type, appended after the act and original-artist tags.
# Kept small and true: a wall of generic tags reads as spam and none of these
# should claim something the post does not contain.
_HOOK_HASHTAGS = {
    "upcoming_show":  ["LiveMusic", "TributeBand", "ConcertNight"],
    "tour_poster":    ["OnTour", "LiveMusic", "TributeBands", "LiveEntertainment"],
    "act_video":      ["LiveMusic", "TributeBand", "LivePerformance"],
    "rebooking":      ["LiveMusic", "TributeBand", "NowBooking"],
    "testimonial":    ["LiveMusic", "TributeBand", "NowBooking"],
    "act_spotlight":  ["LiveMusic", "TributeBand", "NowBooking"],
    "trivia":         ["MusicHistory", "LiveMusic", "TributeBand"],
    "historical_fact": ["MusicHistory", "LiveMusic", "TributeBand"],
}
_DEFAULT_HASHTAGS = ["LiveMusic", "TributeBand", "LiveEntertainment"]


def _to_hashtag(name: str) -> str:
    """"Arrival From Sweden" -> "#ArrivalFromSweden". Empty string if unusable."""
    # Drop anything after a colon or dash: the tail is usually a descriptor
    # ("...: The Music of ABBA") that makes an unreadable tag on its own.
    head = re.split(r"[:\-]", name or "", 1)[0]
    words = [w for w in re.split(r"[^A-Za-z0-9]+", head) if w]
    if not words:
        return ""
    tag = "".join(w[0].upper() + w[1:] for w in words)
    return f"#{tag}" if 2 < len(tag) <= 30 else ""


def build_hashtags(topic: dict, limit: int = _MAX_HASHTAGS) -> list[str]:
    """Deterministic hashtag set for a topic: the act, the original artist, then
    evergreen tags for the hook type. Deduped case-insensitively."""
    from .artist_links import display_act  # local: avoids a circular import

    act = (topic.get("_act") or topic.get("artist") or "").strip()
    tags, seen = [], set()

    def add(tag: str) -> None:
        if tag and tag.lower() not in seen and len(tags) < limit:
            seen.add(tag.lower())
            tags.append(tag)

    add(_to_hashtag(display_act(act) if act else ""))
    for original in re.split(r",|&|/| and ", topic.get("original_artist", "") or ""):
        add(_to_hashtag(original.strip()))
    for word in _HOOK_HASHTAGS.get(topic.get("hook_type", ""), _DEFAULT_HASHTAGS):
        add(f"#{word}")
    return tags


def ensure_hashtags(text: str, topic: dict) -> str:
    """Guarantee an Instagram caption ends with hashtags.

    ``instagram.md`` has asked for 5 to 10 hashtags all along and the model
    supplied them on 2 of 8 posts in a real run. This is the same lesson as
    ``strip_dashes``: a rule the copy must satisfy every time belongs in code,
    not in a prompt. Captions that already carry enough tags are returned
    untouched, so the model's own (better, more specific) tags always win.
    """
    text = (text or "").rstrip()
    if not text or len(_HASHTAG_RE.findall(text)) >= _MIN_HASHTAGS:
        return text
    existing = {t.lower() for t in _HASHTAG_RE.findall(text)}
    fresh = [t for t in build_hashtags(topic) if t.lower() not in existing]
    return f"{text}\n\n{' '.join(fresh)}" if fresh else text


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
    "rebooking": ["Celebration"],
    "testimonial": ["Tribute"],
    "act_spotlight": ["Tribute"],
}


def default_categories(hook_type: str) -> list[str]:
    """Deterministic default category list for a topic's hook type."""
    return list(_HOOK_CATEGORY_DEFAULTS.get(hook_type, ["Uncategorized"]))


def generate_article(
    topic: dict,
    skill_graph: str,
    default_cats: list[str] | None = None,
    appointment_url: str = "",
) -> dict | None:
    """Generate a website news article for a topic.

    Returns {"title": str, "body": str, "categories": list[str]} or None. The
    body is a few short paragraphs of prose in LP brand voice. The only link it
    may contain is one inline HTML anchor, "this link" in the closing booking
    CTA (pointed at ``appointment_url``, Steve's calendar). The article does NOT
    link the tribute act's name or the source (the "Read more" button carries the
    source link, and the reader is already on loveproductions.com). Categories are
    chosen from NEWS_CATEGORIES, seeded by ``default_cats`` and adjusted by Claude
    to fit the story. Gated by the same cost cap as generate_posts().
    """
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(
        topic.get("headline", "")
    ):
        return None

    # The tribute act name: prefer the Airtable act (_act) since for
    # original-artist news the `artist` field can be the original artist.
    tribute = (topic.get("_act") or topic.get("artist", "")).strip()
    this_link_anchor = (
        f'<a href="{appointment_url}">this link</a>' if appointment_url else "this link"
    )
    tie_in_instruction = ""
    if tribute:
        tie_in_instruction = (
            "\nTRIBUTE ACT TIE-IN AND BOOKING CTA (both are REQUIRED):\n"
            f"- The article MUST mention {tribute}, Love Productions' tribute act, at "
            "least once, connecting the news to the act naturally (e.g. how the act "
            "carries this artist's music/spirit to stages today). Write the act's name "
            "as plain text, do NOT hyperlink it.\n"
            "- End the article with a short closing paragraph inviting bookings, "
            f'phrased like: "If you\'re interested in booking {tribute}, please follow '
            f'{this_link_anchor} to set up an appointment with Steve Love." Use that '
            'exact HTML hyperlink on the words "this link"; Steve Love stays plain text.\n'
            "- The \"this link\" anchor is the ONLY link allowed anywhere in the body.\n"
        )

    # As in generate_posts(): a verified testimonial quote must survive into the
    # copy verbatim, so it gets its own instruction rather than sitting in summary.
    article_quote = (topic.get("quote") or "").strip()
    article_quote_instruction = (
        f"This article is built on a verified quote. Include it verbatim, in quotation "
        f"marks, attributed to who said it. Do not paraphrase, shorten, extend, or "
        f"re-attribute it.\n"
        f'  Quote: "{article_quote}"\n'
        f"  Said by: {topic.get('attribution', '')}\n\n"
        if article_quote
        else ""
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
        f"{article_quote_instruction}"
        "Follow the content skill graph (voice, banned words, humanizer rules, "
        "and NEVER use em dashes or en dashes). This is a website article, NOT a social caption: "
        "write 2 to 3 short paragraphs of plain, human prose. Keep it tight: lead with the "
        "specific fact (a venue, an award, a number, a year), vary sentence length, and cut "
        "every empty adjective. No press-release cadence, no wall of prose. Do NOT include hashtags or "
        "emoji. Do NOT include any visible raw URL, the ONLY links permitted are "
        "the two inline HTML anchors described below; a 'Read more' button handles "
        "the source link separately.\n"
        f"{tie_in_instruction}"
        "If the story announces an upcoming show, performance, or event, you MUST "
        "state the show date explicitly in the article (and the city/venue when "
        "known). Do NOT, however, point readers to a specific show, date, or venue "
        "that has ALREADY occurred, for past events, write about the artist and "
        "their broader story instead.\n"
        "Aside from the required booking CTA above, do NOT add any other "
        "call-to-action (no 'visit loveproductions.com', 'click the button/see "
        "below', or 'full event details here' for tickets or the venue). The "
        "booking invitation is the only permitted closing directive.\n\n"
        "Also choose 1 to 3 categories that best fit this story. You MUST pick only "
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
        config.record_api_exception(exc, "Article generation")
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

"""Music-news RSS ingestion.

A second, publisher-sanctioned news channel that runs *alongside* Anthropic's
hosted ``web_search`` (used by ``lp.ai.search_artist_news``). Each whitelisted
outlet publishes an RSS feed for syndication; we read the headline, summary, and
the outlet's own link back to the article — we never republish article bodies.

Only outlets whose RSS is clean and unambiguously offered for reading are
included. Pitchfork's feed works fine from our own server (Condé Nast only
blocks Anthropic's WebFetch IPs, not ours). Deliberately excluded: Consequence
(robots.txt disallows /feed/ and blocks ClaudeBot).

Pure stdlib + ``requests`` — no new dependency (mirrors ``lp.scrape``). Any
failure (network, bad XML, one dead feed) degrades to fewer/no items, never
raises.

Emits topic dicts in the exact shape ``lp.ai.search_artist_news`` returns
(headline, url, summary, hook_type, is_live_event, artist, original_artist) so
matches flow straight into the existing score → dedup → generate → Buffer
pipeline with real source links.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import requests

log = logging.getLogger(__name__)

# Whitelisted outlets only (name, RSS url). Keep this list curated — adding a
# feed here immediately puts it in the weekly run.
MUSIC_FEEDS: list[tuple[str, str]] = [
    ("Billboard", "https://www.billboard.com/feed/"),
    ("Pitchfork", "https://pitchfork.com/feed/feed-news/rss"),
    ("Rolling Stone", "https://www.rollingstone.com/music/feed/"),
    ("Stereogum", "https://www.stereogum.com/feed/"),
    ("Brooklyn Vegan", "https://www.brooklynvegan.com/feed/"),
    ("American Songwriter", "https://americansongwriter.com/feed/"),
    ("NME", "https://www.nme.com/feed"),
]

# Only surface items published within this window, matching the web-search
# recency window in search_artist_news().
FEED_DAYS = 14

_TIMEOUT = 8
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Artist-name tokens shorter than this are too ambiguous to match on (e.g.
# "Kiss", "Yes", "War") without dragging in unrelated stories.
_MIN_NAME_LEN = 4

# Live-event language. Original-artist items that look like a tour/show
# announcement are dropped, mirroring the HARD RULE code guard in
# search_artist_news(): we never promote an original artist's own live dates.
_LIVE_EVENT_RE = re.compile(
    r"\b(tour|tours|touring|concert|concerts|residency|residencies|"
    r"festival|gig|gigs|tickets?|live at|live in|on stage|world tour|"
    r"north american tour|announces? (?:a )?(?:tour|dates|show|residency)|"
    r"tour dates?|show dates?|kicks? off (?:its|their|his|her) tour)\b",
    re.IGNORECASE,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Within-run cache so all feeds are fetched at most once per process.
_items_cache: list[dict] | None = None


def _clean(text: str) -> str:
    """Strip HTML tags and collapse whitespace from a feed field."""
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub(" ", text or "")).strip()


def _parse_date(raw: str) -> datetime | None:
    """Parse an RSS pubDate (RFC-822) into an aware UTC datetime, or None."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fetch_feed(name: str, url: str) -> list[dict]:
    """Fetch and parse one RSS 2.0 feed. Returns [] on any failure."""
    try:
        resp = requests.get(
            url, timeout=_TIMEOUT, headers={"User-Agent": _UA}, allow_redirects=True
        )
        if resp.status_code != 200 or not resp.content:
            log.warning("Feed %s returned HTTP %s", name, resp.status_code)
            return []
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        log.warning("Feed %s unavailable: %s", name, exc)
        return []

    items: list[dict] = []
    # RSS 2.0: channel/item. Tolerate an Atom feed's entry/ as a fallback.
    entries = root.findall(".//item")
    is_atom = not entries
    if is_atom:
        entries = [e for e in root.iter() if e.tag.endswith("}entry")]

    for entry in entries:
        title = link = summary = pub = ""
        for child in entry:
            tag = child.tag.split("}")[-1].lower()
            if tag == "title":
                title = child.text or ""
            elif tag == "link":
                # RSS puts the URL in text; Atom in the href attribute.
                link = (child.text or child.get("href") or "").strip()
            elif tag in ("description", "summary") and not summary:
                summary = child.text or ""
            elif tag in ("pubdate", "published", "updated") and not pub:
                pub = child.text or ""
        title = _clean(title)
        link = link.strip()
        if not title or not link:
            continue
        items.append(
            {
                "title": title,
                "url": link,
                "summary": _clean(summary),
                "published": _parse_date(pub),
                "source": name,
            }
        )
    return items


def load_feed_items(force: bool = False) -> list[dict]:
    """Fetch every whitelisted feed once, newest-first within FEED_DAYS.

    Cached for the life of the process. Each item is a dict with keys
    ``title``, ``url``, ``summary``, ``published`` (aware datetime or None),
    and ``source``. Undated items are kept (some feeds omit pubDate).
    """
    global _items_cache
    if _items_cache is not None and not force:
        return _items_cache

    cutoff = datetime.now(timezone.utc) - timedelta(days=FEED_DAYS)
    collected: list[dict] = []
    for name, url in MUSIC_FEEDS:
        for item in _fetch_feed(name, url):
            if item["published"] is not None and item["published"] < cutoff:
                continue
            collected.append(item)

    collected.sort(key=lambda i: i["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    log.info("Loaded %d recent item(s) across %d music feed(s)", len(collected), len(MUSIC_FEEDS))
    _items_cache = collected
    return collected


def _name_variants(original: str) -> list[str]:
    """Split a mapping's original-artist cell into individual searchable names."""
    parts = re.split(r"\s*(?:,|&|/|\band\b)\s*", original or "")
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_NAME_LEN]


def _mentions(name: str, haystack: str) -> bool:
    """Case-insensitive whole-phrase match of an artist name in text."""
    return re.search(r"\b" + re.escape(name) + r"\b", haystack, re.IGNORECASE) is not None


def search_artist_feeds(tribute: str, original: str, items: list[dict] | None = None) -> list[dict]:
    """Return feed items mentioning the tribute act or its original artist(s).

    Output dicts match search_artist_news()'s shape. Items matched on the
    tribute name are ``tribute_news``; items matched only on an original artist
    are ``original_artist_news`` — and any of those that read as a live-event
    announcement are dropped (never promote the original artist's own shows).
    """
    if items is None:
        items = load_feed_items()

    tribute = (tribute or "").strip()
    originals = _name_variants(original)
    if len(tribute) < _MIN_NAME_LEN and not originals:
        return []

    results: list[dict] = []
    for item in items:
        haystack = f"{item['title']} {item['summary']}"
        hit_tribute = len(tribute) >= _MIN_NAME_LEN and _mentions(tribute, haystack)
        hit_original = any(_mentions(n, haystack) for n in originals)
        if not (hit_tribute or hit_original):
            continue

        hook_type = "tribute_news" if hit_tribute else "original_artist_news"
        is_live_event = bool(_LIVE_EVENT_RE.search(haystack))

        # Mirror the search_artist_news() hard rule: never surface an original
        # artist's own live-event as news.
        if hook_type == "original_artist_news" and is_live_event:
            log.info(
                "Feed: dropping original-artist live-event (never news): %s",
                item["title"][:60],
            )
            continue

        matched_artist = tribute if hit_tribute else next(
            (n for n in originals if _mentions(n, haystack)), original
        )
        summary = item["summary"][:300] or item["title"]
        results.append(
            {
                "headline": item["title"],
                "url": item["url"],
                "summary": f"{summary} (via {item['source']})",
                "hook_type": hook_type,
                "is_live_event": is_live_event,
                "artist": matched_artist,
                "original_artist": original,
            }
        )

    if results:
        log.info("Feeds: %d match(es) for %s", len(results), tribute)
    return results

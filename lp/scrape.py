"""
Page scraping helpers, cheap, no AI call.

``fetch_og_image()`` fetches a source URL once and extracts its preview image
(og:image, with twitter:image / og:image:url fallbacks). Used to populate native
image assets on LinkedIn/Instagram drafts so LinkedIn can be an image-only post
(no link card) and Instagram can show the real article image instead of the LP
logo.

``fetch_page_text()`` returns a page's visible text, and
``verify_quote_on_page()`` uses it to confirm that a quote a model claims to
have found is actually present at the cited URL (the anti-fabrication guard for
testimonial posts).

Pure HTTP + regex, no new dependency beyond ``requests``. Any failure
(network error, non-200, no tag, malformed HTML) returns ``None``/``False`` so
callers can fall back gracefully.
"""

import html
import logging
import re
from urllib.parse import urljoin

import requests

log = logging.getLogger(__name__)

_TIMEOUT = 5
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Match a <meta ...> tag whose property/name is one of the OG image keys and
# capture the content attribute, tolerant of attribute order and quote style.
_META_RE = re.compile(
    r"<meta[^>]+(?:property|name)\s*=\s*['\"]"
    r"(og:image(?::url)?|twitter:image(?::src)?)['\"][^>]*>",
    re.IGNORECASE,
)
_CONTENT_RE = re.compile(r"content\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)

# Within-run cache so the same source URL is fetched at most once.
_cache: dict[str, str | None] = {}
_text_cache: dict[str, str] = {}

# Blocks whose contents are markup, not prose. Dropped before tag stripping so
# inline scripts and styles can't accidentally satisfy a quote match.
_NOISE_RE = re.compile(
    r"<(script|style|noscript|template)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")

# Everything after </main> is chrome: the site footer carries a newsletter
# signup whose "By signing in, you agree to our terms and conditions" copy was
# landing in every act's scraped prose.
_MAIN_END_RE = re.compile(r"</main\b[^>]*>.*", re.IGNORECASE | re.DOTALL)

# loveproductions.com act pages embed a "Very Simple Event List" widget (its
# markup is prefixed ``vsel-``) listing OTHER acts' upcoming events, each with a
# teaser paragraph in <div class="vsel-text"><p>...</p></div>. Those paragraphs
# sit inside <main>, so paragraph extraction swept them up and roughly 40% of
# every act's "own page copy" was actually six other acts' blurbs. That summary
# feeds act spotlight posts, which is why they read as though they could be
# about any act. The div holds only paragraphs (no nested divs), so a non-greedy
# match to the closing tag is safe here.
_EVENT_WIDGET_RE = re.compile(
    r"<div[^>]*\bclass\s*=\s*['\"][^'\"]*\bvsel-text\b[^'\"]*['\"][^>]*>.*?</div>",
    re.IGNORECASE | re.DOTALL,
)

# Typographic characters that differ between a model's transcription of a quote
# and the page's own markup. Normalized away on both sides before comparing.
_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "…": "...",
})

# How much of the quote must match. Long enough that a match can't be
# coincidental, short enough to survive an ellipsis or a trailing attribution.
_QUOTE_WINDOW = 40


def fetch_og_image(url: str) -> str | None:
    """Return the page's preview image URL, or None if unavailable.

    Tries og:image, then og:image:url, then twitter:image. Relative image URLs
    are resolved against the (possibly redirected) page URL. Never raises.
    """
    url = (url or "").strip()
    if not url:
        return None
    if url in _cache:
        return _cache[url]

    result: str | None = None
    try:
        resp = requests.get(
            url,
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            allow_redirects=True,
        )
        if resp.status_code == 200 and resp.text:
            # Prefer og:image over twitter:image when both are present.
            best = None
            for m in _META_RE.finditer(resp.text):
                key = m.group(1).lower()
                content = _CONTENT_RE.search(m.group(0))
                if not content:
                    continue
                img = content.group(1).strip()
                if not img:
                    continue
                img = urljoin(resp.url, img)
                if key.startswith("og:image"):
                    best = img
                    break  # og:image is the strongest signal
                if best is None:
                    best = img  # hold twitter:image as fallback
            result = best
    except Exception as exc:  # noqa: BLE001, any failure is a graceful miss
        log.info("og:image scrape failed for %s: %s", url, exc)
        result = None

    _cache[url] = result
    if result:
        log.info("og:image found for %s -> %s", url, result)
    return result


def _normalize(text: str) -> str:
    """Lowercase, unify quote/dash characters, collapse whitespace."""
    return " ".join(text.translate(_PUNCT_MAP).lower().split())


def fetch_page_text(url: str) -> str:
    """Return a page's visible text, or "" on any failure.

    Scripts, styles and tags are stripped and entities unescaped, leaving
    whitespace-collapsed prose. Cached per URL for the run. Never raises.
    """
    url = (url or "").strip()
    if not url:
        return ""
    if url in _text_cache:
        return _text_cache[url]

    text = ""
    try:
        resp = requests.get(
            url,
            timeout=_TIMEOUT,
            headers={"User-Agent": _UA},
            allow_redirects=True,
        )
        if resp.status_code == 200 and resp.text:
            stripped = _TAG_RE.sub(" ", _NOISE_RE.sub(" ", resp.text))
            text = " ".join(html.unescape(stripped).split())
    except Exception as exc:  # noqa: BLE001, any failure is a graceful miss
        log.info("page text fetch failed for %s: %s", url, exc)
        text = ""

    _text_cache[url] = text
    return text


def fetch_page_prose(url: str, max_chars: int = 2500) -> str:
    """Return a page's body prose, taken from its <p> elements.

    Used to read an act's own loveproductions.com page for spotlight posts.
    Paragraph extraction (rather than whole-page text) keeps out nav, menus and
    footers, which otherwise dominate a WordPress page. Paragraphs shorter than
    a sentence are dropped as chrome. Returns "" on any failure.

    Paragraph extraction alone is not enough on this site: the page tail and the
    related-events widget both contribute prose that is not about this act. Both
    are removed first (see ``_MAIN_END_RE`` and ``_EVENT_WIDGET_RE``) so the
    result is only the act's own copy.
    """
    url = (url or "").strip()
    if not url:
        return ""
    try:
        resp = requests.get(
            url, timeout=_TIMEOUT, headers={"User-Agent": _UA}, allow_redirects=True
        )
        if resp.status_code != 200 or not resp.text:
            return ""
        body = _NOISE_RE.sub(" ", resp.text)
        body = _MAIN_END_RE.sub(" ", body)
        body = _EVENT_WIDGET_RE.sub(" ", body)
    except Exception as exc:  # noqa: BLE001, any failure is a graceful miss
        log.info("page prose fetch failed for %s: %s", url, exc)
        return ""

    paragraphs = []
    for raw in re.findall(r"<p\b[^>]*>(.*?)</p>", body, re.IGNORECASE | re.DOTALL):
        para = " ".join(html.unescape(_TAG_RE.sub(" ", raw)).split())
        if len(para) >= 60:  # anything shorter is a caption, label or nav item
            paragraphs.append(para)

    prose = "\n\n".join(paragraphs)[:max_chars]
    if not prose:
        log.info("No usable prose found at %s", url)
    return prose


# A pull-quote paragraph on an act page: the quote in (possibly mismatched)
# quotation marks, then a dash, then the attribution. The site mixes straight
# and curly marks within a single quote, so the delimiters are matched as a
# character class rather than as a balanced pair.
_PULLQUOTE_RE = re.compile(
    r"""^["“”']\s*(?P<quote>.+?)\s*["“”']\s*[—–-]\s*(?P<attribution>.+)$""",
    re.DOTALL,
)

# Attribution words that mark a quote as coming from a buyer (a venue, promoter
# or festival) rather than from the press. Buyer praise is the stronger proof on
# LinkedIn, so the two are labelled and can be prioritised separately.
_BUYER_WORDS = (
    "producer", "promoter", "director", "manager", "theatre", "theater",
    "center", "centre", "festival", "playhouse", "arena", "venue", "hall",
    "casino", "resort", "cruise", "chairman", "coordinator", "president",
    "days", "council", "fair",
)


def extract_page_quotes(url: str) -> list[dict]:
    """Return attributed pull-quotes published on an act's own LP page.

    Several loveproductions.com act pages carry praise the agency has already
    published and therefore already cleared: buyer quotes from venues and
    festivals, plus press pull-quotes. These are strictly better testimonial
    material than an open-web search, which mostly returns nothing and needs
    :func:`verify_quote_on_page` to guard against reconstructed quotes. Here the
    page *is* the source, so the quote is verified by construction.

    Each returned dict has ``quote``, ``attribution``, ``url`` and
    ``source_type`` (``"buyer"`` or ``"press"``). Returns [] on any failure.
    """
    quotes = []
    for para in fetch_page_prose(url, max_chars=6000).split("\n\n"):
        match = _PULLQUOTE_RE.match(para.strip())
        if not match:
            continue
        # Pages elide the start of long quotes with a leading ellipsis.
        quote = match.group("quote").strip().lstrip(".…").strip()
        attribution = " ".join(match.group("attribution").split())
        if len(quote) < 25 or not 3 <= len(attribution) <= 120:
            continue
        lowered = attribution.lower()
        quotes.append({
            "quote": quote,
            "attribution": attribution,
            "url": url,
            "source_type": "buyer" if any(w in lowered for w in _BUYER_WORDS) else "press",
        })
    return quotes


# Act pages carry no og:image, so a card background has to come from the page
# body. Images are lazy-loaded (the real URL sits in data-src, with an inline SVG
# placeholder in src), and the related-events widget contributes other acts'
# photos, so those blocks are stripped before the search.
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_IMG_SRC_RE = re.compile(
    r"(?:data-src|src)\s*=\s*['\"]([^'\"]+/wp-content/uploads/[^'\"]+\.(?:jpe?g|png))['\"]",
    re.IGNORECASE,
)
_IMG_WIDTH_RE = re.compile(r"width\s*=\s*['\"](\d+)['\"]", re.IGNORECASE)


# Filename markers for artwork that is already a designed graphic: tour posters,
# ad slicks, video thumbnails. They carry their own headline type, which fights
# the card's type no matter how far the background is darkened.
_ARTWORK_RE = re.compile(
    r"(?:^|[_\-/])(?:ad|ads|advert|poster|flyer|cover|banner|logo|thumb|thumbnail|"
    r"promo|video[_\-]?cover|artwork|placeholder)(?:[_\-.\d]|$)",
    re.I,
)


def fetch_act_photo(url: str, act: str = "") -> str | None:
    """Return the best photo on an act's own page to sit behind card type, or None.

    Used as the background for branded cards (:mod:`lp.cards`). Size alone is a
    bad heuristic: the largest asset on these pages is often a promo slick or a
    video cover whose own headline type collides with the card's. So candidates
    are scored, preferring a file whose name matches the act and penalising
    anything that looks like designed artwork, with width as the tiebreaker.
    Never raises.
    """
    url = (url or "").strip()
    if not url:
        return None
    try:
        resp = requests.get(
            url, timeout=_TIMEOUT, headers={"User-Agent": _UA}, allow_redirects=True
        )
        if resp.status_code != 200 or not resp.text:
            return None
    except Exception as exc:  # noqa: BLE001, a missing photo is a graceful miss
        log.info("Act photo fetch failed for %s: %s", url, exc)
        return None

    body = _EVENT_WIDGET_RE.sub(" ", _MAIN_END_RE.sub(" ", _NOISE_RE.sub(" ", resp.text)))

    # Word-level tokens from the act name, used to spot the file that is
    # actually this act's own photo ("Arrival_From_Sweden_1000x1000.jpg").
    tokens = [t for t in re.split(r"[^a-z0-9]+", act.lower()) if len(t) > 3]

    best, best_score, best_width = None, None, 0
    for tag in _IMG_TAG_RE.findall(body):
        if "vsel-" in tag:  # a related act's thumbnail, not this act's photo
            continue
        src = _IMG_SRC_RE.search(tag)
        if not src:
            continue

        absolute = urljoin(resp.url, src.group(1))
        name = absolute.rsplit("/", 1)[-1].lower()
        width = int(m.group(1)) if (m := _IMG_WIDTH_RE.search(tag)) else 0

        score = 0
        if tokens and sum(t in name for t in tokens) >= max(1, len(tokens) // 2):
            score += 2
        if _ARTWORK_RE.search(name):
            score -= 2
        if (score, width) > ((best_score, best_width) if best else (-99, -1)):
            best, best_score, best_width = absolute, score, width

    if best:
        log.info("Act photo for %s -> %s (score %d, %dpx)", url, best, best_score, best_width)
    return best


def verify_quote_on_page(url: str, quote: str) -> bool:
    """True only if ``quote`` demonstrably appears at ``url``.

    The anti-fabrication guard for testimonial posts: a model asked to find a
    real quote will sometimes reconstruct a plausible one from memory and cite a
    URL that never contained it. Prompting alone does not stop that, so every
    testimonial is checked against the actual page before it can be posted.

    Comparison is on normalized text (case, smart quotes, dashes and whitespace
    folded), matching the first ``_QUOTE_WINDOW`` characters of the quote so a
    trailing ellipsis or attribution does not cause a false negative. Fails
    closed: an unreachable page, a paywall, or a short quote returns False.
    """
    quote = (quote or "").strip().strip('"“”')
    if not url or len(quote) < 25:
        return False

    needle = _normalize(quote)[:_QUOTE_WINDOW]
    if len(needle) < 25:
        return False

    haystack = _normalize(fetch_page_text(url))
    if not haystack:
        log.info("Quote verification: no text retrieved from %s", url)
        return False

    found = needle in haystack
    if not found:
        log.info("Quote verification FAILED, not found at %s: %r", url, needle)
    return found

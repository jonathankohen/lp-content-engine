"""
Open Graph image scraping — cheap, no AI call.

Fetches a source URL once and extracts its preview image (og:image, with
twitter:image / og:image:url fallbacks). Used to populate native image assets
on LinkedIn/Instagram drafts so LinkedIn can be an image-only post (no link
card) and Instagram can show the real article image instead of the LP logo.

Pure HTTP + regex — no new dependency beyond ``requests``. Any failure
(network error, non-200, no tag, malformed HTML) returns ``None`` so callers
can fall back gracefully.
"""

import logging
import re
from urllib.parse import urljoin, urlparse

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
    except Exception as exc:  # noqa: BLE001 — any failure is a graceful miss
        log.info("og:image scrape failed for %s: %s", url, exc)
        result = None

    _cache[url] = result
    if result:
        log.info("og:image found for %s -> %s", url, result)
    return result


# A loveproductions.com artist page has no og:image tag; its act photo is the
# first real /wp-content/uploads image in the markup. We skip favicons, logos,
# tiny thumbnails (e.g. -32x32), and full-bleed *background* images (named
# bkgd/bkg/backgd) — the act's square promo image comes right after — and we
# require the page's own host so we never grab a stale staging-CDN copy.
_UPLOAD_IMG_RE = re.compile(
    r"https?://[^\"'\s)]+/wp-content/uploads/[^\"'\s)]+\.(?:jpg|jpeg|png|webp)",
    re.IGNORECASE,
)
_IMG_EXCLUDE_RE = re.compile(
    r"(icon|logo|favicon|cropped-lpi|sprite|placeholder|"
    r"bkgd|bkg|backgd|background|-\d{2,3}x\d{2,3}\.)",
    re.IGNORECASE,
)
_artist_img_cache: dict[str, str | None] = {}


def fetch_artist_image(url: str) -> str | None:
    """Return an artist page's act photo (first non-background upload), or None.

    Used as the featured-image fallback for a news post when the source article
    has no og:image — we'd rather show the act's own photo from their
    loveproductions.com page than the generic LP logo placeholder. Cheap (one
    cached GET + regex, no AI). Never raises.
    """
    url = (url or "").strip()
    if not url:
        return None
    if url in _artist_img_cache:
        return _artist_img_cache[url]

    host = urlparse(url).netloc.lower()
    result: str | None = None
    try:
        resp = requests.get(
            url, timeout=_TIMEOUT, headers={"User-Agent": _UA}, allow_redirects=True
        )
        if resp.status_code == 200 and resp.text:
            for m in _UPLOAD_IMG_RE.finditer(resp.text):
                cand = m.group(0)
                # Same host as the page (avoids stale staging-CDN copies) and not
                # an icon/logo/background/thumbnail.
                if host and urlparse(cand).netloc.lower() != host:
                    continue
                if _IMG_EXCLUDE_RE.search(cand):
                    continue
                result = cand
                break
    except Exception as exc:  # noqa: BLE001 — any failure is a graceful miss
        log.info("artist image scrape failed for %s: %s", url, exc)
        result = None

    _artist_img_cache[url] = result
    if result:
        log.info("artist image for %s -> %s", url, result)
    return result

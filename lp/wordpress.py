"""
LP News WordPress client.

Posts a batch of news items to the LP News plugin on loveproductions.com, which
creates one DRAFT WordPress post per item (featured image, body, category, and a
red "Read more" button). See the plugin at wordpress-plugin/lp-news/lp-news.php.

Each item is a dict:
    {
        "key":          stable dedup key (source URL, else a sheet/topic key),
        "title":        headline,
        "body":         article body (plain text; paragraphs split on blank lines),
        "categories":   list[str] from the allowed category set,
        "button_url":   where the "Read more" button points,
        "button_label": button text (default "Read more"),
        "image_url":    article's og:image, or "" when the article has none
                        (the post is then drafted without a featured image),
    }

The plugin dedups on `key`, so re-sending an item already on the site is a no-op.
"""

import base64
import hashlib
import logging
import os
import time

import requests

from . import config

log = logging.getLogger(__name__)

_TIMEOUT = 60  # image sideloading on the server side can be slow

# Cards render identically on every run, so an upload that already happened does
# not need to happen again. The plugin dedups server-side too; this just saves
# the round trip.
_media_cache: dict[str, str] = {}


# The host 413s any request body over roughly 1MB, before PHP runs. Base64
# inflates by a third, so a single-shot upload has to stay under ~750KB of file.
# Both numbers are set well clear of the cliff rather than at it, since the exact
# limit is the host's to change and we would rather chunk unnecessarily than fail.
_SINGLE_SHOT_MAX = 500 * 1024      # send files above this in pieces
_CHUNK_BYTES = 400 * 1024          # ~547KB once base64-encoded


def _upload_chunked(base: str, key: str, filename: str, blob: bytes,
                    headers: dict) -> str | None:
    """Send a file to /upload-media-chunk in order, return the attachment URL.

    The server appends each piece to a temp file and assembles on the last one,
    so the pieces must arrive in order and any failure aborts the whole upload:
    a partial file that later got treated as complete would be a corrupt video.
    The server answers a dedup hit on the first chunk, so an already-hosted file
    costs one request rather than all of them.
    """
    endpoint = base + "/upload-media-chunk"
    chunks = [blob[i:i + _CHUNK_BYTES] for i in range(0, len(blob), _CHUNK_BYTES)]
    total = len(chunks)
    log.info("Uploading %s in %d chunk(s), %.1f MB", filename, total, len(blob) / 1e6)

    for i, piece in enumerate(chunks):
        payload = {
            "key":            key,
            "filename":       filename,
            "index":          i,
            "total":          total,
            "content_base64": base64.b64encode(piece).decode("ascii"),
        }
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=_TIMEOUT)
        except Exception as exc:  # noqa: BLE001, a failed upload must not end the run
            log.error("Chunk %d/%d request error: %s", i + 1, total, exc)
            return None
        if not resp.ok:
            log.error("Chunk %d/%d failed %d: %s", i + 1, total, resp.status_code, resp.text[:200])
            return None
        try:
            data = resp.json() or {}
        except Exception:  # noqa: BLE001
            log.error("Chunk %d/%d returned non-JSON: %s", i + 1, total, resp.text[:200])
            return None
        # A dedup hit short-circuits the rest of the upload.
        if data.get("url"):
            if data.get("skipped") and i == 0:
                log.info("Already hosted, skipping %d remaining chunk(s)", total - 1)
            return data["url"]

    log.error("Final chunk returned no URL for %s", filename)
    return None


def upload_media(path: str, key: str = "") -> str | None:
    """Upload a local image to the site's media library, return its public URL.

    Buffer will not take image bytes, only a URL it can fetch, so locally
    rendered cards (see ``lp.cards``) have to be hosted before they can be
    attached to a draft. The LP News plugin's ``/upload-media`` route takes the
    bytes and hands back the attachment URL.

    ``key`` dedups both here and on the server; it defaults to a hash of the file
    contents, which for a deterministic card means re-runs reuse one attachment
    instead of littering the media library.
    """
    if not path or not os.path.exists(path):
        log.warning("upload_media: no file at %s", path)
        return None

    if not config.LP_NEWS_URL or not config.LP_NEWS_SECRET:
        log.info("LP_NEWS_URL/LP_NEWS_SECRET not set, cannot host card %s", os.path.basename(path))
        return None

    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError as exc:
        log.warning("upload_media: could not read %s: %s", path, exc)
        return None

    key = key or f"sha1_{hashlib.sha1(blob).hexdigest()[:16]}"
    if key in _media_cache:
        return _media_cache[key]

    base = config.LP_NEWS_URL.rsplit("/", 1)[0]
    headers = {
        "Content-Type":     "application/json",
        "X-LP-News-Secret": config.LP_NEWS_SECRET,
    }

    # Anything that will not fit in one request body goes up in pieces. The host
    # rejects a body over roughly 1MB with a 413 before PHP sees it (measured
    # 2026-08-03), which is what silently killed video: a clip is 3.7 to 6.1MB.
    if len(blob) > _SINGLE_SHOT_MAX:
        url = _upload_chunked(base, key, os.path.basename(path), blob, headers)
        if url:
            _media_cache[key] = url
            log.info("Hosted (chunked): %s", url)
        return url

    endpoint = base + "/upload-media"
    payload = {
        "key":            key,
        "filename":       os.path.basename(path),
        "content_base64": base64.b64encode(blob).decode("ascii"),
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=_TIMEOUT)
    except Exception as exc:  # noqa: BLE001, a failed upload must not end the run
        log.error("Media upload request error: %s", exc)
        return None

    if not resp.ok:
        log.error("Media upload %d: %s", resp.status_code, resp.text[:300])
        return None

    try:
        url = (resp.json() or {}).get("url") or ""
    except Exception:  # noqa: BLE001
        log.error("Media upload returned non-JSON: %s", resp.text[:200])
        return None

    if not url:
        log.error("Media upload succeeded but returned no URL")
        return None

    _media_cache[key] = url
    log.info("Hosted card: %s", url)
    return url


def publish_news_posts(posts: list[dict], dry_run: bool = False) -> dict:
    """POST the batch to the LP News plugin. Returns the plugin's JSON response
    ({created, skipped, would_create, errors}) or {} on failure / when disabled.

    In dry-run mode this still calls the endpoint with ``dry_run: true`` when
    LP_NEWS_URL is configured (the plugin then plans without writing); if the
    endpoint is not configured, it logs the planned posts locally and returns {}.
    """
    if not posts:
        return {}

    if not config.LP_NEWS_URL or not config.LP_NEWS_SECRET:
        log.info(
            "LP_NEWS_URL/LP_NEWS_SECRET not set, skipping website posting (%d item(s)).",
            len(posts),
        )
        for p in posts:
            log.info("  [would post] %s | cats=%s | btn=%s",
                     p.get("title", "")[:70],
                     ", ".join(p.get("categories", [])),
                     p.get("button_url", ""))
        return {}

    payload = {"dry_run": bool(dry_run), "posts": posts}
    headers = {
        "Content-Type": "application/json",
        "X-LP-News-Secret": config.LP_NEWS_SECRET,
    }

    for _ in range(2):
        try:
            resp = requests.post(config.LP_NEWS_URL, json=payload, headers=headers, timeout=_TIMEOUT)
        except Exception as exc:
            log.error("LP News request error: %s", exc)
            return {}

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 61
            log.warning("LP News rate limited, retrying in %ds...", wait)
            time.sleep(wait)
            continue

        if not resp.ok:
            log.error("LP News %d: %s", resp.status_code, resp.text[:500])
            return {}

        try:
            data = resp.json()
        except Exception:
            log.error("LP News returned non-JSON: %s", resp.text[:300])
            return {}

        created = data.get("would_create" if dry_run else "created", []) or []
        skipped = data.get("skipped", []) or []
        errors = data.get("errors", []) or []
        log.info(
            "LP News: %d %s, %d skipped, %d error(s)",
            len(created),
            "planned" if dry_run else "created",
            len(skipped),
            len(errors),
        )
        for c in created:
            log.info("  ✓ %s [%s]", c.get("title", "")[:70], ", ".join(c.get("categories", [])))
        for e in errors:
            log.warning("  ✗ %s, %s", e.get("title", "")[:70], e.get("error", ""))
        return data

    log.error("LP News rate limit persists after retry")
    return {}

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

import logging
import time

import requests

from . import config

log = logging.getLogger(__name__)

_TIMEOUT = 60  # image sideloading on the server side can be slow


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
            "LP_NEWS_URL/LP_NEWS_SECRET not set — skipping website posting (%d item(s)).",
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
            log.warning("LP News rate limited — retrying in %ds...", wait)
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
            log.warning("  ✗ %s — %s", e.get("title", "")[:70], e.get("error", ""))
        return data

    log.error("LP News rate limit persists after retry")
    return {}

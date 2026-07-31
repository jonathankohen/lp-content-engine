import logging

import requests

from . import config

log = logging.getLogger(__name__)

_GRAPH_BASE = "https://graph.facebook.com/v21.0"


def _graph_get(path: str, params: dict, silent: bool = False) -> dict:
    params = {**params, "access_token": config.META_PAGE_ACCESS_TOKEN}
    resp = requests.get(f"{_GRAPH_BASE}/{path}", params=params, timeout=15)
    if not resp.ok:
        if not silent:
            log.error("Meta Graph API %s %s: %s", resp.status_code, path, resp.text[:200])
        return {}
    return resp.json()


def _get_page_id() -> str:
    data = _graph_get("me", {"fields": "id"})
    return data.get("id", "")


def _fetch_facebook_posts(limit: int = 25) -> list[dict]:
    page_id = _get_page_id()
    if not page_id:
        return []
    data = _graph_get(f"{page_id}/posts", {
        "fields": "message,reactions.summary(true).limit(0),comments.summary(true).limit(0),shares",
        "limit": limit,
    })
    posts = []
    for item in data.get("data", []):
        reactions = (item.get("reactions") or {}).get("summary", {}).get("total_count", 0)
        comments = (item.get("comments") or {}).get("summary", {}).get("total_count", 0)
        shares = (item.get("shares") or {}).get("count", 0)
        score = reactions + comments + shares
        if score > 0:
            posts.append({
                "text": item.get("message", ""),
                "platform": "Facebook",
                "engagement_score": score,
            })
    return posts


def _get_ig_account_id() -> str:
    data = _graph_get("me", {"fields": "instagram_business_account"}, silent=True)
    return (data.get("instagram_business_account") or {}).get("id", "")


def _fetch_instagram_posts(limit: int = 25) -> list[dict]:
    ig_id = _get_ig_account_id()
    if not ig_id:
        log.debug("No Instagram business account linked to this Facebook Page, skipping")
        return []
    data = _graph_get(f"{ig_id}/media", {
        "fields": "caption,like_count,comments_count",
        "limit": limit,
    })
    posts = []
    for item in data.get("data", []):
        score = (item.get("like_count") or 0) + (item.get("comments_count") or 0)
        if score > 0:
            posts.append({
                "text": item.get("caption", ""),
                "platform": "Instagram",
                "engagement_score": score,
            })
    return posts


def fetch_meta_top_performers(n: int = 3) -> list[dict]:
    """Return top n posts by engagement across Facebook and Instagram."""
    if not config.META_PAGE_ACCESS_TOKEN:
        return []
    try:
        posts = _fetch_facebook_posts() + _fetch_instagram_posts()
        posts.sort(key=lambda p: p["engagement_score"], reverse=True)
        return posts[:n]
    except Exception as e:
        log.warning("Meta analytics fetch failed: %s", e)
        return []

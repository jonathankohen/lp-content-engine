import csv
import io
import logging
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


def fetch_linkedin_top_performers(n: int = 3) -> list[dict]:
    """Return top n LinkedIn posts by engagement from a manually exported CSV."""
    if not config.LINKEDIN_ANALYTICS_CSV:
        return []
    path = Path(config.LINKEDIN_ANALYTICS_CSV)
    if not path.exists():
        log.debug("LinkedIn analytics CSV not found: %s", path)
        return []
    try:
        return _parse_csv(path, n)
    except Exception as e:
        log.warning("LinkedIn analytics CSV parse failed: %s", e)
        return []


def _parse_csv(path: Path, n: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        raw = f.read()
    # LinkedIn exports prepend a description row before the real headers, skip to it.
    idx = raw.find("Post title")
    if idx == -1:
        log.warning("LinkedIn CSV: could not find 'Post title' header, wrong sheet or format")
        return []
    posts = []
    for row in csv.DictReader(io.StringIO(raw[idx:])):
        text = _col(row, "Post title", "Title", "Update") or ""
        likes = _int(row, "Likes")
        comments = _int(row, "Comments")
        shares = _int(row, "Shares", "Reposts")
        score = likes + comments + shares
        if score > 0 and text:
            posts.append({
                "text": text,
                "platform": "LinkedIn",
                "engagement_score": score,
            })
    log.info("LinkedIn CSV: %d post(s) with engagement > 0 found", len(posts))
    posts.sort(key=lambda p: p["engagement_score"], reverse=True)
    return posts[:n]


def _col(row: dict, *candidates: str) -> str:
    for name in candidates:
        for key in row:
            if key.strip().lower() == name.lower():
                return row[key].strip()
    return ""


def _int(row: dict, *candidates: str) -> int:
    val = _col(row, *candidates)
    try:
        return int(float(val.replace(",", ""))) if val else 0
    except (ValueError, AttributeError):
        return 0

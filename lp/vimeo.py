"""
Short video clips cut from Love Productions' own Vimeo library.

The client asked for "maybe some quick video" (direction 2026-07-31). The agency
has ~1,600 videos on Vimeo, mostly act promo reels, and 23 of the 24 roster acts
are represented. That footage is the agency's own, so unlike anything generated
or scraped the rights are already clean, which is why this is the video source.

The flow is: find an act's best promo video, download a progressive rendition,
cut a short square clip with ffmpeg, then hand the mp4 to
``lp.wordpress.upload_media`` for a public URL that Buffer can attach as a video
asset.

Requires ``ffmpeg``/``ffprobe`` on PATH and a ``VIMEO_ACCESS_TOKEN`` with the
``video_files`` scope (without it the API omits download links entirely).
"""

import logging
import os
import re
import shutil
import subprocess

import requests

from . import config

log = logging.getLogger(__name__)

API = "https://api.vimeo.com"
_TIMEOUT = 30
_HEADERS_ACCEPT = "application/vnd.vimeo.*+json;version=3.4"

# Promo reels are cut for exactly this purpose, so they beat raw live footage.
_PROMO_RE = re.compile(r"\b(promo|trailer|demo|sizzle|reel|highlights?)\b", re.I)

# Below this an act's "video" is usually a shout-out or a stinger, above it we
# are downloading a full concert to use fifteen seconds of it. The floor is well
# above the clip length because a 30-second promo is mostly titles and an end
# card, leaving nothing usable once both are skipped.
_MIN_DURATION = 60
_MAX_DURATION = 900

# Cuts made to advertise one booking carry a burned-in date and venue. Reposting
# those advertises a show that has already happened, so they are skipped in
# favour of the evergreen reel. Matched against the video title.
_DATED_AD_RE = re.compile(
    r"\b(on sale|tickets?|presale|tonight|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
    r"|20\d\d\s*(show|date|night)|shout[- ]?out|apap|iaapa|ieba)\b",
    re.I,
)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.VIMEO_ACCESS_TOKEN}",
        "Accept": _HEADERS_ACCEPT,
    }


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _names_act(title: str, act: str) -> bool:
    """True if a video title actually names this act.

    Tokens of 4+ characters from the act name, ignoring the descriptor after a
    colon, with the filing-order article dropped. A majority must appear.
    """
    from .artist_links import display_act

    head = display_act(act).split(":")[0]
    tokens = [t for t in re.split(r"[^a-z0-9]+", head.lower()) if len(t) > 3]
    if not tokens:
        return False
    low = title.lower()
    return sum(t in low for t in tokens) >= max(1, (len(tokens) + 1) // 2)


def _title_year(title: str) -> int:
    """A 4-digit year in the title, or 0. Used to prefer the newest promo."""
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", title or "")]
    return max(years) if years else 0


def find_act_video(act: str) -> dict | None:
    """Return the best Vimeo video for an act, or None.

    Searches the agency's own library by act name and prefers a promo/trailer of
    a usable length. Returns a dict with ``name``, ``duration``, ``link`` and
    ``download`` (the list of progressive renditions).
    """
    if not config.VIMEO_ACCESS_TOKEN or not act.strip():
        return None

    # Act names carry subtitles the video titles rarely repeat ("Arrival From
    # Sweden: The Music of ABBA" vs "ARRIVAL from Sweden at Red Rocks"), so
    # search on the part before the colon.
    query = act.split(":")[0].strip()
    try:
        resp = requests.get(
            f"{API}/me/videos",
            headers=_headers(),
            params={
                "query": query,
                "per_page": 25,
                "fields": "name,duration,link,download",
            },
            timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001, video is a nice-to-have, never fatal
        log.warning("Vimeo search failed for %s: %s", act, exc)
        return None

    if not resp.ok:
        log.warning("Vimeo search %d for %s: %s", resp.status_code, act, resp.text[:200])
        return None

    usable = [
        v for v in (resp.json().get("data") or [])
        if v.get("download")
        and _MIN_DURATION <= (v.get("duration") or 0) <= _MAX_DURATION
        and not _DATED_AD_RE.search(v.get("name", ""))
    ]
    if not usable:
        log.info("No usable Vimeo video for %s", act)
        return None

    # The Vimeo query is a full-text search over titles, descriptions and tags,
    # so it happily returns other acts. Searching "Platters" on 2026-08-03
    # returned "Phil Dirt and the Dozers" and a package-show sizzle reel
    # alongside the act's own promos. Posting another act's footage under this
    # act's name is the worst thing this function can do, so a title that does
    # not name the act is not a candidate at all: no clip beats a wrong clip.
    named = [v for v in usable if _names_act(v.get("name", ""), act)]
    if not named:
        log.info(
            "No Vimeo video titled for %s (%d search hit(s) were other acts), skipping",
            act, len(usable),
        )
        return None

    # Promo first; among promos (and among non-promos) prefer the shorter cut,
    # which is nearly always the tighter, more recent edit.
    #
    # Recency outranks both: an act re-shoots its promo when the line-up changes,
    # so the newest titled promo is the one showing who is actually on stage.
    # Without this the 2025 sizzle beat "The Platters 2026 Promo Video" purely by
    # being 49 seconds shorter, and the client spotted a stale line-up in it.
    named.sort(key=lambda v: (
        -_title_year(v.get("name", "")),
        not _PROMO_RE.search(v.get("name", "")),
        v.get("duration", 0),
    ))
    best = named[0]
    log.info("Vimeo pick for %s: %s (%ss)", act, best.get("name", "")[:60], best.get("duration"))
    return best


def _best_rendition(video: dict, max_height: int = 720) -> str | None:
    """Best progressive link at or below ``max_height``, smallest file first.

    The ``source`` rendition is deliberately excluded. Vimeo reports it at the
    same height as the transcoded copy but it is the untouched master: 940MB
    against 234MB for one act's promo, for a clip that ends up a few megabytes.
    Same picture, four times the transfer.

    Among the rest the smallest acceptable file wins, because ffmpeg only pulls
    the byte ranges it needs and the clip is re-encoded at 1080 square anyway.
    """
    scored = []
    for f in video.get("download") or []:
        if (f.get("quality") or "").lower() == "source":
            continue
        height = f.get("height") or 0
        if height and height > max_height:
            continue
        if f.get("link"):
            scored.append((f.get("size") or 0, height, f["link"]))
    if not scored:
        return None
    # Prefer at least 540 where available; below that the re-encode looks soft.
    good = [s for s in scored if s[1] >= 540] or scored
    good.sort()
    return good[0][2]


def download_source(video: dict, dest: str, max_height: int = 720) -> str | None:
    """Download a progressive rendition to ``dest``. Returns the path or None."""
    link = _best_rendition(video, max_height)
    if not link:
        log.info("No downloadable rendition for %s", video.get("name", ""))
        return None
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest  # cached from an earlier run

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        with requests.get(link, stream=True, timeout=120) as r:
            if not r.ok:
                log.warning("Vimeo download %d for %s", r.status_code, video.get("name", ""))
                return None
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    except Exception as exc:  # noqa: BLE001
        log.warning("Vimeo download failed for %s: %s", video.get("name", ""), exc)
        return None

    log.info("Downloaded %s (%.1f MB)", os.path.basename(dest), os.path.getsize(dest) / 1e6)
    return dest


def make_clip(
    src: str,
    dest: str,
    *,
    start: float = 0.0,
    duration: float = 15.0,
    square: bool = True,
) -> str | None:
    """Cut a short, social-ready clip from ``src`` with ffmpeg.

    Square by default: it is the aspect that survives both the Instagram feed
    and a LinkedIn scroll without the platform cropping the performer's head
    off. Audio is kept, since a music act muted is pointless, and the output is
    faststart-encoded so it begins playing before it has fully buffered.
    """
    if not ffmpeg_available():
        log.warning("ffmpeg not on PATH, cannot cut clips")
        return None
    is_url = src.startswith(("http://", "https://"))
    if not is_url and not os.path.exists(src):
        return None
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest  # deterministic output, already cut on an earlier run

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Fit the whole 16:9 frame into the square over a blurred copy of itself,
    # rather than centre-cropping. These promos carry burned-in titles and lower
    # thirds, and a centre crop slices them in half ("...OF ABBA / ...IVAL /
    # ...WEDEN"), which looks broken. Padding keeps every frame intact and the
    # blur fills the bars so it does not read as letterboxing.
    vf = (
        "split[bg][fg];"
        "[bg]scale=1080:1080:force_original_aspect_ratio=increase,"
        "crop=1080:1080,gblur=sigma=28,eq=brightness=-0.12[b];"
        "[fg]scale=1080:1080:force_original_aspect_ratio=decrease[f];"
        "[b][f]overlay=(W-w)/2:(H-h)/2"
    ) if square else "scale=-2:1080"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-i", src, "-t", f"{duration:.2f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        dest,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timed out cutting %s", os.path.basename(dest))
        return None

    if proc.returncode != 0 or not os.path.exists(dest):
        log.warning("ffmpeg failed: %s", (proc.stderr or "")[:300])
        return None

    log.info("Cut clip %s (%.1f MB)", os.path.basename(dest), os.path.getsize(dest) / 1e6)
    return dest


def clip_for_act(act: str, *, duration: float = 15.0, skip_intro: float = 10.0) -> str | None:
    """Find and cut one clip for an act. Returns the mp4 path or None.

    The rendition URL is handed straight to ffmpeg rather than downloaded first.
    ffmpeg seeks over HTTP range requests, so cutting 15 seconds transfers a few
    megabytes instead of the whole promo. ``download_source`` remains available
    for the rare case where a source really is needed on disk.

    ``skip_intro`` trims the opening seconds, which on a promo reel are usually
    a title card or a fade rather than the performance.
    """
    video = find_act_video(act)
    if not video:
        return None
    link = _best_rendition(video)
    if not link:
        log.info("No usable rendition for %s", act)
        return None

    # These promos open with the agency's animated logo, so a few seconds of
    # skip is not enough: a clip starting at 3s is still showing the logo.
    # Starting a fixed distance in also lands on the strongest footage, since a
    # reel front-loads titles and saves the performance for the body. The cap
    # keeps the clip clear of the end card.
    total = video.get("duration") or 0
    start = max(skip_intro, total * 0.15)
    if start + duration > total - 3:
        start = max(0.0, total - duration - 3)
    slug = re.sub(r"[^a-z0-9]+", "-", act.lower()).strip("-")[:40]
    dest = os.path.join(config.VIDEO_DIR, f"clip_{slug}_{int(start)}_{int(duration)}.mp4")
    return make_clip(link, dest, start=start, duration=duration)

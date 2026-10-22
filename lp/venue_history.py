"""Raise a re-booking count from Airtable's floor to the real number.

Airtable's `Weekly LPC Central` only reaches back to 2023, so a venue that has
brought an act back for a decade reads as 2x or 3x. The client knows the real
numbers are bigger and wants them on the graphic, so this module goes looking
for the missing dates.

**The source is the act's own published date list, never a fan database and
never a model's count.** Both of those were tried and both were wrong:
setlist.fm has Arrival From Sweden at 82 shows against the ~140 USA tours its
own site claims, and a WebFetch summariser asked to count Free Fallin's Le
Musique rows invented a date and dropped a real one (see CLAUDE.md). What does
work is parsing the act's own page, plus every Wayback snapshot of it, which is
how Arrival's Arlington Music Hall count went from Airtable's 3 to a verified 7.

Two consequences shape the design:

- **`ACT_TOUR_PAGES` is an allowlist, and a missing act returns None.** Most
  acts publish nothing. Guessing a URL and parsing whatever comes back is how a
  wrong number reaches a graphic published under the agency's name, so this
  fails closed exactly like `verify_quote_on_page()` and `ACT_INSTAGRAM`.
- **Every returned date carries the page it was read from.** The count is
  logged in full and a human can check each row. Nothing here is trusted enough
  to publish unseen.

This module only *reads* the act's pages. Deciding what a post may claim lives
in ``lp.date_check``, which combines what is found here with a search for the
dates only Airtable knows about.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date

import requests

log = logging.getLogger(__name__)

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Acts that publish a date list we can parse, keyed by the Airtable name.
#
# Each entry is (url, kind):
#   "archive"  the page already lists past dates, read it as-is
#   "upcoming" the page only shows what is coming, so its history lives in
#              Wayback snapshots and we walk those too
#   "paged"    an archive behind ?calendar_page_prev=N pagination (Bandzoogle)
#
# Add an act here only after opening the page and confirming it lists dated
# rows with venue names. Verified 2026-08-17.
ACT_TOUR_PAGES: dict[str, tuple[tuple[str, str], ...]] = {
    "Arrival from Sweden: The Music of ABBA": (
        ("https://www.themusicofabba.com/previous-shows/", "archive"),
        ("https://www.themusicofabba.com/tourtickets/", "upcoming"),
    ),
    "The Rocket Man Show": (
        ("https://rocketmanshow.com/dates", "paged"),
    ),
    "Free Fallin: The Tom Petty Concert Experience": (
        ("https://freefallin.us/tour-history", "archive"),
    ),
}

# A row carrying one of these is an announcement that did not happen. Arrival's
# own archive still lists its two COVID-era Arlington dates, both marked, and
# counting them would have made a true 7 into a false 9.
_DEAD_ROW_RE = re.compile(r"postpone|cancel|rescheduled\s+to|tba\b", re.I)

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
# "February 8 2024", "Feb 8, 2024", "Fri, Apr 4, 2025". Day-first and ISO forms
# are handled separately below.
_DATE_RE = re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d\d)\b", re.I)
_ISO_RE = re.compile(r"\b(20\d\d)-(\d{2})-(\d{2})\b")
# "April 25th", with the year carried by a section heading further up the page.
# The ordinal suffix is optional because the same page mixes both forms, and one
# unrecognised "Dec 02" is not a missing row, it is a *wrong* one: the row above
# it then runs on until the next date it does recognise and picks up whatever
# venue it finds on the way.
_BARE_DAY_RE = re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)

# "Friday, April 24 @ 7:30PM", the shape a Bandzoogle calendar's first page
# uses: no year anywhere on the page, because everything on it is upcoming. The
# weekday is what recovers the year, and it is self-checking: only one year in a
# three-year window puts April 24 on a Friday.
_WEEKDAY_DAY_RE = re.compile(
    rf"\b(Mon|Tue|Tues|Wed|Wednes|Thu|Thur|Thurs|Fri|Sat|Satur|Sun)(?:day)?\.?,?\s+"
    rf"({_MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
    re.I,
)
_WEEKDAY_NUM = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "wednes": 2, "thu": 3,
                "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "satur": 5, "sun": 6}

_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# How much text after a date is treated as that row's venue and city. Long
# enough to clear a repeated mobile/desktop rendering of the same row (the
# Bandzoogle template prints each one twice), short enough not to reach the
# next row's venue.
_WINDOW = 200


def _norm_venue(s: str) -> str:
    """Normalize a venue name for comparison.

    Same failure this fixes elsewhere in the engine: a leading article, a
    trailing descriptor and inconsistent spacing split one venue into several.
    "The Arlington Music Hall" and "Arlington Music Hall" are one room.
    """
    s = re.sub(r"[‘’“”]", "'", s or "").casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"^(the|a)\s+", "", s)
    return re.sub(r"\s+", " ", s)


# Words that name a *kind* of room rather than the room. A venue is billed
# inconsistently across its own listings ("Agua Caliente Casino" on one page,
# "Agua Caliente Resort" on the next), which is the same failure that had Free
# Fallin's Le Musique count reading 6 when it was 12: count on a normalized
# name, never on the raw string.
_VENUE_GENERIC = {
    "casino", "resort", "theatre", "theater", "center", "centre", "hall",
    "arena", "auditorium", "spa", "complex", "club", "room", "playhouse",
    "pavilion", "amphitheater", "amphitheatre", "opry", "lounge", "showroom",
    "the show", "performing", "arts", "and", "of", "at",
}


def _venue_core(name: str) -> list[str]:
    """The tokens that actually identify a venue, generic room words removed."""
    return [t for t in _norm_venue(name).split() if t not in _VENUE_GENERIC]


def _venue_matches(target: str, window: str) -> bool:
    """True when a row's text names the venue we are counting.

    Full normalized name first. Only if that misses does it fall back to the
    identifying tokens, which must appear **together and in order**, so "Rivers
    Pittsburgh" cannot match a Rivers Philadelphia row.

    **A one-token core falls back to the full name, and that is not fussiness.**
    "Carolina Opry" reduces to "carolina", which matches every row in North
    Carolina: it silently turned a true 11 into 15 the first time this ran. Two
    tokens or the whole name, nothing looser.
    """
    text = _norm_venue(window)
    if _norm_venue(target) in text:
        return True
    core = _venue_core(target)
    return len(core) >= 2 and " ".join(core) in text


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    # Row and cell boundaries become separators so two adjacent cells cannot
    # fuse into one unparseable word.
    html = re.sub(r"(?i)</(tr|td|th|li|p|div|br)>|<br\s*/?>", " | ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    for ent, ch in (("&amp;", "&"), ("&nbsp;", " "), ("&#8217;", "'"),
                    ("&quot;", '"'), ("&#39;", "'"), ("&ndash;", "-")):
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text)


def _rows(text: str) -> list[tuple[date, str]]:
    """Return (date, its own row's text) for every date found in a flattened page.

    Every act page checked prints the venue *after* the date, so the row is the
    text between one date and the next. Deliberately dumb about table structure,
    which differs on all three sites.

    **The window has to stop at the next date, not at a fixed length.** A flat
    200 characters ran past the end of short rows into the following one, so
    every date listed the night before an Arlington show counted as an Arlington
    show. That turned a true 7 into 20 on the first run.
    """
    marks: list[tuple[int, int, date]] = []
    covered: list[tuple[int, int]] = []
    for m in _DATE_RE.finditer(text):
        mon = (_MONTH_NUM.get(m.group(1)[:4].lower().rstrip("."))
               or _MONTH_NUM.get(m.group(1)[:3].lower()))
        if not mon:
            continue
        try:
            marks.append((m.start(), m.end(), date(int(m.group(3)), mon, int(m.group(2)))))
            covered.append((m.start(), m.end()))
        except ValueError:
            continue
    for m in _ISO_RE.finditer(text):
        try:
            marks.append((m.start(), m.end(),
                          date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
            covered.append((m.start(), m.end()))
        except ValueError:
            continue

    # Year-in-the-heading pages. Free Fallin's tour history reads "2024" as a
    # section header and then "April 25th - Le Musique Room", so a rule that
    # demands a year on every row finds nothing at all on it. A bare month and
    # day inherits the nearest year printed above it.
    def _inside(pos: int) -> bool:
        return any(a <= pos < b for a, b in covered)

    years = [(m.start(), int(m.group(1))) for m in re.finditer(r"\b(20\d\d)\b", text)
             if not _inside(m.start())]
    if years:
        for m in _BARE_DAY_RE.finditer(text):
            if _inside(m.start()):
                continue
            mon = (_MONTH_NUM.get(m.group(1)[:4].lower().rstrip("."))
                   or _MONTH_NUM.get(m.group(1)[:3].lower()))
            prior = [y for pos, y in years if pos < m.start()]
            if not mon or not prior:
                continue
            try:
                marks.append((m.start(), m.end(), date(prior[-1], mon, int(m.group(2)))))
            except ValueError:
                continue
    # A page with no year printed on it at all. Nothing above resolves those
    # rows, so they were silently dropped: that is how The Rocket Man Show's
    # April 24 2026 date at Agua Caliente went missing from a count built to
    # check exactly that pairing. The weekday supplies the year, and a date that
    # does not fall on the weekday the page prints is not used at all.
    if not years:
        today = date.today()
        for m in _WEEKDAY_DAY_RE.finditer(text):
            if _inside(m.start()):
                continue
            mon = (_MONTH_NUM.get(m.group(2)[:4].lower().rstrip("."))
                   or _MONTH_NUM.get(m.group(2)[:3].lower()))
            want = _WEEKDAY_NUM.get(m.group(1).lower())
            if mon is None or want is None:
                continue
            hits = []
            for year in range(today.year - 1, today.year + 2):
                try:
                    cand = date(year, mon, int(m.group(3)))
                except ValueError:
                    continue
                if cand.weekday() == want:
                    hits.append(cand)
            # Ambiguity means we cannot say which year it was, so we do not guess.
            if len(hits) == 1:
                marks.append((m.start(), m.end(), hits[0]))

    marks.sort()
    out: list[tuple[date, str]] = []
    for i, (start, end, d) in enumerate(marks):
        # The row runs to the next date's *start*, capped so a page that lists
        # one date followed by prose does not swallow the whole page.
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((d, text[end:min(stop, end + _WINDOW)]))
    return out


def _get(url: str, timeout: int = 30) -> str | None:
    try:
        r = requests.get(url, headers=_UA, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        # web.archive.org refuses TLS from some networks while serving the same
        # path fine over http, and it throttles hard, and both failures are
        # invisible: an empty snapshot list is indistinguishable from an act
        # with no archived page, so a count quietly comes back short. Retry in
        # the clear, with a pause, before believing there is nothing there.
        if url.startswith("https://web.archive.org"):
            for attempt in range(3):
                time.sleep(1.5 * attempt)
                try:
                    r = requests.get("http" + url[5:], headers=_UA, timeout=timeout)
                    if r.status_code == 200:
                        return r.text
                except Exception:
                    continue
        return None


def _wayback_snapshots(url: str, limit: int = 200) -> list[str]:
    """Monthly-collapsed Wayback timestamps for a URL, oldest first.

    Monthly is the right granularity: an act announces a date months ahead and
    leaves it up, so one snapshot a month catches every listing without pulling
    the same page fifty times.
    """
    bare = re.sub(r"^https?://", "", url).rstrip("/")
    cdx = (
        "https://web.archive.org/cdx/search/cdx?url=" + bare
        + "*&output=text&fl=timestamp&collapse=timestamp:6&limit=" + str(limit)
    )
    body = _get(cdx, timeout=45)
    if not body:
        log.warning("No Wayback snapshots returned for %s (archive unreachable "
                    "or nothing archived); the count may come back short", url)
        return []
    return [ln.strip() for ln in body.splitlines() if ln.strip().isdigit()]


def _paged_pages(url: str, max_pages: int = 40) -> list[str]:
    """Walk a Bandzoogle ?calendar_page_prev=N archive until it repeats."""
    pages, seen = [], set()
    for i in range(1, max_pages + 1):
        html = _get(f"{url}?calendar_page_prev={i}")
        if not html:
            break
        text = _strip_html(html)
        # The template renders a fixed header and footer, so the tail is what
        # distinguishes one page of events from another.
        fingerprint = text[-4000:]
        if fingerprint in seen:
            break
        seen.add(fingerprint)
        pages.append(text)
        time.sleep(0.3)
    return pages


def _cache_path(act: str, venue: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{act}-{venue}".casefold()).strip("-")
    return os.path.join(_cache_dir(), f"{slug}.json")


def _cache_dir() -> str:
    d = os.getenv("VENUE_HISTORY_CACHE_DIR", "cache/venue-history")
    os.makedirs(d, exist_ok=True)
    return d


def venue_history(act: str, venue: str, max_wayback: int = 200) -> dict | None:
    """Count an act's dates at one venue from the act's own published listings.

    Returns ``{"count", "dates", "sources", "first_year", "last_year"}`` or
    ``None`` when the act publishes nothing we can read, which is the common
    case and not an error.

    Postponed and cancelled rows are dropped, and dates are deduped, so the
    count is performances rather than announcements. Results are cached on disk
    because walking a decade of Wayback snapshots is slow and the answer only
    changes when the act plays there again.
    """
    pages = None
    for name, urls in ACT_TOUR_PAGES.items():
        if _norm_venue(name) == _norm_venue(act) or name.casefold() == act.casefold():
            pages = urls
            break
    if not pages:
        log.info("No published date archive known for '%s', leaving the count alone", act)
        return None

    cache = _cache_path(act, venue)
    if os.path.exists(cache):
        try:
            cached = json.load(open(cache))
            log.info("Venue history for %s at %s: %dx (cached)",
                     act, venue, cached["count"])
            return cached
        except Exception:
            pass

    target = _norm_venue(venue)
    if not target:
        return None
    found: dict[date, str] = {}
    # A date marked cancelled or postponed on ANY page we read, which is not
    # necessarily the page that also announced it. See the removal below.
    killed: set[date] = set()

    def _scan(text: str, source: str) -> None:
        for d, window in _rows(text):
            if not _venue_matches(venue, window):
                continue
            if _DEAD_ROW_RE.search(window[:120]):
                killed.add(d)
                continue
            found.setdefault(d, source)

    for url, kind in pages:
        if kind == "paged":
            for text in _paged_pages(url):
                _scan(text, url)
        else:
            html = _get(url, timeout=45)
            if html:
                _scan(_strip_html(html), url)
        if kind in ("upcoming", "paged"):
            # An only-upcoming page obviously hides its history in the archive.
            # A *paged* archive needs the same treatment for a less obvious
            # reason: it drops rows. The Rocket Man Show's own archive no longer
            # lists the March 2024 Agua Caliente date it was advertising in
            # February 2024, and that snapshot is the only published proof of a
            # date LP holds a contract for.
            for ts in _wayback_snapshots(url, limit=max_wayback):
                snap = f"https://web.archive.org/web/{ts}id_/{url}"
                html = _get(snap, timeout=45)
                if html:
                    _scan(_strip_html(html), snap)
                time.sleep(0.4)

    # Cancelled beats announced, whichever page says so. Skipping a dead row
    # where it is found is not enough once Wayback is in the mix: Arrival's
    # archive marks its 2020-07-26 Carolina Opry date CANCELLED and its
    # 2021-03-07 and 2022-02-20 dates postponed, but a 2020 snapshot of the
    # act's upcoming page announced all three with no marker, and the snapshot
    # then put them back. That read as 14 where the true number is 11. An
    # announcement is not evidence a night happened; a cancellation notice is
    # evidence it did not.
    for d in killed & set(found):
        log.info("    %s cancelled or postponed elsewhere in the act's listings, "
                 "not counted", d)
        found.pop(d)
    dropped = len(killed)

    if not found:
        log.info("No dates found for %s at %s in its own listings", act, venue)
        return None

    dates = sorted(found)
    result = {
        "count":      len(dates),
        "dates":      [d.isoformat() for d in dates],
        "sources":    sorted({v for v in found.values()}),
        "first_year": dates[0].year,
        "last_year":  dates[-1].year,
    }
    log.info(
        "Venue history for %s at %s: %dx (%d to %d), %d postponed/cancelled rows dropped",
        act, venue, result["count"], result["first_year"], result["last_year"], dropped,
    )
    for d in result["dates"]:
        log.info("    %s", d)
    try:
        json.dump(result, open(cache, "w"), indent=2)
    except Exception:
        pass
    return result

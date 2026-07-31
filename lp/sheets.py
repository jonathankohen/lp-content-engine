import logging
import os
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

from .artist_links import display_act
from .config import SHEETS_ID, TOUR_DATES_SHEET_ID

log = logging.getLogger(__name__)

SHEETS_HEADER      = ["artist", "original_artist", "headline", "url", "date_added"]
SHEETS_SHOW_HEADER = ["artist", "show_date", "venue", "url", "date_added"]


def _get_sheet():
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        log.error("GOOGLE_APPLICATION_CREDENTIALS not set")
        return None
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(SHEETS_ID).sheet1


def read_used_topics() -> set[str]:
    """Return set of all URLs and headlines already in the sheet."""
    sheet = _get_sheet()
    if not sheet:
        return set()
    rows = sheet.get_all_values()
    used: set[str] = set()
    for row in rows[1:]:  # skip header
        if len(row) > 3 and row[3]:
            used.add(row[3].strip())  # url column
        if len(row) > 2 and row[2]:
            used.add(row[2].strip())  # headline column as fallback
    return used


def mark_show_used(show: dict, lpc_key: str, dry_run: bool = False) -> None:
    if dry_run:
        log.info("[dry-run] Would record show in Sheets: %s", lpc_key)
        return
    sheet = _get_sheet()
    if not sheet:
        return
    if not sheet.get_all_values():
        sheet.append_row(SHEETS_SHOW_HEADER)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sheet.append_row([
        show.get("show_title", ""),
        show.get("show_date", ""),
        show.get("venue_address", ""),
        lpc_key,
        today,
    ])
    log.info("Recorded show in Sheets: %s", lpc_key)


# Per-run cache of every tour-dates tab, {tab title: rows minus header}.
#
# This used to be a per-show-title cache that reopened the spreadsheet on each
# miss. Looping the roster then issued a fresh open plus a worksheet list per
# act, which trips gspread's read quota partway through 24 acts. The failure is
# silent: the exception is swallowed and the act gets [], indistinguishable from
# an act genuinely having no dates, so posters and carousel slides went missing
# for whichever acts happened to come later in the loop. One fetch, reused.
_tour_tabs_cache: dict[str, list[list[str]]] | None = None


def load_tour_tabs(refresh: bool = False) -> dict[str, list[list[str]]]:
    """Every tab of the tour dates sheet, {tab title: rows minus header}.

    Fetched once per run. Returns {} when the sheet or credentials are
    unavailable, so callers degrade to "no dates" rather than raising.
    """
    global _tour_tabs_cache
    if _tour_tabs_cache is not None and not refresh:
        return _tour_tabs_cache

    tabs: dict[str, list[list[str]]] = {}
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if TOUR_DATES_SHEET_ID and creds_path:
        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
            spreadsheet = gspread.authorize(creds).open_by_key(TOUR_DATES_SHEET_ID)
            for ws in spreadsheet.worksheets():
                try:
                    tabs[ws.title.strip()] = ws.get_all_values()[1:]
                except Exception as exc:  # noqa: BLE001, one bad tab, not the run
                    log.debug("Could not read tour dates tab %r: %s", ws.title, exc)
        except Exception as exc:  # noqa: BLE001
            log.debug("Tour dates sheet unavailable: %s", exc)

    _tour_tabs_cache = tabs
    return tabs


def _tour_tab_rows(show_title: str) -> list[list[str]]:
    """Rows (minus the header) of the tour-dates tab matching a show title.

    Tabs are named by display name (e.g. "Arrival from Sweden" for "Arrival from
    Sweden: The Music of ABBA"), so the match is a case-insensitive substring.
    Returns [] when the sheet, credentials, or tab are unavailable.
    """
    if not (show_title or "").strip():
        return []

    # Airtable files two acts alphabetically ("Platters, The") while the sheet
    # tabs are named the natural way ("The Platters"), so a plain substring test
    # matched neither direction and both acts silently resolved to zero dates.
    # They were missing from every tour poster and carousel until 2026-07-31.
    # Both spellings are tried, longest tab first so a short tab name cannot
    # claim an act whose name merely contains it.
    candidates = {show_title.lower(), display_act(show_title).lower()}
    for tab, rows in sorted(load_tour_tabs().items(), key=lambda kv: -len(kv[0])):
        name = tab.lower()
        if any(name in title or title.startswith(name) for title in candidates):
            return rows

    log.debug("No tour dates tab found for '%s'", show_title)
    return []


# The tour sheet is hand-maintained, so a region arrives as "NY" on one row and
# "New York" on the next. Cards need one consistent form or the column looks
# broken, and the same show entered both ways would otherwise survive dedup.
_STATE_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}


def _normalize_region(region: str) -> str:
    """'New York' and 'ny' both become 'NY'; anything else is passed through."""
    cleaned = region.strip()
    return _STATE_ABBR.get(cleaned.lower(), cleaned.upper() if len(cleaned) == 2 else cleaned)


def collapse_residencies(dates: list[dict]) -> list[dict]:
    """Merge consecutive dates at one venue into a single row with a date range.

    A residency inside a tour ("Sep 8 to 14 at Cafe Carlyle") is eight identical
    rows on a poster, which crowds out the other cities and reads as a rendering
    fault. Real tour admats print it as one line, so this does the same: the
    merged row keeps the first date, gains ``date_end``, and the renderer can
    show "SEP 08-14". Rows that are not part of a run pass through untouched.

    Consecutive is judged by position in the (already date-sorted) list, not by
    calendar adjacency, so a two-night stand with a night off still merges.
    """
    out: list[dict] = []
    for item in dates:
        key = (item.get("venue", "").strip().lower(), item.get("city", "").strip().lower())
        if out:
            prev = out[-1]
            prev_key = (prev.get("venue", "").strip().lower(), prev.get("city", "").strip().lower())
            if key == prev_key and key != ("", ""):
                prev["date_end"] = item["date"]
                prev["dates_merged"] = prev.get("dates_merged", 1) + 1
                continue
        out.append(dict(item))
    return out


def upcoming_tour_dates(show_title: str, limit: int = 14) -> list[dict]:
    """Future dates for an act from the tour dates sheet, soonest first.

    Feeds the tour posters and the "tours on sale now" carousel (both
    ``lp.cards.render_tour_poster``),
    which is modelled on the post the client sent as the format to emulate: an
    act's date list as a poster, one slide per act.

    Each item is ``{date, venue, city, region, ticket_url}`` with ``date`` a
    ``datetime.date``. Rows without a parseable date, and dates in the past, are
    dropped. Returns [] when the sheet or tab is unavailable.
    """
    today = datetime.now().date()
    out = []
    for row in _tour_tab_rows(show_title):
        if not row or not row[0].strip():
            continue
        try:
            when = datetime.strptime(row[0].strip(), "%m/%d/%y").date()
        except ValueError:
            continue
        if when < today:
            continue

        def cell(i: int) -> str:
            return row[i].strip() if len(row) > i else ""

        out.append({
            "date":       when,
            "venue":      cell(1),
            "city":       cell(2),
            "region":     _normalize_region(cell(3)),
            "ticket_url": cell(5),
        })

    # The same show is sometimes entered twice (once as "Schenectady, NY" and
    # once as "Schenectady, New York"), which reads as a double booking on a
    # card. Region normalisation above makes those collide here.
    seen, deduped = set(), []
    for item in sorted(out, key=lambda d: (d["date"], not d["ticket_url"])):
        key = (item["date"], item["city"].lower(), item["region"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[:limit]


def lookup_ticket_url(show_title: str, show_date: str) -> tuple[str | None, str | None]:
    """Look up ticket URL and venue name for a show in the tour dates sheet.

    Returns (ticket_url, venue_name), either may be None if not found.
    """
    try:
        target_str = datetime.strptime(show_date, "%Y-%m-%d").strftime("%m/%d/%y")
    except ValueError:
        return (None, None)
    for row in _tour_tab_rows(show_title):
        if row and row[0].strip() == target_str:
            url = row[5].strip() if len(row) > 5 else ""
            venue_name = row[1].strip() if len(row) > 1 else ""
            return (url or None, venue_name or None)
    return (None, None)


def lookup_venue_name(show_title: str, show_dates: list[str]) -> str | None:
    """Venue name for an act on any of the given dates, from the tour dates sheet.

    Used by the re-booking scan, where the point of the post is naming the venue,
    so any one of the pairing's dates that resolves is good enough. ``show_dates``
    are ISO (YYYY-MM-DD). Returns None when nothing matches.
    """
    targets = set()
    for d in show_dates:
        try:
            targets.add(datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%y"))
        except ValueError:
            continue
    if not targets:
        return None
    for row in _tour_tab_rows(show_title):
        if row and row[0].strip() in targets and len(row) > 1 and row[1].strip():
            return row[1].strip()
    return None


def mark_topics_used(topics: list[dict], dry_run: bool = False) -> None:
    if not topics:
        return
    if dry_run:
        log.info("[dry-run] Would mark %d topics as used in Sheets", len(topics))
        return
    sheet = _get_sheet()
    if not sheet:
        return
    if not sheet.get_all_values():
        sheet.append_row(SHEETS_HEADER)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for t in topics:
        sheet.append_row([
            t.get("artist", ""),
            t.get("original_artist", ""),
            t.get("headline", ""),
            # An explicit sheet_key wins over the URL so synthesized topics
            # (spotlights, re-bookings, agency posts) dedup on their own cadence
            # rather than on a shared act-page or homepage URL. Matches topic_key().
            (t.get("sheet_key") or "").strip() or t.get("url", ""),
            today,
        ])
    log.info("Marked %d topics as used in Sheets", len(topics))

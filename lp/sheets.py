import logging
import os
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

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


def lookup_ticket_url(show_title: str, show_date: str) -> tuple[str | None, str | None]:
    """Look up ticket URL and venue name for a show in the tour dates sheet.

    Returns (ticket_url, venue_name) — either may be None if not found.
    """
    if not TOUR_DATES_SHEET_ID:
        return (None, None)
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not creds_path:
        return (None, None)
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(TOUR_DATES_SHEET_ID)

        title_lower = show_title.lower()
        target_ws = None
        for ws in spreadsheet.worksheets():
            tab = ws.title.lower()
            if tab in title_lower or title_lower.startswith(tab):
                target_ws = ws
                break
        if not target_ws:
            log.debug("No tour dates tab found for '%s'", show_title)
            return (None, None)

        target_str = datetime.strptime(show_date, "%Y-%m-%d").strftime("%m/%d/%y")
        for row in target_ws.get_all_values()[1:]:
            if row and row[0].strip() == target_str:
                url = row[5].strip() if len(row) > 5 else ""
                venue_name = row[1].strip() if len(row) > 1 else ""
                return (url or None, venue_name or None)
    except Exception as exc:
        log.debug("Ticket URL lookup failed for '%s': %s", show_title, exc)
    return (None, None)


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
            t.get("url", ""),
            today,
        ])
    log.info("Marked %d topics as used in Sheets", len(topics))

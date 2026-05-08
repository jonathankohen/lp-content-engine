import logging
from datetime import datetime, timedelta, timezone

import requests

from .config import (
    AIRTABLE_API_KEY,
    AIRTABLE_BASE_ID,
    AIRTABLE_CALENDAR_BASE_ID,
    AIRTABLE_CALENDAR_TABLE_ID,
    AIRTABLE_PRIORITY_ORDER,
    AIRTABLE_TABLE_ID,
    SHOW_DAYS_AHEAD,
)

log = logging.getLogger(__name__)


def fetch_airtable_artists() -> list[dict]:
    """Fetch artists filtered by Marketing Priority, sorted by priority order."""
    priority_filter = ", ".join(
        f"{{Marketing Priority}}='{p}'" for p in AIRTABLE_PRIORITY_ORDER
    )
    params = {
        "fields[]": ["Artist / Show Name", "Marketing Priority"],
        "filterByFormula": f"OR({priority_filter})",
    }
    try:
        resp = requests.get(
            f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}",
            headers={"Authorization": f"Bearer {AIRTABLE_API_KEY}"},
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.error("Airtable fetch error: %s", exc)
        return []

    def _priority_key(record: dict) -> int:
        p = record["fields"].get("Marketing Priority", "")
        try:
            return AIRTABLE_PRIORITY_ORDER.index(p)
        except ValueError:
            return len(AIRTABLE_PRIORITY_ORDER)

    records = sorted(resp.json().get("records", []), key=_priority_key)
    return [
        {
            "name":     r["fields"].get("Artist / Show Name", ""),
            "priority": r["fields"].get("Marketing Priority", ""),
        }
        for r in records
        if r["fields"].get("Artist / Show Name")
    ]


def fetch_upcoming_shows() -> list[dict]:
    """Return fully-executed shows from the Airtable calendar happening within SHOW_DAYS_AHEAD days."""
    today  = datetime.now(tz=timezone.utc).date()
    cutoff = today + timedelta(days=SHOW_DAYS_AHEAD)
    records: list[dict] = []
    params: dict = {
        "fields[]": ["LPC #", "Show Title", "Show Date", "Venue Address"],
        "filterByFormula": "{LPC Contract Status}='(FE) Fully Executed'",
        "cellFormat": "string",
        "timeZone":   "America/New_York",
        "userLocale": "en-us",
    }
    while True:
        try:
            resp = requests.get(
                f"https://api.airtable.com/v0/{AIRTABLE_CALENDAR_BASE_ID}/{AIRTABLE_CALENDAR_TABLE_ID}",
                headers={"Authorization": f"Bearer {AIRTABLE_API_KEY}"},
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:
            log.error("Airtable calendar fetch error: %s", exc)
            return []
        data = resp.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
        params["offset"] = offset

    def _str(val: object) -> str:
        if isinstance(val, list):
            val = val[0] if val else ""
        return str(val).strip()

    shows = []
    for r in records:
        fields = r.get("fields", {})
        show_date_str = fields.get("Show Date", "")
        if not show_date_str:
            continue
        show_date = None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
            try:
                show_date = datetime.strptime(
                    show_date_str[:10] if fmt == "%Y-%m-%d" else show_date_str, fmt
                ).date()
                break
            except ValueError:
                continue
        if show_date is None:
            log.warning("Could not parse show date: %r", show_date_str)
            continue
        if today <= show_date <= cutoff:
            shows.append({
                "lpc_number":    _str(fields.get("LPC #", r["id"])),
                "show_title":    _str(fields.get("Show Title", "")),
                "show_date":     show_date_str[:10],
                "venue_address": _str(fields.get("Venue Address", "")),
            })

    log.info("Fetched %d upcoming show(s) from Airtable calendar", len(shows))
    return shows


def show_to_topic(show: dict, mappings: dict) -> dict:
    """Convert an Airtable calendar show into a topic dict for generate_posts()."""
    try:
        date_formatted = datetime.strptime(show["show_date"], "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        date_formatted = show["show_date"]
    title = show["show_title"]
    venue = show["venue_address"]
    return {
        "artist":          title,
        "original_artist": mappings.get(title, ""),
        "headline":        f"Upcoming Show: {title} — {venue} — {date_formatted}",
        "url":             f"lpc_{show['lpc_number']}",
        "summary":         f"{title} is performing at {venue} on {date_formatted}. Confirmed booking.",
        "hook_type":       "upcoming_show",
        "ticket_url":      None,
    }

import logging
from datetime import datetime, timedelta, timezone

import requests

from .artist_links import lookup_artist_url
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


def _artist_url_from_fields(fields: dict) -> str:
    """Find an act's loveproductions.com page URL among its Airtable fields.

    The artists table carries the act's site link under a field whose exact name
    varies (e.g. "LPI Web Link"); rather than hard-code it, scan every field's
    value for a loveproductions.com URL. Returns "" when none is present.
    Airtable lookup fields arrive as single-element lists, so those are unwrapped.
    """
    for val in fields.values():
        if isinstance(val, list):
            val = val[0] if val else ""
        s = str(val).strip()
        if "loveproductions.com" in s.lower() and s.lower().startswith("http"):
            return s
    return ""


def fetch_airtable_artists() -> list[dict]:
    """Fetch artists filtered by Marketing Priority, sorted by priority order.

    Returns dicts with name, priority, and artist_url (the act's loveproductions.com
    page, used as a 'Read more' fallback for items with no external source URL).
    All fields are requested (not a fixed subset) so the act page link can be found
    regardless of its exact field name.
    """
    priority_filter = ", ".join(
        f"{{Marketing Priority}}='{p}'" for p in AIRTABLE_PRIORITY_ORDER
    )
    params = {
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
    artists = []
    for r in records:
        name = r["fields"].get("Artist / Show Name", "")
        if not name:
            continue
        # Airtable's own link wins; fall back to the prepopulated static map so a
        # row missing the link field never yields a blank act page at runtime.
        artist_url = _artist_url_from_fields(r["fields"]) or lookup_artist_url(name)
        artists.append({
            "name":       name,
            "priority":   r["fields"].get("Marketing Priority", ""),
            "artist_url": artist_url,
        })
    return artists


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


def fetch_venue_from_contracts(lpc_number: str) -> str | None:
    """Look up the Venue field from the LPI - Contracts table by LPC number."""
    if not AIRTABLE_CALENDAR_BASE_ID or not lpc_number:
        return None
    try:
        resp = requests.get(
            f"https://api.airtable.com/v0/{AIRTABLE_CALENDAR_BASE_ID}/LPI%20-%20Contracts",
            headers={"Authorization": f"Bearer {AIRTABLE_API_KEY}"},
            params={
                "filterByFormula": f"{{LPC #}}='{lpc_number}'",
                "fields[]": "Venue",
                "maxRecords": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        records = resp.json().get("records", [])
        if records:
            return records[0].get("fields", {}).get("Venue") or None
    except Exception as exc:
        log.debug("Contracts venue lookup failed for '%s': %s", lpc_number, exc)
    return None


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
        "url":             "",
        "sheet_key":       f"lpc_{show['lpc_number']}",
        "summary":         f"{title} is performing at {venue} on {date_formatted}. Confirmed booking.",
        "hook_type":       "upcoming_show",
        "ticket_url":      None,
    }

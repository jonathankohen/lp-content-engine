import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

from .artist_links import lookup_artist_url
from .sheets import lookup_venue_name
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


def _str(val: object) -> str:
    """Flatten an Airtable cell (lookups arrive as single-element lists) to a string."""
    if isinstance(val, list):
        val = val[0] if val else ""
    return str(val).strip()


def _parse_show_date(raw: str):
    """Parse a Show Date cell in any of the formats Airtable returns. None if unparseable."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw[:10] if fmt == "%Y-%m-%d" else raw, fmt).date()
        except ValueError:
            continue
    return None


def _fetch_calendar_records() -> list[dict]:
    """Fetch every fully-executed row from the Airtable calendar table (paginated)."""
    records: list[dict] = []
    params: dict = {
        # "Venue" is the venue's *name*. It is populated on every fully-executed
        # row and was simply never requested, which is why re-bookings used to
        # fall back to the tour sheet and the contracts table and drop most
        # pairings for having no name (2026-08-04).
        "fields[]": ["LPC #", "Show Title", "Show Date", "Venue Address", "Venue"],
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
    return records


def fetch_upcoming_shows() -> list[dict]:
    """Return fully-executed shows from the Airtable calendar happening within SHOW_DAYS_AHEAD days."""
    today  = datetime.now(tz=timezone.utc).date()
    cutoff = today + timedelta(days=SHOW_DAYS_AHEAD)

    shows = []
    for r in _fetch_calendar_records():
        fields = r.get("fields", {})
        show_date_str = fields.get("Show Date", "")
        show_date = _parse_show_date(show_date_str)
        if show_date is None:
            if show_date_str:
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


def _slug(text: str) -> str:
    """Lowercase alphanumeric slug, for stable dedup keys."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _norm(text: str) -> str:
    """Normalize an act or venue string for grouping (case and spacing folded)."""
    return " ".join(text.lower().split())


def fetch_rebookings(mappings: dict | None = None, min_bookings: int = 2) -> list[dict]:
    """Return re-booking topics: acts a venue has booked more than once.

    A venue bringing an act back is the strongest proof in live entertainment
    (see ``audience/buyers.md``) and it is already sitting unused in the
    contracts data. Every fully-executed contract is grouped by act + venue;
    any pairing with ``min_bookings`` or more distinct show dates becomes a
    candidate topic.

    The venue must be resolvable to a **name**. A post reading "8901 N Kings Hwy
    has booked them five times" is worthless, so pairings that resolve only to a
    street address are dropped rather than posted.

    **Grouping is by venue name, not by street address (2026-08-04).** The
    calendar table's own ``Venue`` field carries the name on every fully-executed
    row, so it is the primary source and the tour sheet and contracts lookups are
    only a fallback for a blank one. Grouping on the address also split the same
    venue across spelling variants of its address, which undercounted. The two
    changes together took the roster from 7 named pairings to 36, and from one
    pairing at 3+ bookings to seven.

    Returned dicts match the shape :func:`show_to_topic` produces, sorted
    strongest first (most bookings, then most recent).
    """
    mappings = mappings or {}
    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"dates": set(), "lpc": "", "title": "", "venue_raw": "", "venue": ""}
    )

    for r in _fetch_calendar_records():
        fields = r.get("fields", {})
        title = _str(fields.get("Show Title", ""))
        venue_raw = _str(fields.get("Venue Address", ""))
        # Venue names are hand-typed and some carry a line break, where the
        # second line is the parent complex, not part of the name ("New Barn
        # Theater\nRenfro Valley Entertainment Center"). Joining the lines put
        # the whole thing in the middle of a sentence and on a stat card, and the
        # client's correction (2026-08-12) was that the venue is just "New Barn
        # Theater". So keep the FIRST line only. Grouping is unaffected in the
        # cases that matter: two rooms inside one complex still differ on line
        # one, which is the line that names them.
        venue_field = _str(fields.get("Venue", ""))
        first_line = next((ln for ln in venue_field.splitlines() if ln.strip()), "")
        venue = " ".join(first_line.split())
        show_date = _parse_show_date(fields.get("Show Date", ""))
        if not title or not (venue or venue_raw) or show_date is None:
            continue
        # Group on the name when we have one so "Celebrity Theatre" is one venue
        # however its address was typed that week; fall back to the address only
        # for rows with no name at all.
        g = groups[(_norm(title), _norm(venue or venue_raw))]
        g["dates"].add(show_date)
        g["title"] = title
        g["venue"] = g["venue"] or venue
        g["venue_raw"] = g["venue_raw"] or venue_raw
        # Keep any LPC number from the group; used only to look up the venue name.
        g["lpc"] = g["lpc"] or _str(fields.get("LPC #", ""))

    qualifying = [g for g in groups.values() if len(g["dates"]) >= min_bookings]
    # Strongest proof first: most bookings, then whichever ran most recently.
    qualifying.sort(key=lambda g: (len(g["dates"]), max(g["dates"])), reverse=True)

    topics = []
    unnamed = 0
    for g in qualifying:
        title = g["title"]
        dates = sorted(g["dates"])
        venue = (
            g["venue"]
            or lookup_venue_name(title, [d.strftime("%Y-%m-%d") for d in dates])
            or fetch_venue_from_contracts(g["lpc"])
        )
        if not venue:
            unnamed += 1
            log.debug(
                "Re-booking skipped, no venue name for '%s' at %s", title, g["venue_raw"]
            )
            continue
        count = len(dates)
        first_year, last_year = dates[0].year, dates[-1].year
        span = f"{first_year}" if first_year == last_year else f"{first_year} to {last_year}"
        date_list = ", ".join(d.strftime("%B %d, %Y") for d in dates)
        topics.append({
            "artist":          title,
            "original_artist": mappings.get(title, ""),
            "headline":        f"{venue} has booked {title} {count} times ({span})",
            "url":             "",
            "sheet_key":       f"rebook_{_slug(title)}_{_slug(venue)}",
            # No interpretive sentence here. It used to end "a venue bringing an
            # act back is proof the act draws and delivers", which is the model
            # being handed the exact explain-their-own-business line the client
            # rejected (2026-08-03). The facts are the post.
            "summary": (
                f"{venue} has booked {title} {count} separate times between {span}. "
                f"Confirmed dates: {date_list}."
            ),
            "hook_type":       "rebooking",
            "ticket_url":      None,
            "_act":            title,
            "_rebooking_count": count,
            # Kept as discrete fields (not just baked into the prose) so
            # lp.stats can render the number on a card without re-parsing it
            # back out of the summary.
            "_venue":          venue,
            "_span":           span,
        })

    log.info(
        "Found %d re-booked act/venue pairing(s), %d dropped for having no venue name",
        len(topics), unnamed,
    )
    return topics


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
        "headline":        f"Upcoming Show: {title}, {venue}, {date_formatted}",
        "url":             "",
        "sheet_key":       f"lpc_{show['lpc_number']}",
        "summary":         f"{title} is performing at {venue} on {date_formatted}. Confirmed booking.",
        "hook_type":       "upcoming_show",
        "ticket_url":      None,
        # Carried as its own field so lookup_venue_handle() can tag the venue on
        # Instagram. Often a resolved venue name by this point, sometimes still a
        # street address, in which case the lookup simply misses.
        "venue":           venue,
    }

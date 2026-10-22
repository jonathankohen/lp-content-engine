"""Corroborate a re-booking's dates outside Airtable.

**The rule (client direction 2026-08-18): a count that can only be verified
through Airtable is not publishable as-is. It has to be searched for.**

Airtable's `Weekly LPC Central` is the agency's own contract record. It is the
right place to *find* a re-booking and the wrong place to *prove* one: a signed
contract is not evidence the night happened, and a hard number on a graphic goes
out under the agency's name. So every date behind a published count has to be
demonstrable somewhere else.

Three tiers, strongest first, and every one of them is checked in code:

1. The act's own published listings, including Wayback snapshots of them
   (``lp.venue_history``). Free, no model, and it covers most of what we post.
2. A cited page that we fetch and find the date printed on
   (``verify_date_on_page``).
3. A cited URL that spells the date out in the path itself
   (``url_encodes_date``), the one concession to ticket pages that are
   JavaScript-rendered or bot-blocked.

The search only ever *finds* pages. It is never asked whether a date is real,
because a model asked that says yes: it invented a date and dropped a real one
the last time it was asked to read a tour page (see CLAUDE.md).

Uncorroborated dates are dropped from the count rather than argued with, and a
pairing left under two dates is not a re-booking any more, so it is skipped.
Results are cached on disk: the answer only changes when the act plays there
again, and searching is the expensive part of the whole engine.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date

from . import config
from .scrape import url_encodes_date, verify_date_on_page

log = logging.getLogger(__name__)


def _cache_dir() -> str:
    path = os.environ.get("DATE_CHECK_CACHE_DIR", "cache/date-checks")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_path(act: str, venue: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{act}-{venue}".casefold()).strip("-")
    return os.path.join(_cache_dir(), f"{slug}.json")


def _load(act: str, venue: str) -> dict:
    try:
        return json.load(open(_cache_path(act, venue)))
    except Exception:
        return {}


def _save(act: str, venue: str, data: dict) -> None:
    try:
        json.dump(data, open(_cache_path(act, venue), "w"), indent=2, sort_keys=True)
    except Exception:
        pass


def corroborate_dates(
    act: str,
    venue: str,
    dates: list[str],
    known: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return ``{iso date: evidence}`` for the dates that check out.

    ``known`` carries dates already proven by the act's own listings (tier 1),
    which are passed straight through without spending a search on them.
    Anything else is searched for once and the answer cached, including a
    negative one: a date nobody published is not going to become published, and
    re-searching it every week costs a Claude call for the same "no".
    """
    known = known or {}
    out: dict[str, str] = {d: src for d, src in known.items() if d in dates}
    missing = [d for d in dates if d not in out]
    if not missing:
        return out

    cache = _load(act, venue)
    fresh = False
    for iso in missing:
        if iso in cache:
            if cache[iso]:
                out[iso] = cache[iso]
            continue
        cache[iso] = _search_one(act, venue, iso)
        fresh = True
        if cache[iso]:
            out[iso] = cache[iso]
    if fresh:
        _save(act, venue, cache)
    return out


_searches_run = 0


def _search_one(act: str, venue: str, iso: str) -> str:
    """Find and code-verify one date. Returns the evidence URL, or "" for none."""
    global _searches_run
    if _searches_run >= config.DATE_CHECK_MAX_SEARCHES:
        log.warning("  %s at %s on %s: date-check search budget spent (%d), "
                    "date left uncorroborated",
                    act, venue, iso, config.DATE_CHECK_MAX_SEARCHES)
        return ""
    _searches_run += 1
    # Imported here rather than at module scope: this is the only path that
    # spends a Claude call, and lp.ai pulls in the whole generation stack.
    from .ai import search_date_citations

    try:
        day = date.fromisoformat(iso)
    except ValueError:
        return ""

    for url in search_date_citations(act, venue, day.strftime("%B %d, %Y")):
        if verify_date_on_page(url, day, venue):
            log.info("  %s at %s on %s: confirmed on %s", act, venue, iso, url)
            return url
        if url_encodes_date(url, day):
            log.info("  %s at %s on %s: date is in the URL of %s", act, venue, iso, url)
            return url
    log.warning("  %s at %s on %s: NOT corroborated outside Airtable, dropped "
                "from the count", act, venue, iso)
    return ""


def verify_rebooking(topic: dict) -> dict | None:
    """Settle a re-booking topic's number, or drop the topic.

    Does both halves of the job in one place, because they are the same
    question asked in two directions:

    - **Up.** Airtable starts in 2023, so a venue that has brought an act back
      since 2016 reads as 3x. The act's own listings raise it (this is where
      Arrival's Arlington Music Hall count goes from 3 to 7).
    - **Down.** A date only Airtable knows about is dropped, unless a search
      turns up a page that a code check confirms carries it.

    Returns the topic with its count, span, headline and summary rewritten to
    match the evidence, or ``None`` when fewer than two dates survive, which is
    not a re-booking any more.
    """
    from .venue_history import venue_history

    act = (topic.get("_act") or topic.get("artist") or "").strip()
    venue = (topic.get("_venue") or "").strip()
    airtable = list(topic.get("_dates") or [])
    if not act or not venue or not airtable:
        return topic

    hist = venue_history(act, venue) if config.VENUE_HISTORY_MIN_COUNT else None
    # Every date on the act's own page is proof for itself, so it is passed in
    # as known rather than searched for again.
    published = {d: " ".join(hist["sources"]) for d in hist["dates"]} if hist else {}
    checked = corroborate_dates(act, venue, airtable, published)

    evidence = dict(published)
    evidence.update(checked)
    dates = sorted(evidence)
    dropped = [d for d in airtable if d not in evidence]

    if len(dates) < 2:
        log.warning(
            "Re-booking dropped: %s at %s has %d date(s) corroborated outside "
            "Airtable (Airtable claims %d). Uncorroborated: %s",
            act, venue, len(dates), len(airtable), ", ".join(dropped) or "none",
        )
        return None

    if dropped:
        log.warning(
            "Re-booking count cut from Airtable's %d to %d for %s at %s. "
            "No published source found for: %s",
            len(airtable), len(dates), act, venue, ", ".join(dropped),
        )
    elif len(dates) > len(airtable):
        log.info("Re-booking raised from Airtable's %d to %d for %s at %s",
                 len(airtable), len(dates), act, venue)

    first, last = dates[0][:4], dates[-1][:4]
    span = first if first == last else f"{first} to {last}"
    topic = dict(topic)
    topic["_rebooking_count"] = len(dates)
    topic["_span"] = span
    topic["_dates"] = dates
    # Kept in the shape lp.stats already reads, so the card can name what the
    # number was read from rather than asserting it.
    topic["_venue_history"] = {
        "count":   len(dates),
        "dates":   dates,
        "sources": sorted({u for v in evidence.values() for u in v.split() if u}),
    }
    topic["headline"] = f"{venue} has booked {act} {len(dates)} times ({span})"
    topic["summary"] = (
        f"{venue} has booked {act} {len(dates)} separate times between {span}. "
        f"Confirmed dates: {', '.join(dates)}."
    )
    return topic

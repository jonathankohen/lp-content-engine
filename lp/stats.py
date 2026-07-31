"""
Hard numbers for stat cards, from the three sources the client signed off on.

The client's direction (2026-07-31) was "maybe some hard numbers with a graphic".
This module answers the "hard numbers" half: it decides what figure a card leads
with, and where that figure came from. :mod:`lp.cards` draws it.

Every number here traces to something already verified elsewhere. Nothing is
invented, estimated, or rounded up:

* **Re-bookings** come from fully-executed Airtable contracts, counted in
  :func:`lp.airtable.fetch_rebookings`.
* **Agency facts** are parsed out of ``content-skill-graph/engine/agency-facts.md``,
  which is the client-verified allowlist. Parsing the file rather than hardcoding
  the figures keeps that file the single source of truth, so a number the client
  revokes there stops appearing on cards with no code change.
* **Act credentials** are extracted from the act's own loveproductions.com page
  and then checked back against that page's text, so a figure the model
  hallucinates cannot reach a card.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone

import anthropic
from anthropic.types import TextBlock

from . import config
from .cards import display_act
from .scrape import fetch_page_prose

log = logging.getLogger(__name__)

_AGENCY_FACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "content-skill-graph", "engine", "agency-facts.md",
)

# A figure worth a card is a number with something countable attached. Bare
# years ("1985") are handled separately, since the card wants "40 years", not
# the year itself.
_STAT_RE = re.compile(r"\b(\d[\d,]*\+?)\s+([a-z][a-z \-]{2,40}?)(?=[.,]|$)", re.I)
_FOUNDED_RE = re.compile(r"\bfounded in (\d{4})\b", re.I)

# An act page that dates the act ("Founded and named by Herb Reed in 1953") is
# stating a checkable fact, and "73 years performing" is a fair reading of it.
# But that is arithmetic, and the extraction guard below correctly refuses to
# take a model's arithmetic on trust: it once returned "7 decades performing"
# for a page whose only number was 1953. So the sum is done here in code, from
# a year that is literally on the page, instead of being asked for.
_ACT_SINCE_RE = re.compile(
    r"\b(?:founded|formed|established|performing|touring|started|began)\b[^.]{0,40}?\b"
    r"(?:in|since)\s+(19\d{2}|20[0-2]\d)\b",
    re.I,
)


def _verified_lines() -> list[str]:
    """The bullets under '## Verified' in agency-facts.md, nothing below it."""
    try:
        with open(_AGENCY_FACTS, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        log.warning("Could not read agency-facts.md: %s", exc)
        return []

    block = re.split(r"^##\s+", text, flags=re.M)
    for section in block:
        if section.lower().startswith("verified"):
            return [
                ln.strip("- ").strip()
                for ln in section.splitlines()
                if ln.strip().startswith("-")
            ]
    return []


def agency_stats(today_year: int) -> list[dict]:
    """Card-ready figures about the agency, parsed from the verified allowlist.

    Returns dicts of ``{value, label, context, source}``. The years-in-business
    figure is derived from the founding year rather than stored, so it stays
    right without anyone remembering to update it.
    """
    stats = []
    for line in _verified_lines():
        founded = _FOUNDED_RE.search(line)
        if founded:
            year = int(founded.group(1))
            if today_year > year:
                stats.append({
                    "value":   f"{today_year - year}",
                    "label":   "years booking live entertainment",
                    "context": f"Love Productions has been booking acts since {year}.",
                    "source":  "agency-facts.md",
                })
            continue

        match = _STAT_RE.search(line)
        if match:
            label = match.group(2).strip().lower()
            # When the bullet says nothing the number and label do not already
            # say ("Represents 180+ acts"), the context line is just the card
            # repeating itself, so leave it off.
            body = re.sub(r"[^a-z0-9]+", "", line.lower())
            echo = re.sub(r"[^a-z0-9]+", "", (match.group(1) + label).lower())
            stats.append({
                "value":   match.group(1),
                "label":   label,
                "context": "" if len(body) - len(echo) < 12 else line.rstrip("."),
                "source":  "agency-facts.md",
            })

    return stats


def rebooking_stat(topic: dict) -> dict | None:
    """Turn a re-booking topic into the strongest single number it contains.

    A venue booking the same act repeatedly is the most direct answer to the
    only question a talent buyer is actually asking, so this is the card that
    earns its slot. Two bookings is a coincidence and reads as weak proof, so
    the floor is three.
    """
    count = int(topic.get("_rebooking_count") or 0)
    venue = (topic.get("_venue") or "").strip()
    act = (topic.get("_act") or topic.get("artist") or "").strip()
    if count < 3 or not venue or not act:
        return None

    span = (topic.get("_span") or "").strip()
    context = f"{venue} has brought {act} back {count} times"
    context = f"{context}, {span}." if span else f"{context}."
    return {
        "value":   f"{count}x",
        "label":   f"booked by {venue}",
        "context": context,
        "source":  "airtable contracts",
    }


_CREDENTIAL_PROMPT = """Below is the copy from a tribute act's page on its booking
agency's website.

Find at most 2 hard, checkable NUMBERS in this copy that would matter to a talent
buyer deciding whether to book this act. Good examples: number of countries
toured, years the act has been performing, number of shows played, size of the
band, a chart position, a year of a notable first.

Rules:
- The number must appear in the copy below. Do not infer, estimate or calculate.
- It must be a COUNT of something, not a calendar year. "100 sold-out tours" and
  "25 years performing" are good. "1973" and "founded in 1953" are not: a bare
  year is a date, and on its own it tells a buyer nothing.
- Skip anything about the ORIGINAL artist's own career (record sales, their tours).
  We want facts about THIS tribute act.
- Skip vague quantities ("many", "countless", "hundreds of fans").
- label must be lowercase, under 40 characters, and read naturally after the
  number, e.g. "countries toured" or "years on the road".
- If the copy contains no such number, return an empty list. That is a fine
  answer and is better than stretching for a weak one.

Return JSON only: {"stats": [{"value": "70+", "label": "countries toured",
"context": "one sentence of supporting detail, quoted or closely paraphrased from the copy"}]}

COPY:
"""


def act_credential_stats(artist: dict, max_items: int = 1) -> list[dict]:
    """Extract act-level numbers from the act's own page, then verify them.

    The page is the agency's own published copy, so a figure taken from it is
    already client-approved. The verification step exists because the model,
    not the page, chooses which figure to surface: any value whose digits are
    not literally present in the prose is dropped.
    """
    name = (artist.get("name") or "").strip()
    url = (artist.get("artist_url") or "").strip()
    if not name or not url:
        return []
    if config.claude_call_count >= config.CLAUDE_CALL_LIMIT or not config.under_cost_cap(name):
        return []

    prose = fetch_page_prose(url, max_chars=6000)
    if not prose or len(prose) < 200:
        return []

    # Years-active is computed here rather than extracted, so it needs no model
    # call and cannot be got wrong. It is also the most common real number on
    # these pages, which is why the extraction path so often came back empty.
    since = _ACT_SINCE_RE.search(prose)
    if since:
        year = int(since.group(1))
        years = datetime.now(timezone.utc).year - year
        if years >= 10:
            return [{
                "value":   f"{years}",
                "label":   "years performing",
                "context": f"{display_act(name)} has been performing since {year}.",
                "source":  url,
            }]

    # No web_search tool here: the act page is the only permitted source, and
    # letting the model search would reintroduce exactly the unverifiable
    # figures the extraction guard below exists to stop.
    config.claude_throttle()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        raw = client.messages.with_raw_response.create(
            model=config.SEARCH_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": _CREDENTIAL_PROMPT + prose}],
        )
        resp = raw.parse()
        config.claude_call_done(dict(raw.headers))
        config.track_cost(resp, config.SEARCH_MODEL)
    except Exception as exc:  # noqa: BLE001
        log.error("Credential stat extraction failed for %s: %s", name, exc)
        return []

    text = "".join(b.text for b in resp.content if isinstance(b, TextBlock))
    text = re.sub(r"```(?:json)?\s*", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    out = []
    for item in (data.get("stats") or [])[:max_items]:
        value = str(item.get("value") or "").strip()
        # Not lowercased here: the prompt already asks for lowercase, and forcing
        # it turns "sold-out tours in the US" into "...in the us".
        label = str(item.get("label") or "").strip()
        if not value or not label:
            continue

        # The digits must actually appear in the page copy. This is the whole
        # guard: without it a plausible-sounding invented number reaches a card
        # that goes out under the agency's name.
        digits = re.sub(r"[^\d]", "", value)
        if not digits or digits not in re.sub(r"[^\d]", "", prose):
            log.info("Dropped unverified credential stat for %s: %s %s", name, value, label)
            continue

        # A bare year slips past the prompt often enough to be worth catching
        # here: "1973" is a date, and a card whose whole message is a date says
        # nothing to a buyer.
        if re.fullmatch(r"(19|20)\d{2}", digits):
            log.info("Dropped calendar-year credential stat for %s: %s %s", name, value, label)
            continue

        out.append({
            "value":   value,
            "label":   label,
            "context": str(item.get("context") or "").strip(),
            "source":  url,
        })

    return out

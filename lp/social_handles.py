"""Instagram handles for the acts we represent and the venues that book them.

Tagging an act's own account puts the post in front of that act's followers and
gives them something to reshare, which is the cheapest reach the engine has.
Tagging the venue does the same on the buyer's side.

**Instagram only, deliberately.** A plain-text "@name" linkifies and notifies on
Instagram. It does not on Facebook, where tagging a Page from the API needs the
"@[pageid:0:Name]" markup Buffer does not expose, and it does not on LinkedIn,
where a mention is an entity reference rather than text. On both of those an
"@handle" is literal characters the reader cannot tap, so we do not write one.

**Every handle here was confirmed on the act's or venue's own website, or on two
platforms agreeing (an Instagram bio matching a Facebook page of the same name
and description).** Do not add one from a search result alone: a wrong handle
tags a stranger in a post published under the agency's name, which is worse than
not tagging at all. Leave the entry out instead. Same rule as
``verify_quote_on_page()``: fail closed.

Keys are lowercased act names as they appear in Airtable. ``lookup_act_handle()``
tolerates the usual drift ("The Platters" against "Platters, The", and the
descriptor after a colon), the same way ``load_artist_mappings()`` does.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("lp.social_handles")

# Act name -> Instagram handle, no leading "@". Verified 2026-08-05.
ACT_INSTAGRAM: dict[str, str] = {
    "platters, the": "theplattersofficial",          # Herb Reed Enterprises, the act LP books
    "arrival from sweden: the music of abba": "arrival.from.sweden",
    "dolly show, the": "thedollyshowtribute",        # Kelly O'Brien
    "the rocket man show": "rusrocketman",           # Rus Anderson
    "calpulli mex dance co.": "calpullimexicandance",
    "back 2 mac: a tribute to fleetwood mac": "back2macband_",
    "priscilla presley": "priscillapresley",
    "the wankers": "thewankersnj",
    "tony danza: standards & stories": "tonydanza",
    "kiss the sky: a jimi hendrix tribute": "ktsjimihendrix",
    "reza": "rezaillusionist",
    "michael griffin escapes": "michaelgriffinescapes",
    "kyle martin's piano man": "pianomanlv",
    "legends of classic rock": "legendsofclassicrocktheband",
    "monkee men": "themonkeemen",
    "free fallin: the tom petty concert experience": "freefallinband",
    "vitaly: an evening of wonders!": "vitalyeveningofwonders",
    "love tko teddy pendergrass": "theteddypendergrassband",
    "a1a: the original jimmy buffett tribute": "jeffpikeanda1a",
    "bohemian queen": "bohemianqueenband",
}

# Four acts have no account we could confirm, and are left out on purpose rather
# than guessed at. Ask the client whether these exist before adding them:
#   Elvis: The Concert of Kings   (elvisconcertofkings.com links generic
#                                  facebook.com/instagram.com placeholders)
#   Eagle Wings & More / End of the Innocence
#   Always Celine                 (promoted through venue pages only)
#   Legends of Pop in Concert

# Venue name -> Instagram handle. Seeded with venues that actually recur in the
# re-booking data, since those are the ones that reach a post. Partial by
# design: there are 30-odd venues in the calendar and more every week, so treat
# this as a list to grow, not one to complete. A venue that is missing simply
# does not get tagged.
VENUE_INSTAGRAM: dict[str, str] = {
    "orange blossom opry": "orangeblossomopry",
    "the arcada theater": "arcadatheatre",
    "arcada theatre": "arcadatheatre",
    "celebrity theatre": "thecelebrityphx",
    "the carolina opry": "thecarolinaoprytheater",
    "carolina opry": "thecarolinaoprytheater",
    "the wilbur": "the_wilbur",
    "mohegan sun": "mohegansun",
    "genesee theatre": "genesee_theatre",
}

_HANDLE_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")


def _norm(name: str) -> str:
    """Lowercase, collapse whitespace. Line breaks show up in venue names."""
    return re.sub(r"\s+", " ", (name or "")).strip().lower()


def _variants(name: str):
    """The forms one name is filed under: as given, article moved, colon cut."""
    key = _norm(name)
    if not key:
        return
    yield key
    if key.startswith("the "):
        yield f"{key[4:]}, the"
        yield key[4:]
    elif key.endswith(", the"):
        yield f"the {key[:-5]}"
        yield key[:-5]
    head = key.split(":", 1)[0].strip()
    if head != key:
        yield head
        if head.startswith("the "):
            yield head[4:]


def _lookup(table: dict[str, str], name: str) -> str:
    for key in _variants(name):
        if key in table:
            return table[key]
    return ""


def lookup_act_handle(act: str) -> str:
    """Instagram handle for an act, without the "@". Empty when unknown."""
    return _lookup(ACT_INSTAGRAM, act)


def lookup_venue_handle(venue: str) -> str:
    """Instagram handle for a venue, without the "@". Empty when unknown."""
    return _lookup(VENUE_INSTAGRAM, venue)


def mention(handle: str) -> str:
    """"theplattersofficial" -> "@theplattersofficial". Empty if unusable."""
    handle = (handle or "").lstrip("@").strip()
    return f"@{handle}" if _HANDLE_RE.match(handle) else ""


def mentions_for_topic(topic: dict) -> list[str]:
    """Every "@handle" worth putting in this topic's Instagram caption.

    The act first, then the venue, because if only one survives a length or
    count trim it should be the act's own account.
    """
    out: list[str] = []
    act = topic.get("_act") or topic.get("artist") or ""
    for handle in (lookup_act_handle(act), lookup_venue_handle(_topic_venue(topic))):
        tag = mention(handle)
        if tag and tag.lower() not in {o.lower() for o in out}:
            out.append(tag)
    return out


def _topic_venue(topic: dict) -> str:
    """The venue a topic is about, wherever the building step happened to put it.

    Re-bookings carry ``_venue``; shows carry ``venue``. Neither is guaranteed.
    """
    return topic.get("_venue") or topic.get("venue") or ""

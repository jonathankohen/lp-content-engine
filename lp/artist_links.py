"""Static fallback map of tribute act → loveproductions.com page URL.

Every act page lives under the site's ``/title-item/`` directory. The canonical
source of these URLs is the Airtable artists table (``_artist_url_from_fields``
in ``lp/airtable.py``), but that scan can come back blank if a future act's row
is missing the link field, which would leave a news article with no page to
link the act's name to. This module prepopulates the links for the known roster
so ``fetch_airtable_artists()`` can fall back to them and never emit a blank at
runtime.

All URLs below were verified to return HTTP 200 on 2026-07-16. When a new act is
added to Airtable with its LPI web link populated, the Airtable value wins and no
edit here is needed; add an entry here only if a new act's Airtable link is (or
might be) missing. Keys are matched case-insensitively with surrounding
whitespace stripped (see ``lookup_artist_url``).
"""

import re

# Display name (as it appears in Airtable) → verified title-item page URL.
ARTIST_PAGE_URLS: dict[str, str] = {
    "Platters, The": "https://www.loveproductions.com/title-item/the-platters/",
    "Arrival From Sweden: The Music of ABBA": "https://www.loveproductions.com/title-item/arrival-from-sweden/",
    "Dolly Show, The": "https://www.loveproductions.com/title-item/the-dolly-show/",
    "The Rocket Man Show": "https://www.loveproductions.com/title-item/the-rocket-man-show/",
    "Elvis: The Concert of Kings": "https://www.loveproductions.com/title-item/elvis-the-concert-of-kings/",
    "Calpulli Mex Dance Co.": "https://www.loveproductions.com/title-item/capulli-mexican-dance-company/",
    "Back 2 Mac: A Tribute to Fleetwood Mac": "https://www.loveproductions.com/title-item/back-2-mac/",
    "Priscilla Presley": "https://www.loveproductions.com/title-item/priscilla-presley/",
    "The Wankers": "https://www.loveproductions.com/title-item/the-wankers/",
    "Tony Danza: Standards & Stories": "https://www.loveproductions.com/title-item/tony-danza/",
    "Eagle Wings & More - The Ultimate Tribute Salute w/ End of the Innocence": "https://www.loveproductions.com/title-item/end-of-the-innocence/",
    "Kiss The Sky: A Jimi Hendrix Tribute": "https://www.loveproductions.com/title-item/kiss-the-sky/",
    "Always Celine": "https://www.loveproductions.com/title-item/always-celine/",
    "Reza": "https://www.loveproductions.com/title-item/reza/",
    "Michael Griffin Escapes": "https://www.loveproductions.com/title-item/michael-griffin/",
    "Kyle Martin's Piano Man": "https://www.loveproductions.com/title-item/piano-man/",
    "Legends of Classic Rock": "https://www.loveproductions.com/title-item/legends-of-classic-rock/",
    "Monkee Men": "https://www.loveproductions.com/title-item/the-monkee-men/",
    "Free Fallin: The Tom Petty Concert Experience": "https://www.loveproductions.com/title-item/free-fallin/",
    "Vitaly: An Evening of Wonders!": "https://www.loveproductions.com/title-item/vitaly/",
    "Love TKO Teddy Pendergrass": "https://www.loveproductions.com/title-item/love-tko/",
    "A1A: The Original Jimmy Buffett Tribute": "https://www.loveproductions.com/title-item/a1a/",
    "Legends of Pop in Concert": "https://www.loveproductions.com/title-item/legends-of-pop-in-concert/",
    "Bohemian Queen": "https://www.loveproductions.com/title-item/bohemian-queen/",
}

# Normalized (stripped + casefolded) lookup index, built once at import.
_INDEX: dict[str, str] = {k.strip().casefold(): v for k, v in ARTIST_PAGE_URLS.items()}


def lookup_artist_url(name: str) -> str:
    """Return the static title-item URL for an act name, or "" if not known.

    Matching ignores surrounding whitespace and case so it tolerates the trailing
    spaces some Airtable names carry (e.g. "Platters, The ").
    """
    return _INDEX.get((name or "").strip().casefold(), "")


_INVERTED_RE = re.compile(r"^(.*?),\s*(the|a|an)$", re.I)

# Airtable name (normalized) → the name we are allowed to print.
#
# Client direction 2026-08-12: the act may not be called "Elvis" anything, so it
# is called by the rest of its own name. The Airtable name is left alone
# deliberately: dedup keys, the artists.md mapping and the tour-sheet tab match
# all key on it. Only the printed form changes.
ACT_DISPLAY_OVERRIDES: dict[str, str] = {
    "elvis: the concert of kings": "The Concert of Kings",
    "concert of kings": "The Concert of Kings",
}

# Names an act may not be called in published copy. Scoped to the ACT, not to
# the word: the client's ban (2026-08-12) is on calling the act "Elvis", and
# "Elvis Presley" as the original artist is still fair game (client confirmed).
# So the pattern matches "Elvis" only when it is NOT followed by "Presley",
# which is how the act gets referred to and how the old name is spelled.
_BANNED_ACT_RE = re.compile(r"\belvis\b(?!\s+presley\b)", re.I)


def banned_act_name_in(text: str) -> str:
    """Return the offending fragment if ``text`` calls an act by a banned name.

    Empty string means clean. Used as a fail-closed guard on generated copy and
    on act-page URLs: a slug is visible text on LinkedIn, so
    ``/title-item/elvis-the-concert-of-kings/`` breaks the rule as loudly as the
    copy would.
    """
    match = _BANNED_ACT_RE.search((text or "").replace("-", " "))
    return match.group(0) if match else ""


def display_act(act: str) -> str:
    """The act's name as it is allowed to appear in published copy.

    Un-inverts filing order ('Platters, The' -> 'The Platters'): Airtable files
    two acts alphabetically, which is right for a list and wrong everywhere else.
    It looks like a database error on a card, and it silently broke the
    tour-sheet lookup, whose tabs are named the natural way.

    Then applies ACT_DISPLAY_OVERRIDES, which is how a name the client cannot
    print gets replaced everywhere at once.
    """
    act = (act or "").strip()
    match = _INVERTED_RE.match(act)
    if match:
        act = f"{match.group(2).title()} {match.group(1)}"
    return ACT_DISPLAY_OVERRIDES.get(act.casefold(), act)


def short_act_name(act: str) -> str:
    """The name an act is called in a sentence: "Kiss The Sky", not
    "Kiss The Sky: A Jimi Hendrix Tribute".

    Roster names carry a descriptor after a colon or a dash for the catalogue.
    That is right in a headline and clumsy mid-sentence, which is where the
    act-page link line puts it ("More on Kiss The Sky here: ..."). Filing order
    is un-inverted first.
    """
    name = display_act(act)
    head = re.split(r"\s*[:\-]\s+", name, 1)[0].strip()
    # Only shorten if something usable is left; "A1A" alone would be too thin.
    return head if len(head) >= 4 else name

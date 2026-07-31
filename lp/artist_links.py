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

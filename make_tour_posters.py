"""
Generate a tour poster per act: the act's own key art above its upcoming dates.

The roster's acts do not publish real tour posters. What they post is
single-show flyers, usually venue-branded and stale within the week (the sweep
is written up in ``assets/act-key-art/TOUR-POSTERS-FINDINGS.md``), so the
multi-date poster has to be built from what we already hold: client-owned key
art in ``assets/act-key-art/`` and the dates in the tour dates sheet.

    python make_tour_posters.py                 # every act with enough dates
    python make_tour_posters.py --act "Reza"    # one act
    python make_tour_posters.py --min-dates 3   # loosen the cut-off
    python make_tour_posters.py --size square   # 1080x1080 instead of 4:5

Dates come from ``lp.sheets.load_tour_tabs()``, which reads every tab in a
single pass and caches it for the run.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from lp import config
from lp.cards import key_art_for, render_tour_poster
from lp.sheets import _normalize_region, load_tour_tabs

log = logging.getLogger("make_tour_posters")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "assets", "tour-posters", "generated")


def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in (text or "").lower())
    return "-".join(p for p in out.split("-") if p)


def match_tab(act: str, tabs: dict) -> str | None:
    """Tab title for an act, tolerating the wording drift between the two.

    The sheet says "The Dolly Show" where the roster says "Dolly Show, The", and
    "Concert of Kings" where the roster says "Elvis: The Concert of Kings".
    Slugs are compared both ways round, longest match winning so that a short
    name cannot claim a longer act's tab.
    """
    want = _slug(act)
    if not want:
        return None
    exact = [t for t in tabs if _slug(t) == want]
    if exact:
        return exact[0]
    hits = [t for t in tabs if _slug(t) in want or want in _slug(t)]
    if hits:
        return max(hits, key=lambda t: len(_slug(t)))

    # Substring matching misses the two acts the sheet writes the other way
    # round: "The Dolly Show" against "Dolly Show, The". Compare the word sets
    # with the leading article dropped, and require every sheet word to appear
    # so "Legends of Pop" cannot claim the "Legends of Classic Rock" tab.
    def words(text: str) -> set[str]:
        return {w for w in _slug(text).split("-") if w and w != "the"}

    want_words = words(act)
    if not want_words:
        return None
    subsets = [t for t in tabs if words(t) and words(t) <= want_words]
    return max(subsets, key=lambda t: len(words(t))) if subsets else None


def upcoming_from_rows(rows: list[list[str]]) -> list[dict]:
    """Future dates from raw tab rows, soonest first, deduped.

    Mirrors ``lp.sheets.upcoming_tour_dates``: columns are Date (MM/DD/YY),
    Venue, City, Region, Country, Ticket URL.
    """
    today = datetime.now().date()
    out = []
    for row in rows:
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

    seen, deduped = set(), []
    for item in sorted(out, key=lambda d: (d["date"], not d["ticket_url"])):
        key = (item["date"], item["city"].lower(), item["region"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act", help="render a single act instead of the roster")
    ap.add_argument("--min-dates", type=int, default=4,
                    help="skip acts with fewer upcoming dates (default 4)")
    ap.add_argument("--max-rows", type=int, default=12,
                    help="most date rows on one poster (default 12)")
    ap.add_argument("--size", default="poster", choices=("poster", "square", "linkedin"))
    ap.add_argument("--allow-residency", action="store_true",
                    help="include acts whose dates are all at one venue")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.load_env()

    from lp.artist_links import ARTIST_PAGE_URLS
    acts = [args.act] if args.act else sorted(ARTIST_PAGE_URLS)

    tabs = load_tour_tabs()
    if not tabs:
        log.error("No tour dates sheet. Check TOUR_DATES_SHEET_ID and "
                  "GOOGLE_APPLICATION_CREDENTIALS.")
        return 1
    log.info("Read %d tabs from the tour dates sheet\n", len(tabs))

    made, skipped = [], []
    for act in acts:
        tab = match_tab(act, tabs)
        if not tab:
            skipped.append((act, "no tab in the sheet"))
            continue
        dates = upcoming_from_rows(tabs[tab])
        if len(dates) < args.min_dates:
            skipped.append((act, f"{len(dates)} upcoming date(s), need {args.min_dates}"))
            continue

        # A residency renders as a column of identical rows ("AUG 01 REZA LIVE
        # THEATRE" sixteen times), which reads as a bug rather than a run of
        # shows. It is also not a tour, which is what this poster claims to be.
        #
        # Judge only the dates that will actually be drawn. Reza plays 92 dates
        # in two places, so a check across the whole list passes while every row
        # on the poster still says the same venue.
        visible = dates[: args.max_rows * 2]
        places = {(d.get("city") or d.get("venue") or "").strip().lower() for d in visible}
        if len(places) < 2 and not args.allow_residency:
            skipped.append((act, f"soonest {len(visible)} dates all at one venue (residency)"))
            continue

        art = key_art_for(act)
        if not art:
            skipped.append((act, "no key art on disk"))
            continue

        path = render_tour_poster(
            act, dates,
            out_dir=args.out_dir, size=args.size, max_rows=args.max_rows,
            art_path=art,
        )
        if path:
            made.append((act, len(dates), path))
        else:
            skipped.append((act, "renderer returned nothing"))

    print(f"\nRendered {len(made)} poster(s) into {args.out_dir}")
    for act, n, path in made:
        print(f"  {n:3d} dates  {act}\n            {os.path.basename(path)}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for act, why in skipped:
            print(f"  {act}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

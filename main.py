"""
LP Content Engine — Weekly social media draft generator.

Fetches artists from Airtable, searches for recent news via Claude web search,
deduplicates against a Google Sheet of already-used topics, generates
platform-native LinkedIn/Instagram/Facebook posts via Claude, and queues
them as Buffer drafts for human review.

Usage:
  python main.py                       # normal weekly run
  python main.py --dry-run             # preview only — no Sheets or Buffer writes
  python main.py --artist "Act Name"   # single artist
  python main.py --test-airtable       # print artist list and exit
  python main.py --test-calendar       # print upcoming shows and exit
  python main.py --test-buffer         # verify Buffer API and post test drafts
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from lp import config
from lp.airtable import fetch_airtable_artists, fetch_upcoming_shows, show_to_topic
from lp.ai import (
    filter_new_topics,
    generate_posts,
    score_and_rank_topics,
    search_artist_news,
)
from lp.buffer import (
    discover_buffer_profiles,
    get_occupied_slots,
    post_draft_to_buffer,
    purge_expired_show_drafts,
    test_buffer,
)
from lp.loaders import load_artist_mappings, load_skill_graph
from lp.sheets import mark_show_used, mark_topics_used, read_used_topics

_EASTERN = ZoneInfo("America/New_York")
_POST_HOUR = 10  # 10am ET


def get_week_slots() -> list[datetime]:
    """Return the next 7 days' 10am ET slots from now (always a full week)."""
    now = datetime.now(_EASTERN)
    slots = []
    day = now.date()
    while len(slots) < 7:
        slot = datetime(day.year, day.month, day.day, _POST_HOUR, 0, 0, tzinfo=_EASTERN)
        if slot > now:
            slots.append(slot)
        day += timedelta(days=1)
    return slots


def _select_with_diversity(ranked: list[dict], n_slots: int) -> list[dict]:
    """Pick up to n_slots topics; no act may appear more than twice consecutively."""
    selected: list[dict] = []
    used: set[int] = set()
    while len(selected) < n_slots:
        last_two = [t.get("_act", "") for t in selected[-2:]]
        chosen = None
        for i, topic in enumerate(ranked):
            if i in used:
                continue
            act = topic.get("_act", "")
            if len(last_two) == 2 and last_two[0] == act and last_two[1] == act:
                continue  # would be a 3rd consecutive — skip
            chosen = i
            break
        if chosen is None:
            # All remaining violate the rule — take best available rather than leave slot empty
            for i in range(len(ranked)):
                if i not in used:
                    chosen = i
                    break
        if chosen is None:
            break
        used.add(chosen)
        selected.append(ranked[chosen])
    return selected


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(dry_run: bool = False, single_artist: str = "") -> None:
    config.load_env()

    # Uncomment to delete expired show announcement drafts from Buffer at the start of each run:
    purge_expired_show_drafts(dry_run=dry_run)

    log.info("Loading content skill graph...")
    skill_graph = load_skill_graph()
    mappings = load_artist_mappings()
    log.info("Loaded %d artist mappings", len(mappings))

    if single_artist:
        artists = [{"name": single_artist, "priority": "manual"}]
        log.info("Single-artist mode: %s", single_artist)
    else:
        artists = fetch_airtable_artists()
        if not artists:
            log.error("No artists fetched from Airtable — aborting")
            return
        log.info("Fetched %d artists from Airtable", len(artists))

    used = read_used_topics()
    log.info("Loaded %d already-used topics from Sheets", len(used))

    buffer_profiles = discover_buffer_profiles()
    if not buffer_profiles and not dry_run:
        log.error("No Buffer profiles found — aborting")
        return

    all_new_topics: list[dict] = []

    # ── Show announcements from Airtable calendar ─────────────────────────────
    for show in fetch_upcoming_shows():
        topic = show_to_topic(show, mappings)
        if topic["url"] in used:
            log.info("Show already drafted: %s", topic["headline"])
            continue
        log.info("Generating show announcement: %s", topic["headline"])
        posts = generate_posts(topic, skill_graph)
        if not posts:
            continue
        for platform in ("linkedin", "instagram", "facebook"):
            text = posts.get(platform, "").replace(" — ", "—").replace(" – ", "–")
            if not text:
                log.warning("No %s post for '%s'", platform, topic["headline"])
                continue
            profile_id = buffer_profiles.get(platform, "")
            if not profile_id and not dry_run:
                log.warning("No Buffer profile for %s — skipping", platform)
                continue
            ok = post_draft_to_buffer(
                text,
                profile_id,
                platform=platform,
                dry_run=dry_run,
                image=config._IG_PLACEHOLDER if platform == "instagram" else None,
            )
            if ok:
                log.info("  %s show draft queued (%d chars)", platform, len(text))
            else:
                log.warning("  %s show draft FAILED — see error above", platform)
        used.add(topic["url"])
        all_new_topics.append(topic)
        mark_show_used(show, topic["url"], dry_run=dry_run)

    # ── Artist news pipeline ──────────────────────────────────────────────────
    all_slots = get_week_slots()
    occupied = (
        get_occupied_slots() if not dry_run else set()
    )
    week_slots = [s for s in all_slots if s.date().isoformat() not in occupied]
    log.info(
        "Week slots: %d available, %d already occupied in Buffer",
        len(week_slots),
        len(all_slots) - len(week_slots),
    )

    # Phase 1: collect all candidate topics across every artist
    all_candidates: list[dict] = []
    for artist in artists:
        name = artist["name"]
        original = mappings.get(name, "")
        log.info("--- Searching: %s [%s]", name, artist["priority"])

        found = search_artist_news(name, original)
        new_topics = filter_new_topics(found, used)

        if not new_topics:
            log.info("No new topics for %s", name)
            continue

        log.info("%d new topic(s) for %s", len(new_topics), name)
        for t in new_topics:
            t["_priority"] = artist["priority"]
            t["_act"] = name  # Airtable act name used for diversity grouping
            key = t.get("url", "").strip() or t.get("headline", "").strip()
            used.add(key)  # prevent cross-artist duplicates
        all_candidates.extend(new_topics)

    if not all_candidates:
        log.info("No new topics found this week")
    elif not week_slots:
        log.warning("No open slots this week — skipping artist news posts")
    else:
        # Phase 2: score, rank, drop below-threshold, pick top N with diversity
        ranked = score_and_rank_topics(all_candidates)
        selected = _select_with_diversity(ranked, len(week_slots))
        log.info(
            "Scheduling %d/%d topic(s) across %d slot(s)",
            len(selected),
            len(all_candidates),
            len(week_slots),
        )

        # Phase 3: generate posts and schedule to Buffer
        scheduled_topics: list[dict] = []
        for i, topic in enumerate(selected):
            headline = topic.get("headline", "")[:80]
            slot = week_slots[i]
            log.info(
                "Generating: %s (score=%.2f, slot=%s)",
                headline,
                topic.get("_score", 0),
                slot.strftime("%a %b %d"),
            )

            posts = generate_posts(topic, skill_graph)
            if not posts:
                continue

            is_original_artist = topic.get("hook_type") == "original_artist_news"
            platforms = ("instagram", "facebook") if is_original_artist else ("linkedin", "instagram", "facebook")
            if is_original_artist:
                log.info("  Original artist news — skipping LinkedIn for '%s'", headline)
            for platform in platforms:
                text = posts.get(platform, "").replace(" — ", "—").replace(" – ", "–")
                if not text:
                    log.warning("No %s post generated for '%s'", platform, headline)
                    continue
                profile_id = buffer_profiles.get(platform, "")
                if not profile_id and not dry_run:
                    log.warning("No Buffer profile for %s — skipping", platform)
                    continue
                ok = post_draft_to_buffer(
                    text,
                    profile_id,
                    platform=platform,
                    dry_run=dry_run,
                    scheduled_at=slot,
                    image=config._IG_PLACEHOLDER if platform == "instagram" else None,
                )
                if ok:
                    log.info(
                        "  %s draft scheduled %s (%d chars)",
                        platform,
                        slot.strftime("%a %b %d %I:%M%p %Z"),
                        len(text),
                    )
                else:
                    log.warning("  %s draft FAILED — see error above", platform)

            scheduled_topics.append(topic)

        all_new_topics.extend(scheduled_topics)
        mark_topics_used(scheduled_topics, dry_run=dry_run)

    log.info(
        "=== Run complete. Topics processed: %d | Est. cost: $%.4f ===",
        len(all_new_topics),
        config.estimated_cost_usd,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LP Content Engine — weekly social draft generator"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log outputs, skip Sheets and Buffer writes",
    )
    parser.add_argument(
        "--test-airtable",
        action="store_true",
        help="Print Airtable artist list and exit",
    )
    parser.add_argument(
        "--test-calendar",
        action="store_true",
        help="Print upcoming shows from the Airtable calendar and exit",
    )
    parser.add_argument(
        "--test-buffer",
        action="store_true",
        help="List Buffer channels and post a test draft to each",
    )
    parser.add_argument(
        "--artist",
        metavar="NAME",
        help="Run the full pipeline for a single artist (skips Airtable fetch)",
    )
    args = parser.parse_args()

    if args.test_airtable:
        config.load_env()
        artists = fetch_airtable_artists()
        if not artists:
            print(
                "No artists returned. Check AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_ARTIST_TABLE."
            )
        for a in artists:
            print(f"  [{a['priority']}] {a['name']}")
        sys.exit(0)

    if args.test_calendar:
        config.load_env()
        shows = fetch_upcoming_shows()
        if not shows:
            print(
                f"No upcoming shows in the next {config.SHOW_DAYS_AHEAD} days with FE status."
            )
        else:
            print(f"Upcoming shows (next {config.SHOW_DAYS_AHEAD} days, FE contracts):")
            for s in shows:
                print(f"  {s['show_date']}  {s['show_title']}  —  {s['venue_address']}")
        sys.exit(0)

    if args.test_buffer:
        config.load_env()
        test_buffer()
        sys.exit(0)

    main(dry_run=args.dry_run, single_artist=args.artist or "")

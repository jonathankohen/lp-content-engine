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
import re
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from lp import config
from lp.airtable import fetch_airtable_artists, fetch_upcoming_shows, show_to_topic
from lp.ai import (
    classify_show_announcements,
    default_categories,
    filter_new_topics,
    format_performance_context,
    generate_article,
    generate_posts,
    score_and_rank_topics,
    search_artist_news,
    search_historical_facts,
    search_trivia,
)
from lp.buffer import (
    discover_buffer_profiles,
    fetch_buffer_posts,
    fetch_top_performers,
    get_occupied_slots,
    post_draft_to_buffer,
    purge_expired_show_drafts,
    test_buffer,
)
from lp.airtable import fetch_venue_from_contracts
from lp.feeds import load_feed_items, search_artist_feeds
from lp.loaders import load_artist_mappings, load_skill_graph
from lp.scrape import fetch_og_image
from lp.sheets import lookup_ticket_url, mark_show_used, mark_topics_used, read_used_topics
from lp.wordpress import publish_news_posts

_EASTERN = ZoneInfo("America/New_York")
_POST_HOUR = 10  # 10am ET


def get_week_slots() -> list[datetime]:
    """Return 7 slots for Tue–Mon of the current content week, at 10am ET."""
    today = datetime.now(_EASTERN).date()
    days_until_tuesday = (1 - today.weekday()) % 7 or 7
    tuesday = today + timedelta(days=days_until_tuesday)
    return [
        datetime(tuesday.year, tuesday.month, tuesday.day, _POST_HOUR, 0, 0, tzinfo=_EASTERN) + timedelta(days=i)
        for i in range(7)
    ]


def _image_for(platform: str, og_image: str | None) -> str | None:
    """Per-platform image routing for a topic's scraped og:image.

    - instagram: scraped image, falling back to the LP logo placeholder
    - linkedin: scraped image only (None -> text-only post, never a link card)
    - facebook: None (relies on the URL in the body for a native link preview)
    """
    if platform == "instagram":
        return og_image or config._IG_PLACEHOLDER
    if platform == "linkedin":
        return og_image
    return None


def _news_button_url(topic: dict, artist_url: str) -> str:
    """Where a news post's 'Read more' button points — always somewhere.

    Source URL → ticket URL → the act's loveproductions.com page → the homepage.
    """
    return (
        (topic.get("url") or "").strip()
        or (topic.get("ticket_url") or "").strip()
        or (artist_url or "").strip()
        or config.LP_HOMEPAGE
    )


def _build_news_post(
    topic: dict,
    skill_graph: str,
    og_image: str | None,
    artist_url: str,
) -> dict | None:
    """Generate a website article for a topic and package it for the LP News plugin.

    Returns the payload dict (or None if article generation fails / is capped).
    The dedup key prefers the source URL, then the show sheet key, then the headline.

    Show announcements are deliberately excluded from the website news section
    (the client does not want them there), so any ``upcoming_show`` topic returns
    None before any Claude call.
    """
    if topic.get("hook_type") == "upcoming_show":
        log.info("Skipping news post for show announcement: %s", topic.get("headline", "")[:70])
        return None
    article = generate_article(
        topic, skill_graph, default_categories(topic.get("hook_type", ""))
    )
    if not article:
        return None
    key = (
        (topic.get("url") or "").strip()
        or (topic.get("sheet_key") or "").strip()
        or topic.get("headline", "").strip()
    )
    # Featured image: ONLY the source article's own image (og:image). We never
    # fall back to the artist's profile photo or the LP placeholder — if the
    # article has no image, the post is drafted imageless and handled manually.
    return {
        "key":          key,
        "title":        article["title"],
        "body":         article["body"],
        "categories":   article["categories"],
        "button_url":   _news_button_url(topic, artist_url),
        "button_label": "Read more",
        "image_url":    og_image or "",
    }


_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_CTA_LABELS = r"read more|full story(?: here)?|get tickets|tickets|more here|link"


def _first_url(text: str) -> str:
    """First http(s) URL in text, with trailing punctuation trimmed."""
    m = _URL_RE.search(text or "")
    return m.group(0).rstrip(".,);]") if m else ""


def _strip_source_urls(text: str) -> str:
    """Remove URLs (and their CTA-label prefixes like 'Read more:') from text.

    Used when reconstructing an article body from a social post so the raw link
    doesn't leak into the website body — the 'Read more' button carries it.
    """
    t = text or ""
    # Whole "Read more: <url>" style lines first, then any stray URLs, then any
    # now-dangling CTA labels left behind.
    t = re.sub(rf"(?im)^\s*(?:{_CTA_LABELS})\s*[:\-]?\s*https?://\S+\s*$", "", t)
    t = _URL_RE.sub("", t)
    t = re.sub(rf"(?im)^\s*(?:{_CTA_LABELS})\s*[:\-]?\s*$", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _artist_name_variants(name: str) -> list[str]:
    """Lowercased name variants to match against post text (full name, the part
    before a colon/dash/"w/", and a ", The" → "The …" flip). Variants shorter
    than 5 chars are dropped to avoid spurious substring hits."""
    n = (name or "").strip()
    variants = {n}
    if n.lower().endswith(", the"):
        base = re.sub(r",\s*the\s*$", "", n, flags=re.IGNORECASE)
        variants.update({base, f"The {base}"})
    for sep in (":", " - ", " w/", " with "):
        if sep in n:
            variants.add(n.split(sep)[0])
    out: list[str] = []
    for v in variants:
        v = v.strip().lower()
        if len(v) >= 5 and v not in out:
            out.append(v)
    return out


def _match_artist_url(text: str, artists: list[dict], mappings: dict) -> str:
    """Best-effort: find the act a post is about and return its loveproductions.com
    page URL. Tries the tribute act name (and variants) first, then the original
    artist name. Returns "" when nothing matches."""
    t = (text or "").lower()
    for a in artists:
        if any(v in t for v in _artist_name_variants(a["name"])):
            return a.get("artist_url", "")
    for a in artists:
        orig = (mappings.get(a["name"], "") or "").strip().lower()
        if len(orig) >= 5 and orig in t:
            return a.get("artist_url", "")
    return ""


def _buffer_posts_to_news(
    posts: list[dict],
    skill_graph: str,
    artists: list[dict],
    mappings: dict,
    dry_run: bool,
) -> None:
    """Convert a list of Buffer Facebook posts into website news drafts and publish.

    Each post's text is rebuilt into a long-form article via generate_article().
    The source URL (extracted from the body) becomes the 'Read more' button and
    seeds the featured image (og:image). When there's no source URL, the post is
    matched to an act so the button points to that act's page; the featured image
    is always the article's own og:image and never falls back to a profile photo
    or placeholder — imageless articles are drafted for manual handling. Deduped
    by source URL (falling back to the Buffer post id) so re-runs and the weekly
    pipeline never double-post the same story.
    """
    # Show announcements don't belong in the website news section. Backfilled
    # topics carry no hook_type, so classify the posts (one Haiku call for the
    # batch) and skip any flagged as a live-event/show announcement.
    show_flags = classify_show_announcements([p.get("text", "") for p in posts])

    news_posts: list[dict] = []
    for p, is_show in zip(posts, show_flags):
        text = p.get("text", "")
        if not text.strip():
            continue
        if is_show:
            log.info("Skipping show announcement (not website news): %.70s", text.replace("\n", " "))
            continue
        url = _first_url(text)
        artist_url = _match_artist_url(text, artists, mappings)
        topic = {
            "artist":          "",
            "original_artist": "",
            "headline":        "",
            "summary":         _strip_source_urls(text),
            "url":             url,
            "hook_type":       "",
        }
        og_image = fetch_og_image(url) if url else None
        news_post = _build_news_post(topic, skill_graph, og_image, artist_url)
        if not news_post:
            continue
        if not news_post["key"]:
            news_post["key"] = f"buffer_{p['id']}"
        news_posts.append(news_post)
        log.info("  prepared: %s [%s]", news_post["title"][:70], ", ".join(news_post["categories"]))

    if news_posts:
        publish_news_posts(news_posts, dry_run=dry_run)
    log.info(
        "=== Backfill complete. %d post(s) prepared | Est. cost: $%.4f ===",
        len(news_posts),
        config.estimated_cost_usd,
    )


def backfill_news_from_buffer(dry_run: bool = False) -> None:
    """Convert the current Buffer Facebook drafts into website news posts.

    Facebook is the universal channel (every topic routes to it) and its body
    carries the source URL, so one news post per FB draft = one per topic (no
    cross-platform duplicates).
    """
    config.load_env()
    skill_graph = load_skill_graph()
    artists = fetch_airtable_artists()
    mappings = load_artist_mappings()
    drafts = fetch_buffer_posts(services={"facebook"}, statuses=("draft",))
    if not drafts:
        log.info("No Facebook drafts found in Buffer to convert.")
        return
    log.info("Found %d Facebook draft(s) to convert to news posts", len(drafts))
    _buffer_posts_to_news(drafts, skill_graph, artists, mappings, dry_run)


def backfill_news_from_queue(dry_run: bool = False) -> None:
    """Convert the current Buffer Facebook *queue* into website news posts.

    Targets posts with status 'scheduled' — i.e. approved and waiting in the
    queue to go out, distinct from unreviewed 'draft's and already-'sent' posts.
    Facebook is the universal channel (every topic routes to it) and its body
    carries the source URL, so one news post per FB queued post = one per topic
    (no cross-platform duplicates).
    """
    config.load_env()
    skill_graph = load_skill_graph()
    artists = fetch_airtable_artists()
    mappings = load_artist_mappings()
    queued = fetch_buffer_posts(services={"facebook"}, statuses=("scheduled",))
    if not queued:
        log.info("No Facebook posts in the Buffer queue to convert.")
        return
    log.info("Found %d queued Facebook post(s) to convert to news posts", len(queued))
    _buffer_posts_to_news(queued, skill_graph, artists, mappings, dry_run)


def backfill_news_from_week(days: int = 7, dry_run: bool = False) -> None:
    """Convert the past ``days`` of *published* Buffer Facebook posts into news posts.

    Pulls Facebook posts with status 'sent' whose sentAt falls within the window.
    Facebook is the universal channel so this is one news post per published topic.
    """
    config.load_env()
    skill_graph = load_skill_graph()
    artists = fetch_airtable_artists()
    mappings = load_artist_mappings()
    since = datetime.now(tz=_EASTERN) - timedelta(days=days)
    sent = fetch_buffer_posts(services={"facebook"}, statuses=("sent",), since=since)
    if not sent:
        log.info("No Facebook posts sent in the past %d day(s).", days)
        return
    log.info("Found %d Facebook post(s) sent in the past %d day(s) to convert", len(sent), days)
    _buffer_posts_to_news(sent, skill_graph, artists, mappings, dry_run)


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


def _fill_with_facts(
    *,
    label: str,
    search_fn,
    remaining_slots: list[datetime],
    artists: list[dict],
    mappings: dict,
    scheduled_topics: list[dict],
    used: set,
    skill_graph: str,
    perf_context: str,
    buffer_profiles: dict,
    dry_run: bool,
    news_posts: list[dict],
) -> None:
    """Fill remaining week slots with Instagram + Facebook-only fact posts (trivia / historical).

    Only acts with an original-artist mapping that haven't already been scheduled this run are
    eligible. ``search_fn`` is a callable ``(act_name, original, slot) -> list[dict]``.
    Mutates ``scheduled_topics`` and ``used`` in place.
    """
    if not remaining_slots:
        return
    scheduled_acts = {t.get("_act", "") for t in scheduled_topics}
    candidates = [
        a for a in artists
        if mappings.get(a["name"]) and a["name"] not in scheduled_acts
    ]
    for slot, artist in zip(remaining_slots, candidates):
        original = mappings[artist["name"]]
        log.info("--- %s: %s [%s]", label, artist["name"], artist["priority"])
        found = search_fn(artist["name"], original, slot)
        new_items = filter_new_topics(found, used)
        if not new_items:
            log.info("No %s found for %s", label.lower(), artist["name"])
            continue
        item = new_items[0]
        item["_priority"] = artist["priority"]
        item["_act"] = artist["name"]
        key = item.get("url", "").strip() or item.get("headline", "").strip()
        used.add(key)
        log.info(
            "Generating %s: %s (slot=%s)",
            label.lower(),
            item.get("headline", "")[:80],
            slot.strftime("%a %b %d"),
        )
        posts = generate_posts(item, skill_graph, perf_context)
        if not posts:
            continue
        og_image = fetch_og_image(item.get("url", ""))
        for platform in ("instagram", "facebook"):
            text = posts.get(platform, "").replace(" — ", "—").replace(" – ", "–")
            if not text:
                log.warning("No %s post for %s '%s'", platform, label.lower(), item.get("headline", "")[:60])
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
                image=_image_for(platform, og_image),
            )
            if ok:
                log.info(
                    "  %s %s scheduled %s (%d chars)",
                    platform,
                    label.lower(),
                    slot.strftime("%a %b %d %I:%M%p %Z"),
                    len(text),
                )
            else:
                log.warning("  %s draft FAILED — see error above", platform)
        news_post = _build_news_post(item, skill_graph, og_image, artist.get("artist_url", ""))
        if news_post:
            news_posts.append(news_post)
        scheduled_topics.append(item)


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(dry_run: bool = False, single_artist: str = "", days: int | None = None) -> None:
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
    news_posts: list[dict] = []  # website draft posts, sent to LP News at the end
    act_url = {a["name"]: a.get("artist_url", "") for a in artists}

    top_performers = fetch_top_performers(n=3)
    perf_context = format_performance_context(top_performers)
    if top_performers:
        log.info("Loaded %d top-performing post(s) as style context", len(top_performers))

    # ── Compute week slots before show announcements so shows claim the earliest ones ──
    all_slots = get_week_slots()
    if days is not None:
        all_slots = all_slots[:days]
        log.info("Limiting run to the first %d day slot(s)", days)
    occupied = (
        get_occupied_slots() if not dry_run else set()
    )
    week_slots = [s for s in all_slots if s.date().isoformat() not in occupied]
    log.info(
        "Week slots: %d available, %d already occupied in Buffer",
        len(week_slots),
        len(all_slots) - len(week_slots),
    )
    show_slots_used = 0

    # ── Show announcements from Airtable calendar ─────────────────────────────
    for show in fetch_upcoming_shows():
        topic = show_to_topic(show, mappings)
        if topic["sheet_key"] in used:
            log.info("Show already drafted: %s", topic["headline"])
            continue
        ticket_url, venue_name = lookup_ticket_url(show["show_title"], show["show_date"])
        if not venue_name:
            venue_name = fetch_venue_from_contracts(show["lpc_number"])
        if ticket_url:
            topic["ticket_url"] = ticket_url
            log.info("  Ticket URL found: %s", ticket_url)
        if venue_name:
            addr = show["venue_address"]
            topic["headline"] = topic["headline"].replace(addr, venue_name)
            topic["summary"] = topic["summary"].replace(addr, venue_name)
            log.info("  Venue name: %s", venue_name)
        log.info("Generating show announcement: %s", topic["headline"])
        posts = generate_posts(topic, skill_graph, perf_context)
        if not posts:
            continue
        slot = week_slots[show_slots_used] if show_slots_used < len(week_slots) else None
        effective_slot = slot if slot and slot > datetime.now(_EASTERN) else None
        og_image = fetch_og_image(topic.get("url", ""))
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
                image=_image_for(platform, og_image),
                scheduled_at=effective_slot,
            )
            if ok:
                if effective_slot:
                    log.info("  %s show draft scheduled %s (%d chars)", platform, effective_slot.strftime("%a %b %d %I:%M%p %Z"), len(text))
                else:
                    log.info("  %s show draft queued (%d chars)", platform, len(text))
            else:
                log.warning("  %s show draft FAILED — see error above", platform)
        news_post = _build_news_post(topic, skill_graph, og_image, act_url.get(topic["artist"], ""))
        if news_post:
            news_posts.append(news_post)
        used.add(topic["sheet_key"])
        all_new_topics.append(topic)
        mark_show_used(show, topic["sheet_key"], dry_run=dry_run)
        show_slots_used += 1

    # ── Artist news pipeline ──────────────────────────────────────────────────
    week_slots = week_slots[show_slots_used:]

    # Phase 1: collect all candidate topics across every artist
    feed_items = load_feed_items()  # whitelisted music-news RSS, fetched once
    all_candidates: list[dict] = []
    for artist in artists:
        name = artist["name"]
        original = mappings.get(name, "")
        log.info("--- Searching: %s [%s]", name, artist["priority"])

        found = search_artist_news(name, original)
        found += search_artist_feeds(name, original, feed_items)
        # Web search and a feed can surface the same URL — dedup within this
        # artist's batch before the run-wide filter (which dedups vs. `used`).
        seen_keys: set[str] = set()
        deduped = []
        for item in found:
            key = item.get("url", "").strip() or item.get("headline", "").strip()
            if key and key not in seen_keys:
                seen_keys.add(key)
                deduped.append(item)
        new_topics = filter_new_topics(deduped, used)

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

            posts = generate_posts(topic, skill_graph, perf_context)
            if not posts:
                continue

            hook = topic.get("hook_type", "")
            is_ig_fb_only = hook in ("original_artist_news", "historical_fact", "trivia")
            platforms = ("instagram", "facebook") if is_ig_fb_only else ("linkedin", "instagram", "facebook")
            if is_ig_fb_only:
                log.info("  %s — skipping LinkedIn for '%s'", hook, headline)
            effective_slot = slot if slot > datetime.now(_EASTERN) else None
            og_image = fetch_og_image(topic.get("url", ""))
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
                    scheduled_at=effective_slot,
                    image=_image_for(platform, og_image),
                )
                if ok:
                    log.info(
                        "  %s draft %s (%d chars)",
                        platform,
                        effective_slot.strftime("scheduled %a %b %d %I:%M%p %Z") if effective_slot else "queued",
                        len(text),
                    )
                else:
                    log.warning("  %s draft FAILED — see error above", platform)

            news_post = _build_news_post(topic, skill_graph, og_image, act_url.get(topic.get("_act", ""), ""))
            if news_post:
                news_posts.append(news_post)
            scheduled_topics.append(topic)

        # Phase 4: fill remaining slots with original-artist trivia (Instagram + Facebook only)
        if not single_artist:
            _fill_with_facts(
                label="Trivia",
                search_fn=lambda name, original, slot: search_trivia(name, original),
                remaining_slots=week_slots[len(scheduled_topics):],
                artists=artists,
                mappings=mappings,
                scheduled_topics=scheduled_topics,
                used=used,
                skill_graph=skill_graph,
                perf_context=perf_context,
                buffer_profiles=buffer_profiles,
                dry_run=dry_run,
                news_posts=news_posts,
            )

            # Phase 5: fill any still-remaining slots with pre-1990 historical facts (IG + FB only)
            _fill_with_facts(
                label="Historical fact",
                search_fn=search_historical_facts,
                remaining_slots=week_slots[len(scheduled_topics):],
                artists=artists,
                mappings=mappings,
                scheduled_topics=scheduled_topics,
                used=used,
                skill_graph=skill_graph,
                perf_context=perf_context,
                buffer_profiles=buffer_profiles,
                dry_run=dry_run,
                news_posts=news_posts,
            )

        all_new_topics.extend(scheduled_topics)
        mark_topics_used(scheduled_topics, dry_run=dry_run)

    # ── Website news posts (loveproductions.com via the LP News plugin) ────────
    if news_posts:
        log.info("Publishing %d news post(s) to loveproductions.com...", len(news_posts))
        publish_news_posts(news_posts, dry_run=dry_run)

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
        "--days",
        type=int,
        default=None,
        help="Cap the run to the first N day slots (default: full 7-day week)",
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
        "--test-analytics",
        action="store_true",
        help="Print top-performing Buffer posts with engagement scores and exit",
    )
    parser.add_argument(
        "--artist",
        metavar="NAME",
        help="Run the full pipeline for a single artist (skips Airtable fetch)",
    )
    parser.add_argument(
        "--news-from-buffer",
        action="store_true",
        help="Convert the current Buffer Facebook drafts into website news posts and exit",
    )
    parser.add_argument(
        "--news-from-queue",
        action="store_true",
        help="Convert the current Buffer Facebook queue (scheduled posts) into website news posts and exit",
    )
    parser.add_argument(
        "--news-from-week",
        nargs="?",
        type=int,
        const=7,
        metavar="DAYS",
        help="Convert the past week's published Buffer Facebook posts into website news posts and exit (optional DAYS, default 7)",
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

    if args.test_analytics:
        config.load_env()
        posts = fetch_top_performers(n=10)
        if not posts:
            print("No posts with engagement data found (analytics may not be available yet).")
        else:
            print(f"Top {len(posts)} post(s) by engagement:\n")
            for p in posts:
                platform = p.get("platform") or (p.get("serviceType") or "unknown").capitalize()
                score = p.get("engagement_score", 0)
                text = (p.get("text") or "")[:120]
                print(f"  [{platform}] score={score}  id={p.get('id', '')}")
                print(f"  {text!r}\n")
        sys.exit(0)

    if args.news_from_buffer:
        backfill_news_from_buffer(dry_run=args.dry_run)
        sys.exit(0)

    if args.news_from_queue:
        backfill_news_from_queue(dry_run=args.dry_run)
        sys.exit(0)

    if args.news_from_week is not None:
        backfill_news_from_week(days=args.news_from_week, dry_run=args.dry_run)
        sys.exit(0)

    main(dry_run=args.dry_run, single_artist=args.artist or "", days=args.days)

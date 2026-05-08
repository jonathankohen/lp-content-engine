# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# LP Content Engine — Claude Context

## For Claude: keep this file current

After any change that affects how the system works — new files, changed conventions, new scripts, toggled features, renamed artists, architectural decisions — update this file before ending the session. Future Claude has no memory of this conversation; this file is the handoff. If something was decided but not yet activated (e.g. a commented-out feature), note it here so it doesn't get re-implemented or re-debated.

## What this is

Automated weekly social media pipeline for **Love Productions** (loveproductions.com), a New York booking agency (est. 1985, 180+ acts, 70+ countries). The engine turns news about tribute acts and their original artists into platform-native LinkedIn/Instagram/Facebook draft posts, queued in Buffer for human review before publishing.

## Pipeline (end to end)

```
Airtable calendar (appXLETHThc0p5MOz) → upcoming FE shows (next 7 days)
  → Tour dates sheet (TOUR_DATES_SHEET_ID) → ticket URL lookup
  → Google Sheets dedup → Claude Sonnet show announcement posts → Buffer drafts

Airtable artists (appMMwX47V1g2Sv5u, priority-ordered)
  → Claude Haiku web search (news per artist)
  → Google Sheets dedup → Claude Sonnet content generation → Buffer drafts
  [Phase 4] → remaining slots → Claude Haiku historical fact search → Claude Sonnet posts → Buffer drafts

Buffer analytics → top performers → injected as style examples into every generate_posts() call
```

Show announcements run first, then artist news (Phases 1–3), then historical facts (Phase 4) to fill any remaining slots. All phases feed the same dedup store and Buffer queue.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in keys
```

## Commands

```bash
python main.py                              # full weekly run
python main.py --dry-run                    # preview only — no Sheets or Buffer writes, full post text printed
python main.py --artist "Concert of Kings"  # single artist
python main.py --test-calendar              # print upcoming shows from Airtable calendar and exit
python main.py --test-analytics             # print top-performing Buffer posts with engagement scores and exit

python clean_up.py                          # preview dash fixes + expired purge
python clean_up.py --apply                  # apply both
python clean_up.py --fix-dashes --apply     # dashes only
python clean_up.py --purge-expired --apply  # expired show drafts only
```

## Key files

| File                                         | Purpose                                                                                                                 |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `main.py`                                    | Thin orchestration layer + CLI entry point                                                                              |
| `clean_up.py`                                | Buffer draft maintenance — fixes em/en dash spacing and purges expired show announcement drafts                         |
| `lp/config.py`                               | All env vars, constants, cost tracking state + helpers, `load_env()`                                                    |
| `lp/loaders.py`                              | `load_skill_graph()`, `load_artist_mappings()`                                                                          |
| `lp/airtable.py`                             | `fetch_airtable_artists()`, `fetch_upcoming_shows()`, `show_to_topic()`                                                 |
| `lp/sheets.py`                               | Google Sheets read/write — `read_used_topics()`, `mark_topics_used()`, `mark_show_used()`, `lookup_ticket_url()`        |
| `lp/ai.py`                                   | Claude helpers — `search_artist_news()`, `search_historical_facts()`, `score_and_rank_topics()`, `generate_posts()`, `format_performance_context()` |
| `lp/buffer.py`                               | Buffer GraphQL — `discover_buffer_profiles()`, `post_draft_to_buffer()`, `fetch_top_performers()`, `purge_expired_show_drafts()`, `test_buffer()` |
| `content-skill-graph/`                       | Knowledge base loaded as Claude's system prompt at generation time                                                      |
| `content-skill-graph/index.md`               | Master index + execution instructions for content generation                                                            |
| `content-skill-graph/engine/artists.md`      | Tribute act → original artist mappings (markdown table)                                                                 |
| `content-skill-graph/engine/scoring.md`      | Scoring criteria for ranking candidate topics before generation                                                         |
| `content-skill-graph/voice/brand-voice.md`   | Core personality, vocabulary, banned words                                                                              |
| `content-skill-graph/voice/platform-tone.md` | Per-platform tone adaptations                                                                                           |
| `content-skill-graph/voice/humanizer.md`     | AI writing patterns to strip from every post                                                                            |

## Important decisions and conventions

**Skill graph as system prompt.** All 13 markdown files in `content-skill-graph/` are concatenated and passed as the system prompt to every `generate_posts()` call. Editing any `.md` file there immediately affects generation — no code change needed.

**Two models.** Haiku (`SEARCH_MODEL`) for news search and anything cheap/mechanical. Sonnet (`CONTENT_MODEL`) for content generation. Never swap them without considering cost impact.

**Cost cap.** Default $5/run (`COST_CAP_USD`). Each full topic costs ~$0.06 (1 search + 1 generate). Cap is checked before every Claude call via `config.under_cost_cap()`. Mutable cost state lives in `lp/config.py` (`estimated_cost_usd`, `claude_call_count`) and is mutated directly from `lp/ai.py`.

**Buffer rate limits.** Two distinct limits:
- *Per-minute:* 60 requests/minute. Both `main.py` and `clean_up.py` read `retryAfter` from 429 responses and retry. If no `retryAfter`, defaults to 61s wait.
- *Daily:* Hard 24h cap per API client (`"window":"24h"` in the 429 body). When hit, `clean_up.py` exits immediately with a clear message. Nothing to do but wait until the next day. Running `main.py` followed immediately by `clean_up.py` is enough to exhaust the daily limit, so space them out.
- Both scripts use the GraphQL API at `https://api.buffer.com`. The legacy REST API (`api.bufferapp.com`) does **not** accept OIDC tokens — don't use it.

**Buffer GraphQL notes.** Fetching posts requires `organizationId` (not `channelId`) in `PostsInput`. Updating posts uses `editPost` mutation (not `updatePost`). Facebook edits require `metadata.facebook.type`. Instagram edits require `metadata.instagram.type` + at least one image (use the LP logo placeholder). **Note:** Buffer removed `status` and `after` (pagination cursor) from `PostsInput` in a 2026 API update. `get_occupied_slots()` now fetches a single page only (no pagination) — sufficient for checking this week's 7 slots. `fetch_top_performers()` uses a `statistics` field that may or may not be available; it degrades gracefully to `[]` if not.

**Em dash convention.** Posts are written with `—` (no spaces). The `replace(" — ", "—")` normalization in `main.py` catches anything Claude generates with spaces. `clean_up.py --fix-dashes` retroactively cleans existing Buffer drafts.

**Show announcements (calendar pipeline).** `fetch_upcoming_shows()` pulls from Airtable base `appXLETHThc0p5MOz` (env: `AIRTABLE_CALENDAR_BASE_ID`), table `tblK2LMog1WUEv3j0`. Filters for `LPC Contract Status = "(FE) Fully Executed"` and show dates within the next 7 days (`SHOW_DAYS_AHEAD`). Dedup key is `lpc_{LPC #}` stored in the url column of Sheets. Test with `python main.py --test-calendar`.

**Artist names are exact.** The `artists.md` table maps tribute act names exactly as they appear in Airtable. Claude searches for the exact name — do not paraphrase. "Concert of Kings" (not "Elvis: The Concert of Kings"). Priscilla Presley is a separate act from Concert of Kings. **Note:** Airtable still has the old name "Elvis: The Concert of Kings" — needs to be updated there too for the mapping to work correctly.

**Expired draft cleanup.** `purge_expired_show_drafts()` lives in `lp/buffer.py` but is **commented out** at the top of `main()` in `main.py`. It detects show announcement drafts with past dates and deletes them. Uncomment when ready — always test with `--dry-run` first. The same logic is active (not commented out) in `clean_up.py --purge-expired`.

**Dry run output.** `main.py --dry-run` prints full post text (not truncated) with scheduled slot per post. Safe to run to preview output without writing to Sheets or Buffer.

**Scoring and scheduling (artist news pipeline).** `main.py` runs in four phases: (1) search all artists and collect every candidate topic, (2) score all candidates in one Haiku call (`score_and_rank_topics()` in `lp/ai.py`) using five criteria (relevance, freshness, velocity, virality, uniqueness) plus an exclusivity +1 bonus for Top of Roster / Exclusive acts and two hardcoded exceptions (Tony Danza, The Rocket Man Show), (3) generate posts and schedule to Buffer for top-scoring topics — one per remaining day at 10am ET, max 7, (4) fill any remaining week slots with historical facts (`search_historical_facts()`) for artists with an original artist mapping that had no news. Topics scoring below 0.40 are dropped. `get_week_slots()` always returns exactly 7 slots — the next 7 days from now at 10am ET. Show announcements bypass scoring and are always drafted without a scheduled time. `post_draft_to_buffer()` accepts an optional `scheduled_at: datetime` — when set, uses `schedulingType: "scheduled"` and a Unix timestamp in seconds; otherwise falls back to queue mode.

**Platform routing by hook type.** Topics with `hook_type` of `"original_artist_news"` or `"historical_fact"` are posted to Instagram and Facebook only — never LinkedIn. Tribute act news (`tribute_news`, `upcoming_show`) goes to all three platforms. `hook_type` is set by Claude during search (`search_artist_news()`) or hardcoded to `"historical_fact"` by `search_historical_facts()`.

**Historical facts (Phase 4).** After the artist news pipeline fills its slots, any remaining week slots are filled by `search_historical_facts()` — pre-1990 archival facts about original artists from sources like archive.org, old Rolling Stone, Billboard, etc. These are Instagram + Facebook only, lowest priority, generated only for artists with an original artist mapping, and excluded in `--artist` single-artist mode. Deduped via the same Sheets mechanism as news topics.

**Ticket links in show announcements.** `lookup_ticket_url()` in `lp/sheets.py` looks up a ticket URL in the tour dates Google Sheet (`TOUR_DATES_SHEET_ID`) produced by the `love-automations` repo. The sheet has per-artist tabs named by display name (e.g., "Arrival From Sweden" for "Arrival From Sweden: The Music of ABBA") with columns: Date (MM/DD/YY), Venue, City, Region, Country, Ticket URL, Source. Matching is by tab name (substring of show_title, case-insensitive) + date. If no match, the post is generated without a ticket link.

**Self-learning via performance context.** `fetch_top_performers()` in `lp/buffer.py` queries Buffer for recent posts with engagement data (reactions + comments + reposts + clicks). `format_performance_context()` in `lp/ai.py` formats the top 3 as style examples that are injected into every `generate_posts()` call. Returns `[]` / empty string gracefully if Buffer analytics aren't available. Use `--test-analytics` to inspect what's being loaded.

## Environment variables (`.env`)

```
ANTHROPIC_API_KEY
BUFFER_API_KEY
AIRTABLE_API_KEY
AIRTABLE_BASE_ID        # default: appMMwX47V1g2Sv5u
AIRTABLE_ARTIST_TABLE   # default: tbloEhiPP4kyTTVDb
FOUND_NEWS_STORIES_SHEETS_ID
TOUR_DATES_SHEET_ID     # optional: tour dates sheet from love-automations repo (for ticket links)
COST_CAP_USD            # default: 5.00
GOOGLE_APPLICATION_CREDENTIALS_JSON  # service account JSON string (CI/Actions)
```

## Adding or editing artists

Edit `content-skill-graph/engine/artists.md`. The table maps `Tribute Act | Original Artist`. Leave Original Artist blank if not applicable. Changes take effect on the next run — no code change needed.

## Extending the skill graph

Drop a new `.md` file anywhere under `content-skill-graph/`. It gets picked up automatically by `load_skill_graph()` (alphabetical sort). Use the existing files as style reference — keep them instructional, not descriptive.

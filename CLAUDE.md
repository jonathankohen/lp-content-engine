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
  → Google Sheets dedup → Claude Sonnet show announcement posts → Buffer drafts

Airtable artists (appMMwX47V1g2Sv5u, priority-ordered)
  → Claude Haiku web search (news per artist)
  → Google Sheets dedup → Claude Sonnet content generation → Buffer drafts
```

Show announcements run first, then artist news. Both feed the same dedup store and Buffer queue.

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
| `lp/sheets.py`                               | Google Sheets read/write — `read_used_topics()`, `mark_topics_used()`, `mark_show_used()`                               |
| `lp/ai.py`                                   | Claude throttle helpers, `search_artist_news()`, `score_and_rank_topics()`, `generate_posts()`, `filter_new_topics()`  |
| `lp/buffer.py`                               | Buffer GraphQL — `discover_buffer_profiles()`, `post_draft_to_buffer()`, `purge_expired_show_drafts()`, `test_buffer()` |
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

**Buffer GraphQL notes.** Fetching posts requires `organizationId` (not `channelId`) in `PostsInput`. Updating posts uses `editPost` mutation (not `updatePost`). Facebook edits require `metadata.facebook.type`. Instagram edits require `metadata.instagram.type` + at least one image (use the LP logo placeholder). Posts are paginated — use `pageInfo.hasNextPage` / `endCursor` to fetch all pages.

**Em dash convention.** Posts are written with `—` (no spaces). The `replace(" — ", "—")` normalization in `main.py` catches anything Claude generates with spaces. `clean_up.py --fix-dashes` retroactively cleans existing Buffer drafts.

**Show announcements (calendar pipeline).** `fetch_upcoming_shows()` pulls from Airtable base `appXLETHThc0p5MOz` (env: `AIRTABLE_CALENDAR_BASE_ID`), table `tblK2LMog1WUEv3j0`. Filters for `LPC Contract Status = "(FE) Fully Executed"` and show dates within the next 7 days (`SHOW_DAYS_AHEAD`). Dedup key is `lpc_{LPC #}` stored in the url column of Sheets. Test with `python main.py --test-calendar`.

**Artist names are exact.** The `artists.md` table maps tribute act names exactly as they appear in Airtable. Claude searches for the exact name — do not paraphrase. "Concert of Kings" (not "Elvis: The Concert of Kings"). Priscilla Presley is a separate act from Concert of Kings. **Note:** Airtable still has the old name "Elvis: The Concert of Kings" — needs to be updated there too for the mapping to work correctly.

**Expired draft cleanup.** `purge_expired_show_drafts()` lives in `lp/buffer.py` but is **commented out** at the top of `main()` in `main.py`. It detects show announcement drafts with past dates and deletes them. Uncomment when ready — always test with `--dry-run` first. The same logic is active (not commented out) in `clean_up.py --purge-expired`.

**Dry run output.** `main.py --dry-run` prints full post text (not truncated) with scheduled slot per post. Safe to run to preview output without writing to Sheets or Buffer.

**Scoring and scheduling (artist news pipeline).** `main.py` now runs the artist news pipeline in three phases: (1) search all artists and collect every candidate topic, (2) score all candidates in one Haiku call (`score_and_rank_topics()` in `lp/ai.py`) using five criteria (relevance, freshness, velocity, virality, uniqueness) plus an exclusivity +1 bonus for Top of Roster / Exclusive acts and two hardcoded exceptions (Tony Danza, The Rocket Man Show), (3) generate posts and schedule to Buffer only for the top-scoring topics — one per remaining day in the current Mon–Sun week, at 10am ET, max 7. Topics scoring below 0.40 are dropped. `get_week_slots()` in `main.py` always returns exactly 7 slots — the next 7 days from now at 10am ET — so a full week is filled regardless of what day the script runs. Show announcements bypass scoring and are always drafted without a scheduled time. `post_draft_to_buffer()` accepts an optional `scheduled_at: datetime` — when set, uses `schedulingType: "scheduled"` and a Unix timestamp in seconds; otherwise falls back to queue mode.

**Platform routing by hook type.** Topics with `hook_type == "original_artist_news"` are posted to Instagram and Facebook only — never LinkedIn. Tribute act news (`tribute_news`, `upcoming_show`) goes to all three platforms. The `hook_type` field is set by Claude during the news search in `lp/ai.py`.

## Environment variables (`.env`)

```
ANTHROPIC_API_KEY
BUFFER_API_KEY
AIRTABLE_API_KEY
AIRTABLE_BASE_ID        # default: appMMwX47V1g2Sv5u
AIRTABLE_ARTIST_TABLE   # default: tbloEhiPP4kyTTVDb
FOUND_NEWS_STORIES_SHEETS_ID
COST_CAP_USD            # default: 5.00
GOOGLE_APPLICATION_CREDENTIALS_JSON  # service account JSON string (CI/Actions)
```

## Adding or editing artists

Edit `content-skill-graph/engine/artists.md`. The table maps `Tribute Act | Original Artist`. Leave Original Artist blank if not applicable. Changes take effect on the next run — no code change needed.

## Extending the skill graph

Drop a new `.md` file anywhere under `content-skill-graph/`. It gets picked up automatically by `load_skill_graph()` (alphabetical sort). Use the existing files as style reference — keep them instructional, not descriptive.

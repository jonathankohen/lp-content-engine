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
  → Claude Haiku web search (news per artist) + whitelisted music-news RSS feeds (lp/feeds.py)
  → Google Sheets dedup → Claude Sonnet content generation → Buffer drafts
  [Phase 4] → remaining slots → Claude Haiku trivia search (original artists) → Claude Sonnet posts → Buffer drafts
  [Phase 5] → still-remaining slots → Claude Haiku historical fact search → Claude Sonnet posts → Buffer drafts

Buffer analytics → top performers → injected as style examples into every generate_posts() call

Every topic queued to Buffer (shows, artist news, trivia, historical facts)
  → Claude Sonnet long-form article (generate_article) → LP News WordPress plugin → draft post on loveproductions.com
```

Show announcements run first, then artist news (Phases 1–3), then original-artist trivia (Phase 4), then historical facts (Phase 5) to fill any still-remaining slots. All phases feed the same dedup store and Buffer queue. **Every** topic that gets queued to Buffer also gets a dedicated long-form article drafted as a standard WordPress post on loveproductions.com (see "Website news posts" below).

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

python main.py --news-count N               # news-only mode: create up to N news posts queued to Buffer (no schedule), skip shows

python main.py --news-from-buffer           # convert current Buffer FB drafts (status 'draft') into website news posts and exit
python main.py --news-from-queue            # convert current Buffer FB queue (status 'scheduled') into website news posts and exit
python main.py --news-from-week [DAYS]      # convert past DAYS (default 7) of sent FB posts into website news posts and exit

python clean_up.py          # preview expired show draft purge
python clean_up.py --apply  # delete expired show drafts
```

## Key files

| File                                         | Purpose                                                                                                                 |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `main.py`                                    | Thin orchestration layer + CLI entry point                                                                              |
| `clean_up.py`                                | Buffer draft maintenance — fixes em/en dash spacing and purges expired show announcement drafts                         |
| `lp/config.py`                               | All env vars, constants, cost tracking state + helpers, `load_env()`                                                    |
| `lp/loaders.py`                              | `load_skill_graph()`, `load_artist_mappings()`                                                                          |
| `lp/airtable.py`                             | `fetch_airtable_artists()`, `fetch_upcoming_shows()`, `show_to_topic()`, `fetch_venue_from_contracts()`                 |
| `lp/sheets.py`                               | Google Sheets read/write — `read_used_topics()`, `mark_topics_used()`, `mark_show_used()`, `lookup_ticket_url()`        |
| `lp/ai.py`                                   | Claude helpers — `search_artist_news()`, `search_trivia()`, `search_historical_facts()`, `score_and_rank_topics()`, `generate_posts()`, `generate_article()`, `default_categories()`, `format_performance_context()` |
| `lp/buffer.py`                               | Buffer GraphQL — `discover_buffer_profiles()`, `post_draft_to_buffer()`, `fetch_top_performers()`, `purge_expired_show_drafts()`, `test_buffer()` |
| `lp/wordpress.py`                            | LP News client — `publish_news_posts()` posts a batch of news items to the LP News WordPress plugin (draft posts on loveproductions.com) |
| `lp/scrape.py`                               | `fetch_og_image()` — cheap og:image scrape of a topic's source URL (no AI call), for native image assets                |
| `lp/feeds.py`                                | `load_feed_items()`, `search_artist_feeds()` — whitelisted music-news RSS ingestion (no AI call), a second news channel alongside web search |
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

**Buffer GraphQL notes.** Fetching posts requires `organizationId` (not `channelId`) in `PostsInput`. Updating posts uses `editPost` mutation (not `updatePost`). **Deleting posts:** `deletePost` returns the union `DeletePostPayload` whose members are `DeletePostSuccess { id }` and `VoidMutationError { message }` — **not** the `PostActionSuccess`/`MutationError` union used by create/edit. Spreading the wrong fragments causes a silent GraphQL 400 (`_buffer_gql` then returns `{}`); the delete helpers now treat success as `"id" in result` so an empty/errored response is logged as a failure, not a false "Deleted". Both delete paths (`purge_expired_show_drafts()` in `lp/buffer.py` and `_delete_post()` in `clean_up.py`) use this corrected shape. Facebook edits require `metadata.facebook.type`. Instagram edits require `metadata.instagram.type` + at least one image (use the LP logo placeholder). **Note:** Buffer removed `status` and `after` (pagination cursor) from `PostsInput` in a 2026 API update. `get_occupied_slots()` now fetches a single page only (no pagination) — sufficient for checking this week's 7 slots. `fetch_top_performers()` uses a `statistics` field that may or may not be available; it degrades gracefully to `[]` if not.

**Em dash convention.** Posts are written with `—` (no spaces). The `replace(" — ", "—")` normalization in `main.py` catches anything Claude generates with spaces. `clean_up.py --fix-dashes` retroactively cleans existing Buffer drafts. The skill graph enforces a hard cap of **one em dash per post maximum** (see `humanizer.md`).

**Show announcements (calendar pipeline).** `fetch_upcoming_shows()` pulls from Airtable base `appXLETHThc0p5MOz` (env: `AIRTABLE_CALENDAR_BASE_ID`), table `tblK2LMog1WUEv3j0`. Filters for `LPC Contract Status = "(FE) Fully Executed"` and show dates within the next 7 days (`SHOW_DAYS_AHEAD`). Dedup key is `lpc_{LPC #}` stored in the url column of Sheets. Test with `python main.py --test-calendar`. Show announcements are scheduled to the **first available week slots** (10am ET), before any artist news posts claim them. Artist news then fills the remaining slots.

**Artist names are exact.** The `artists.md` table maps tribute act names exactly as they appear in Airtable. Claude searches for the exact name — do not paraphrase. "Concert of Kings" (not "Elvis: The Concert of Kings"). Priscilla Presley is a separate act from Concert of Kings. **Note:** Airtable still has the old name "Elvis: The Concert of Kings" — needs to be updated there too for the mapping to work correctly.

**Expired draft cleanup.** `purge_expired_show_drafts()` lives in `lp/buffer.py` but is **commented out** at the top of `main()` in `main.py`. It detects show announcement drafts with past dates and deletes them. Uncomment when ready — always test with `--dry-run` first. The same logic is active (not commented out) in `clean_up.py --purge-expired`.

**Dry run output.** `main.py --dry-run` prints full post text (not truncated) with scheduled slot per post. Safe to run to preview output without writing to Sheets or Buffer.

**Scoring and scheduling (artist news pipeline).** `main.py` runs in five phases: (1) search all artists and collect every candidate topic, (2) score all candidates in one Haiku call (`score_and_rank_topics()` in `lp/ai.py`) using five criteria (relevance, freshness, velocity, virality, uniqueness) plus an exclusivity +1 bonus for Top of Roster / Exclusive acts and two hardcoded exceptions (Tony Danza, The Rocket Man Show), (3) generate posts and schedule to Buffer for top-scoring topics — one per remaining day at 10am ET, max 7, (4) fill any remaining week slots with original-artist trivia (`search_trivia()`), (5) fill any still-remaining slots with historical facts (`search_historical_facts()`). Phases 4 and 5 share the `_fill_with_facts()` helper in `main.py`, only run for artists with an original artist mapping that weren't already scheduled, and are both skipped in `--artist` single-artist mode. Topics scoring below 0.40 are dropped. `get_week_slots()` always returns exactly 7 slots — the next 7 days from now at 10am ET. Show announcements bypass scoring and are always drafted without a scheduled time. `post_draft_to_buffer()` accepts an optional `scheduled_at: datetime` — when set, uses `schedulingType: "scheduled"` and a Unix timestamp in seconds; otherwise falls back to queue mode.

**News-only mode (`--news-count N`).** A standalone run mode for filling Buffer with news on demand, independent of the weekly calendar. Sets `news_count` on `main()`, which: (1) **skips show announcements** entirely — both the Airtable calendar show loop (iterates an empty list) *and* tribute-act live-date items surfaced by the news search: candidates with `hook_type == "upcoming_show"` are dropped from the pool before scoring/selection (the same items the website step already skips via `_build_news_post`), so freed slots backfill with real news/trivia/historical and the run still targets N. (Normal weekly runs keep `upcoming_show` news items to promote gigs — the drop is news-only-mode only.) (2) replaces the week-slot list with `[None] * N` — a `None` slot flows through the whole pipeline as "queue this draft" (`post_draft_to_buffer(scheduled_at=None)` → Buffer queue mode, no fixed time), ignoring which slots are already occupied, (3) targets **up to N** posts: artist news first (Phase 3, capped at N via `_select_with_diversity(ranked, N)`), then tops up any shortfall with trivia (Phase 4) and historical facts (Phase 5) to reach N. Fewer than N real news stories → fewer real-news posts, filled to N by trivia/historical. All the slot-formatting log lines guard for `None` ("queue"/"queued"), and `_fill_with_facts()` uses today (ET) as the search-date context when the slot is `None` (needed because `search_historical_facts()` requires a real date). Website news posts, dedup, and cost caps behave exactly as in a normal run. Combine with `--dry-run` to preview.

**Music-news RSS feeds (second news channel).** `lp/feeds.py` reads a curated whitelist of music-outlet RSS feeds (`MUSIC_FEEDS`) as a second candidate-topic source *alongside* Anthropic's hosted `web_search` (used by `search_artist_news()`). RSS is the publisher-sanctioned syndication channel: we read headline + summary + the outlet's own article link, never article bodies. In Phase 1 of `main()`, `load_feed_items()` fetches all feeds **once** (stdlib `xml.etree` + `requests`, no new dependency, never raises, in-process cache, items filtered to the last `FEED_DAYS`=14 days). For each artist, `search_artist_feeds(name, original, feed_items)` returns items whose title/summary mention the tribute act (→ `tribute_news`) or an original artist (→ `original_artist_news`), in the exact dict shape `search_artist_news()` returns, so they flow through the same score → dedup → generate → Buffer path. Feed + web-search results are deduped by url/headline within each artist's batch before `filter_new_topics()`. The **original-artist live-event hard rule** is enforced here too: any `original_artist_news` item matching `_LIVE_EVENT_RE` is dropped. Name matching uses word boundaries and a 4-char minimum to avoid ambiguous short-name collisions; original-artist cells with multiple names are split on `,`/`&`/`/`/`and`. **Whitelist (as of 2026-07-03):** Billboard, Pitchfork (`/feed/feed-news/rss`), Rolling Stone, Stereogum, Brooklyn Vegan, American Songwriter, NME. Pitchfork's feed works from our own server via `requests` (Condé Nast only IP-blocks Anthropic's WebFetch/web_search infra, not us). **Deliberately excluded:** Consequence (robots.txt disallows `/feed/` and blocks ClaudeBot). To add/remove an outlet, edit `MUSIC_FEEDS` in `lp/feeds.py` — verify the feed's robots.txt permits it first. Note tribute-name/original-artist nickname collisions (e.g. "The Fab Four" the act vs. "Fab Four" the Beatles nickname) can classify an item as `tribute_news`; harmless (still relevant, routes to all platforms).

**Platform routing by hook type.** Topics with `hook_type` of `"original_artist_news"`, `"trivia"`, or `"historical_fact"` are posted to Instagram and Facebook only — never LinkedIn. Tribute act news (`tribute_news`, `upcoming_show`) goes to all three platforms. `hook_type` is set by Claude during search (`search_artist_news()`), or hardcoded to `"trivia"` by `search_trivia()` and `"historical_fact"` by `search_historical_facts()`.

**Per-platform link/image handling (og:image scraping).** `fetch_og_image()` in `lp/scrape.py` scrapes a topic's source URL for its Open Graph preview image (og:image → og:image:url → twitter:image fallback), resolving relative URLs and returning `None` on any failure. It's cheap — one `requests.get()` with a 5s timeout + regex, **no AI call**, in-run cache by URL, no new dependency. Called once per topic in all three posting blocks of `main.py`; `_image_for(platform, og_image)` then routes the result:
- **LinkedIn** — image-only post, **no link card**. Gets the scraped image as a native asset; if none is found, posts text-only. The source URL is deliberately **omitted** from the LinkedIn body (see `generate_posts()` instruction) so LinkedIn can never render a link-preview card. This is what the client wants.
- **Instagram** — uses the scraped image, falling back to the LP logo placeholder (`config._IG_PLACEHOLDER`) when none is found.
- **Facebook** — gets **no** image asset; the source URL stays in the body so Facebook auto-unfurls its own native link-preview card (with the og:image). `generate_posts()` now weaves the source URL into the **Facebook post only** (previously LinkedIn + Facebook).

`post_draft_to_buffer()` attaches `assets.image.url` for both Instagram and LinkedIn (Facebook never gets an asset). **Unverified:** that Buffer's GraphQL accepts `assets` for LinkedIn channels — confirm with one live (non-dry-run) post; if rejected, the LinkedIn asset/metadata shape may need adjustment (e.g. a `metadata.linkedin` field).

**Trivia (Phase 4).** After the artist news pipeline fills its slots, the next remaining week slots are filled by `search_trivia()` — surprising, lesser-known facts about the **original artist**, any era, not date-bound (distinct from historical facts, which are pre-1990 archival and date-anchored). Sits just below original-artist news in the hierarchy and **above** historical facts. Instagram + Facebook only (`hook_type = "trivia"`), generated only for artists with an original artist mapping that weren't already scheduled this run, and excluded in `--artist` single-artist mode. Deduped via the same Sheets mechanism as news topics. Hook guidance lives in `content-skill-graph/engine/hooks.md` ("Trivia about Artist Represented by Tribute").

**Historical facts (Phase 5).** After trivia fills what it can, any still-remaining week slots are filled by `search_historical_facts()` — pre-1990 archival facts about original artists from sources like archive.org, old Rolling Stone, Billboard, etc. These are Instagram + Facebook only, lowest priority, generated only for artists with an original artist mapping, and excluded in `--artist` single-artist mode. Deduped via the same Sheets mechanism as news topics. Phases 4 and 5 share the `_fill_with_facts()` helper in `main.py`.

**Original artists' shows are never news (hard rule).** We never announce an original artist's own tour, concert, residency, festival appearance, or any live-performance date and then point to our tribute band — it looks bad. Enforced in two places: (1) `search_artist_news()` instructs the model to never return original-artist live-event news and to set an `is_live_event` boolean per item; (2) a code-level guard in `search_artist_news()` drops any `original_artist_news` item with `is_live_event == True`, regardless of how the model classified `hook_type`. Original-artist *non-live* news (album/release, award, induction, biopic, anniversary, milestone, passing, etc.) is still eligible. The rule and an updated example also live in `content-skill-graph/engine/hooks.md` under "News from Artist Represented by Tribute". Tribute acts' own live dates are unaffected — those are exactly what we promote.

**Ticket links and venue names in show announcements.** `lookup_ticket_url()` in `lp/sheets.py` looks up a ticket URL *and* venue name in the tour dates Google Sheet (`TOUR_DATES_SHEET_ID`) produced by the `lp-tour-dates` repo (formerly `love-automations`; local dir `/Users/jonathankohen/lp-tour-dates`). The sheet has per-artist tabs named by display name (e.g., "Arrival from Sweden" for "Arrival from Sweden: The Music of ABBA") with columns: Date (MM/DD/YY), Venue, City, Region, Country, Ticket URL, Source. Matching is by tab name (substring of show_title, case-insensitive) + date. If no venue name is found there, `fetch_venue_from_contracts()` in `lp/airtable.py` falls back to the "LPI - Contracts" table in the calendar Airtable base (`AIRTABLE_CALENDAR_BASE_ID`), filtering by `LPC #`. The `Venue` field from that table is used to replace the raw address in the headline and summary. If neither source has a venue name, the address is used as-is.

**Website news posts (LP News plugin).** Every topic queued to Buffer is also drafted as a standard WordPress **post** on loveproductions.com. After Buffer drafting in each block (shows, artist news Phase 3, and trivia/historical via `_fill_with_facts`), `_build_news_post()` in `main.py` calls `generate_article()` (a dedicated Sonnet call, separate from the social copy) to produce a 2–4 paragraph web article `{title, body, categories}`, then packages it with an image and a button URL. The batch is sent once at the end of the run via `publish_news_posts()` in `lp/wordpress.py` to the **LP News** WordPress plugin (lives in this repo at `wordpress-plugin/lp-news/lp-news.php`; deployed to loveproductions.com). The plugin creates each post as a **draft** for review, assigns built-in `category` terms, sideloads the featured image, and appends a red "Read more" button.
- **Categories** — chosen from a fixed 11-item list (`NEWS_CATEGORIES` in `lp/ai.py`: Celebration, Celebrity, Theatre, Tour, Condolences, Festival, Interview, Sold Out, Tribute, TV Show, Uncategorized). Hybrid: `default_categories(hook_type)` seeds a deterministic default (e.g. `tribute_news→Tribute`, `upcoming_show→Tour`), then Claude adjusts within the allowed list during `generate_article()`. The plugin re-filters server-side to the allowlist (dropping anything off-list) and creates allowlisted terms that don't yet exist.
- **Button URL ("always link somewhere")** — `_news_button_url()` falls back through: source URL → ticket URL → the act's loveproductions.com page (`artist_url` from Airtable) → the homepage (`config.LP_HOMEPAGE`). Label is always "Read more".
- **No show announcements** — the client does not want show/ticket announcements in the website news section. In the weekly pipeline, `_build_news_post()` returns `None` for any `upcoming_show` topic (no article, no Claude call). In the Buffer backfills (`--news-from-buffer` = drafts, `--news-from-queue` = scheduled/queued posts, `--news-from-week` = sent posts — all three share `_buffer_posts_to_news()`), reconstructed topics carry no `hook_type`, so `classify_show_announcements()` in `lp/ai.py` (one batched Haiku call) flags which Buffer posts are live-event/show announcements and they're skipped. A post counts as a show announcement if its primary purpose is promoting attendance at a specific upcoming live event/venue/date (concert, show, residency, festival, **or a personal/live appearance**) — general artist news that merely mentions the tribute act or a release date does not. (A regex helper `is_show_announcement()` exists in `lp/buffer.py` for the expired-draft purge, but it false-positives on tribute-act names containing "concert"/"show" and misses non-keyword events like "personal appearance", so the backfill uses the Haiku classifier instead.) Note: `_extract_earliest_date()` in `lp/buffer.py` now strips ordinal suffixes ("July 3rd"→"July 3") before parsing — previously such dates never parsed, so the expired-show purge silently missed them.
- **Image** — **only** the source article's own og:image (`fetch_og_image`). There is deliberately **no fallback** to the artist's profile photo or the LP logo placeholder: if the article has no image, `image_url` is sent as `""` and the plugin drafts the post with no featured image for manual handling. (The plugin already treats an empty `image_url` as "no featured image".)
- **Dedup** — the plugin stores a `_lp_news_key` meta per post (source URL → show sheet key → headline) and skips re-sends, so re-runs never duplicate a story. This is independent of the Sheets/Buffer dedup.
- **Disabled when unconfigured** — if `LP_NEWS_URL`/`LP_NEWS_SECRET` are unset, the website step is skipped (Buffer drafting is unaffected) and the planned posts are logged. `--dry-run` sends `dry_run: true` to the plugin (plans, no writes) when configured.
- **Artist page URLs** — `fetch_airtable_artists()` now requests **all** artist fields (not a fixed subset) and `_artist_url_from_fields()` scans values for a loveproductions.com URL → `artist_url` on each artist dict. This is the source URL fallback for source-less items (trivia/historical/shows without a ticket link). If the artists table has no such field, `artist_url` is "" and the button falls back to the homepage.
- **Cost** — `generate_article()` is a second Sonnet call per topic (on top of `generate_posts()`), gated by the same `COST_CAP_USD` / `CLAUDE_CALL_LIMIT`. ~7 articles/run adds roughly $0.10–0.15.

**Self-learning via performance context.** `fetch_top_performers()` in `lp/buffer.py` delegates to `fetch_meta_top_performers()` in `lp/meta.py`, which queries the Meta Graph API for recent Facebook Page posts ranked by shares + reactions. `format_performance_context()` in `lp/ai.py` formats the top 3 as style examples injected into every `generate_posts()` call. Instagram is supported in the code but requires the LP Instagram account to be linked to the Facebook Page in Meta Business settings. Buffer's own `statistics` field is paywalled — Meta Graph API is the correct data source. Token setup: run `python get_page_token.py` after generating a fresh User Access Token in Graph API Explorer with these permissions: `pages_read_engagement`, `pages_show_list`, `instagram_basic`, `pages_read_user_content`, `pages_manage_posts`. The resulting Page Access Token does not expire. Notes: `me/posts` fails for New Pages Experience pages — use `{page_id}/posts` instead. `reactions.summary(true)` and `comments.summary(true)` require `pages_read_user_content` + `pages_manage_posts` in addition to `pages_read_engagement`.

## Environment variables (`.env`)

```
ANTHROPIC_API_KEY
BUFFER_API_KEY
AIRTABLE_API_KEY
AIRTABLE_BASE_ID        # default: appMMwX47V1g2Sv5u
AIRTABLE_ARTIST_TABLE   # default: tbloEhiPP4kyTTVDb
FOUND_NEWS_STORIES_SHEETS_ID
TOUR_DATES_SHEET_ID     # optional: tour dates sheet from the lp-tour-dates repo (formerly love-automations) (for ticket links)
COST_CAP_USD            # default: 5.00
GOOGLE_APPLICATION_CREDENTIALS_JSON  # service account JSON string (CI/Actions)
LINKEDIN_ANALYTICS_CSV  # optional: path to manually exported LinkedIn analytics CSV
LP_NEWS_URL             # optional: LP News plugin endpoint, e.g. https://www.loveproductions.com/wp-json/lp-news/v1/publish-news
LP_NEWS_SECRET          # optional: secret from the WP "Settings → LP News" page (X-LP-News-Secret header). Unset → website posting skipped
```

## LinkedIn analytics (self-learning)

LinkedIn API access was applied for but not approved. Instead, export analytics manually:

1. Go to the LinkedIn Company Page admin → **Analytics → Updates**
2. Click **Export** (top right) to download a CSV
3. Set `LINKEDIN_ANALYTICS_CSV=/path/to/export.csv` in `.env`
4. Run `python main.py --test-analytics` to confirm LinkedIn posts appear in the output

`lp/linkedin.py` parses the CSV and feeds the top performers into `fetch_top_performers()` alongside Meta posts. Re-export monthly (or whenever) to refresh the data. The CSV column names LinkedIn uses are: `Post title`, `Impressions`, `Likes`, `Comments`, `Shares`.

## Adding or editing artists

Edit `content-skill-graph/engine/artists.md`. The table maps `Tribute Act | Original Artist`. Leave Original Artist blank if not applicable. Changes take effect on the next run — no code change needed.

## Extending the skill graph

Drop a new `.md` file anywhere under `content-skill-graph/`. It gets picked up automatically by `load_skill_graph()` (alphabetical sort). Use the existing files as style reference — keep them instructional, not descriptive.

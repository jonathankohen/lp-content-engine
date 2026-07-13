import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── API keys & IDs ────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BUFFER_API_KEY = os.environ.get("BUFFER_API_KEY", "")
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY", "")

AIRTABLE_BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "")
AIRTABLE_TABLE_ID = os.environ.get("AIRTABLE_ARTIST_TABLE", "")
SHEETS_ID = os.environ.get("FOUND_NEWS_STORIES_SHEETS_ID", "")

AIRTABLE_CALENDAR_BASE_ID = os.environ.get("AIRTABLE_CALENDAR_BASE_ID", "")
AIRTABLE_CALENDAR_TABLE_ID = os.environ.get("AIRTABLE_CALENDAR_TABLE_ID", "")
TOUR_DATES_SHEET_ID = os.environ.get("TOUR_DATES_SHEET_ID", "")

META_PAGE_ACCESS_TOKEN = os.environ.get("META_PAGE_ACCESS_TOKEN", "")
LINKEDIN_ANALYTICS_CSV = os.environ.get("LINKEDIN_ANALYTICS_CSV", "")

# LP News WordPress plugin (loveproductions.com) — drafts a news post per topic.
# Set LP_NEWS_URL to the plugin's publish endpoint and LP_NEWS_SECRET to the
# secret shown on its Settings → LP News page. When LP_NEWS_URL is unset, the
# website-posting step is skipped (Buffer drafting is unaffected).
LP_NEWS_URL = os.environ.get("LP_NEWS_URL", "")
LP_NEWS_SECRET = os.environ.get("LP_NEWS_SECRET", "")

# ── Constants ─────────────────────────────────────────────────────────────────

AIRTABLE_PRIORITY_ORDER = ["Top of Roster", "Exclusive", "Core Roster"]
SHOW_DAYS_AHEAD = 7

# Baseline score assigned to an upcoming-show announcement when shows and news
# compete for the same week slots (see main.py). Sits in the normal news-score
# range so a strong/exclusive news story can outrank a routine show. The same
# exclusivity bonus news gets (ai.exclusivity_bonus) is added on top for shows
# by exclusive/top-of-roster acts. Tunable via env.
SHOW_BASE_SCORE = float(os.environ.get("SHOW_BASE_SCORE", "0.75"))

SEARCH_MODEL = "claude-haiku-4-5"
CONTENT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

BUFFER_API_URL = "https://api.buffer.com"
SKILL_GRAPH_DIR = Path(__file__).parent.parent / "content-skill-graph"
_IG_PLACEHOLDER = "https://www.loveproductions.com/wp-content/uploads/2022/03/LPI_logo_RGB_Red_BLK.png"
LP_HOMEPAGE = "https://www.loveproductions.com"

# ── Cost tracking ─────────────────────────────────────────────────────────────

COST_CAP_USD = float(os.environ.get("COST_CAP_USD", "5.00"))
_HAIKU_INPUT = 1.00 / 1_000_000
_HAIKU_OUTPUT = 5.00 / 1_000_000
_SONNET_INPUT = 3.00 / 1_000_000
_SONNET_OUTPUT = 15.00 / 1_000_000
_SEARCH_COST = 0.01

estimated_cost_usd = 0.0
claude_call_count = 0
CLAUDE_CALL_LIMIT = 50


def track_cost(resp, model: str) -> None:
    global estimated_cost_usd
    usage = getattr(resp, "usage", None)
    if usage:
        if "sonnet" in model:
            in_cost, out_cost = _SONNET_INPUT, _SONNET_OUTPUT
        else:
            in_cost, out_cost = _HAIKU_INPUT, _HAIKU_OUTPUT
        estimated_cost_usd += getattr(usage, "input_tokens", 0) * in_cost
        estimated_cost_usd += getattr(usage, "output_tokens", 0) * out_cost
    server_tool = getattr(getattr(resp, "usage", None), "server_tool_use", None)
    searches = getattr(server_tool, "web_search_requests", 0) if server_tool else 0
    estimated_cost_usd += searches * _SEARCH_COST
    log.debug("Est. cost so far: $%.4f / $%.2f cap", estimated_cost_usd, COST_CAP_USD)


def under_cost_cap(label: str) -> bool:
    if estimated_cost_usd >= COST_CAP_USD:
        log.warning(
            "Cost cap $%.2f reached (est. $%.4f) — skipping %s",
            COST_CAP_USD,
            estimated_cost_usd,
            label,
        )
        return False
    return True


# ── Claude rate-limit throttle ────────────────────────────────────────────────

_THROTTLE_FILE = "/tmp/lp_content_throttle.txt"
_THROTTLE_BUFFER = 2


def _load_throttle() -> float:
    try:
        return float(Path(_THROTTLE_FILE).read_text().strip())
    except Exception:
        return 0.0


def _save_throttle(t: float) -> None:
    try:
        Path(_THROTTLE_FILE).write_text(str(t))
    except Exception:
        pass


def claude_throttle() -> None:
    wait = _load_throttle() - time.time()
    if wait > 0:
        log.info("Rate limit: waiting %.0fs...", wait)
        time.sleep(wait)


def claude_call_done(headers: dict) -> None:
    global claude_call_count
    claude_call_count += 1
    reset_str = headers.get("anthropic-ratelimit-input-tokens-reset") or headers.get(
        "anthropic-ratelimit-tokens-reset"
    )
    if reset_str:
        try:
            reset_dt = datetime.fromisoformat(reset_str.replace("Z", "+00:00"))
            next_at = reset_dt.timestamp() + _THROTTLE_BUFFER
            _save_throttle(next_at)
            log.info(
                "Token reset at %s — next call allowed in %.0fs",
                reset_str,
                max(0, next_at - time.time()),
            )
            return
        except Exception:
            pass
    _save_throttle(time.time() + 90)


# ── Env validation ────────────────────────────────────────────────────────────


def load_env() -> None:
    gc_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
    if gc_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(gc_json)
        tmp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        log.info("Wrote Google credentials from env to %s", tmp.name)

    li_content = os.environ.get("LINKEDIN_ANALYTICS_CSV_CONTENT", "")
    if li_content and not os.environ.get("LINKEDIN_ANALYTICS_CSV"):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write(li_content)
        tmp.flush()
        os.environ["LINKEDIN_ANALYTICS_CSV"] = tmp.name
        global LINKEDIN_ANALYTICS_CSV
        LINKEDIN_ANALYTICS_CSV = tmp.name
        log.info("Wrote LinkedIn analytics CSV from env to %s", tmp.name)

    missing = [
        k
        for k, v in {
            "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
            "BUFFER_API_KEY": BUFFER_API_KEY,
            "AIRTABLE_API_KEY": AIRTABLE_API_KEY,
            "FOUND_NEWS_STORIES_SHEETS_ID": SHEETS_ID,
        }.items()
        if not v
    ]
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

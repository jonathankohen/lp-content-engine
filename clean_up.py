"""
clean_up.py — Buffer draft maintenance utility.

Two operations, both dry-run by default:

  1. Fix dashes   — strip spaces around em/en dashes in all drafts
  2. Purge expired — delete show announcement drafts whose date has passed

Usage:
  python clean_up.py                      # preview both operations
  python clean_up.py --apply              # apply both
  python clean_up.py --fix-dashes         # preview dash fixes only
  python clean_up.py --purge-expired      # preview expired purge only
  python clean_up.py --fix-dashes --apply # apply dash fixes only
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

BUFFER_API_URL = "https://api.buffer.com"
BUFFER_API_KEY = os.environ.get("BUFFER_API_KEY", "")

# ── Buffer API ─────────────────────────────────���──────────────────────────────

def _gql(query: str, variables: dict | None = None) -> dict:
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    for attempt in range(4):
        resp = requests.post(
            BUFFER_API_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {BUFFER_API_KEY}",
            },
            timeout=15,
        )
        if resp.status_code == 429:
            try:
                body = resp.json()
                window = body.get("errors", [{}])[0].get("extensions", {}).get("window", "")
                retry_after = body.get("retryAfter") or body.get("retry_after")
            except Exception:
                window, retry_after = "", None
            if window == "24h":
                sys.exit("Buffer daily API limit reached. Try again tomorrow.")
            wait = int(retry_after) if retry_after else 61
            print(f"Rate limited — retrying in {wait}s...")
            time.sleep(wait)
            continue
        if not resp.ok:
            sys.exit(f"Buffer API error {resp.status_code}: {resp.text}")
        return resp.json()
    sys.exit("Buffer rate limit persists. Try again in a minute.")


def _get_org_and_channels() -> tuple[str, list[str]]:
    data = _gql("query { account { organizations { id name } } }")
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        sys.exit("No Buffer organizations found — check BUFFER_API_KEY")
    org_id = orgs[0]["id"]
    print(f"Org: {orgs[0].get('name', '')} ({org_id})")

    data = _gql(
        "query GetChannels($input: ChannelsInput!) { channels(input: $input) { id service displayName } }",
        {"input": {"organizationId": org_id}},
    )
    channels = data.get("data", {}).get("channels", [])
    ids = [c["id"] for c in channels if c.get("service", "").lower() in ("linkedin", "instagram", "facebook")]
    print(f"Channels: {len(ids)}\n")
    return org_id, ids


def _get_drafts(org_id: str) -> list[dict]:
    all_nodes = []
    cursor = None
    while True:
        variables: dict = {"input": {"organizationId": org_id}}
        if cursor:
            variables["after"] = cursor
        data = _gql(
            """
            query GetDrafts($input: PostsInput!, $after: String) {
              posts(input: $input, after: $after) {
                edges { node { id text status channel { service } } cursor }
                pageInfo { hasNextPage endCursor }
              }
            }
            """,
            variables,
        )
        posts_data = data.get("data", {}).get("posts", {})
        edges = posts_data.get("edges", [])
        all_nodes.extend(e["node"] for e in edges if e.get("node"))
        page_info = posts_data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        time.sleep(1)
    return [n for n in all_nodes if n.get("status", "").lower() == "draft"]


def _update_post(post: dict, text: str) -> bool:
    post_id = post["id"]
    service = post.get("channel", {}).get("service", "").lower()

    _IG_PLACEHOLDER = "https://www.loveproductions.com/wp-content/uploads/2022/03/LPI_logo_RGB_Red_BLK.png"

    edit_input: dict = {
        "id": post_id,
        "text": text,
        "schedulingType": "automatic",
        "mode": "addToQueue",
    }
    if service == "facebook":
        edit_input["metadata"] = {"facebook": {"type": "post"}}
    elif service == "instagram":
        edit_input["metadata"] = {"instagram": {"type": "post", "shouldShareToFeed": True}}
        edit_input["assets"] = {"images": [{"url": _IG_PLACEHOLDER}]}

    result = _gql(
        """
        mutation EditPost($input: EditPostInput!) {
          editPost(input: $input) {
            ... on PostActionSuccess { post { id } }
            ... on MutationError { message }
          }
        }
        """,
        {"input": edit_input},
    ).get("data", {}).get("editPost", {})
    if "message" in result:
        print(f"  ERROR: {result['message']}")
        return False
    return True


def _delete_post(post_id: str) -> bool:
    result = _gql(
        """
        mutation DeletePost($input: DeletePostInput!) {
          deletePost(input: $input) {
            ... on PostActionSuccess { post { id } }
            ... on MutationError { message }
          }
        }
        """,
        {"input": {"id": post_id}},
    ).get("data", {}).get("deletePost", {})
    if "message" in result:
        print(f"  ERROR deleting {post_id}: {result['message']}")
        return False
    return True


# ── Dash fixing ──────────────────────────────��────────────────────────────────

def _fix_dashes(text: str) -> str:
    return text.replace(" — ", "—").replace(" – ", "–")


def run_fix_dashes(org_id: str, apply: bool) -> None:
    print("── Fix dashes ────────────────────���──────────────────────")
    changed = skipped = 0
    for post in _get_drafts(org_id):
            text = post.get("text", "")
            fixed = _fix_dashes(text)
            if fixed == text:
                skipped += 1
                continue
            print(f"\nPost {post['id']}:")
            print(f"  BEFORE: {text[:120]!r}")
            print(f"  AFTER:  {fixed[:120]!r}")
            if apply:
                ok = _update_post(post, fixed)
                print(f"  {'Updated.' if ok else 'FAILED.'}")
            changed += 1
    print(f"\n{'Applied' if apply else 'Would fix'} {changed} post(s). {skipped} already clean.\n")


# ── Expired show purge ────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:,?\s+\d{4})?\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)
_SHOW_KEYWORDS = re.compile(
    r"\b(ticket|tickets|show|concert|performance|doors open|venue|live at|"
    r"on sale|book now|get your tickets|don't miss|link in bio)\b",
    re.IGNORECASE,
)


def _extract_earliest_date(text: str) -> datetime | None:
    today = datetime.now(tz=timezone.utc).date()
    earliest = None
    for match in _DATE_RE.finditer(text):
        raw = match.group()
        for fmt in (
            "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
            "%B %d", "%b %d",
            "%d %B %Y", "%d %b %Y", "%d %B", "%d %b",
            "%m/%d/%Y", "%m/%d/%y",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(raw.strip(), fmt)
                if dt.year == 1900:
                    dt = dt.replace(year=today.year)
                    if dt.date() < today:
                        dt = dt.replace(year=today.year + 1)
                if earliest is None or dt.date() < earliest:
                    earliest = dt.date()
                break
            except ValueError:
                continue
    return datetime(earliest.year, earliest.month, earliest.day, tzinfo=timezone.utc) if earliest else None


def _is_expired_show(text: str) -> bool:
    if not _SHOW_KEYWORDS.search(text):
        return False
    dt = _extract_earliest_date(text)
    if dt is None:
        return False
    return dt.date() < datetime.now(tz=timezone.utc).date()


def run_purge_expired(org_id: str, apply: bool) -> None:
    print("── Purge expired show drafts ───────────────────────���─────")
    found = skipped = 0
    for post in _get_drafts(org_id):
            text = post.get("text", "")
            if not _is_expired_show(text):
                skipped += 1
                continue
            print(f"\nPost {post['id']} (expired):")
            print(f"  {text[:120]!r}")
            if apply:
                ok = _delete_post(post["id"])
                print(f"  {'Deleted.' if ok else 'FAILED.'}")
            found += 1
    print(f"\n{'Deleted' if apply else 'Would delete'} {found} expired post(s). {skipped} not expired.\n")


# ── Entry point ─────────────────────────────��─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buffer draft maintenance utility")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument("--fix-dashes", action="store_true", help="Run dash fix only")
    parser.add_argument("--purge-expired", action="store_true", help="Run expired purge only")
    args = parser.parse_args()

    if not BUFFER_API_KEY:
        sys.exit("BUFFER_API_KEY not set")

    run_all = not args.fix_dashes and not args.purge_expired

    org_id, _ = _get_org_and_channels()

    if run_all or args.fix_dashes:
        run_fix_dashes(org_id, apply=args.apply)
    if run_all or args.purge_expired:
        run_purge_expired(org_id, apply=args.apply)

    if not args.apply:
        print("Dry run — pass --apply to write changes.")

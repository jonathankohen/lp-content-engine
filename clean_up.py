"""
clean_up.py — Buffer draft maintenance utility.

Purges expired show announcement drafts (dry-run by default).

Usage:
  python clean_up.py                        # preview expired drafts
  python clean_up.py --apply                # delete expired drafts
  python clean_up.py --delete-all           # preview ALL drafts for deletion
  python clean_up.py --delete-all --apply   # delete ALL drafts unconditionally
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

# ── Buffer API ────────────────────────────────────────────────────────────────

def _gql(query: str, variables: dict | None = None) -> dict:
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    for _ in range(8):
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


def _get_org_id() -> str:
    data = _gql("query { account { organizations { id name } } }")
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        sys.exit("No Buffer organizations found — check BUFFER_API_KEY")
    print(f"Org: {orgs[0].get('name', '')} ({orgs[0]['id']})\n")
    return orgs[0]["id"]


def _get_drafts(org_id: str) -> list[dict]:
    """Fetch a single page of posts (pagination removed in 2026 Buffer API update)."""
    data = _gql(
        """
        query GetDrafts($input: PostsInput!) {
          posts(input: $input) {
            edges { node { id text } }
          }
        }
        """,
        {"input": {"organizationId": org_id}},
    )
    return [
        e["node"]
        for e in data.get("data", {}).get("posts", {}).get("edges", [])
        if e.get("node")
    ]


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


def run_delete_all(org_id: str, apply: bool) -> None:
    print("── Delete all drafts ────────────────────────────────────")
    drafts = _get_drafts(org_id)
    for post in drafts:
        print(f"\nPost {post['id']}:")
        print(f"  {post.get('text', '')[:120]!r}")
        if apply:
            ok = _delete_post(post["id"])
            print(f"  {'Deleted.' if ok else 'FAILED.'}")
    print(f"\n{'Deleted' if apply else 'Would delete'} {len(drafts)} draft(s).\n")


def run_purge_expired(org_id: str, apply: bool) -> None:
    print("── Purge expired show drafts ────────────────────────────")
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buffer draft maintenance utility")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument("--delete-all", action="store_true", help="Delete ALL drafts unconditionally (use with --apply)")
    args = parser.parse_args()

    if not BUFFER_API_KEY:
        sys.exit("BUFFER_API_KEY not set")

    org_id = _get_org_id()

    if args.delete_all:
        run_delete_all(org_id, apply=args.apply)
        if not args.apply:
            print("Dry run — pass --apply to delete all drafts.")
    else:
        run_purge_expired(org_id, apply=args.apply)
        if not args.apply:
            print("Dry run — pass --apply to delete expired drafts.")

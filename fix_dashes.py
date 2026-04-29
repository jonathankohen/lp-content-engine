"""
fix_dashes.py — Strip spaces around em/en dashes in Buffer drafts.

Dry-run by default: prints what would change.
Pass --apply to write changes to Buffer.

Usage:
  python fix_dashes.py           # preview
  python fix_dashes.py --apply   # apply
"""

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BUFFER_API_URL = "https://api.buffer.com"
BUFFER_API_KEY = os.environ.get("BUFFER_API_KEY", "")


def _gql(query: str, variables: dict | None = None) -> dict:
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    for attempt in range(2):
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
                wait = int(resp.json().get("retryAfter", 61))
            except Exception:
                wait = 61
            print(f"Rate limited — retrying in {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    sys.exit("Buffer rate limit persists. Try again in a minute.")


def get_org_id() -> str:
    data = _gql("query { account { organizations { id name } } }")
    orgs = data.get("data", {}).get("account", {}).get("organizations", [])
    if not orgs:
        sys.exit("No Buffer organizations found — check BUFFER_API_KEY")
    print(f"Org: {orgs[0].get('name', '')} ({orgs[0]['id']})")
    return orgs[0]["id"]


def get_channel_ids(org_id: str) -> list[str]:
    data = _gql(
        """
        query GetChannels($input: ChannelsInput!) {
          channels(input: $input) { id service displayName }
        }
        """,
        {"input": {"organizationId": org_id}},
    )
    channels = data.get("data", {}).get("channels", [])
    ids = [c["id"] for c in channels if c.get("service", "").lower() in ("linkedin", "instagram", "facebook")]
    print(f"Channels found: {len(ids)}")
    return ids


def get_drafts(channel_id: str) -> list[dict]:
    data = _gql(
        """
        query GetDrafts($input: PostsInput!) {
          posts(input: $input) {
            edges { node { id text status } }
          }
        }
        """,
        {"input": {"channelId": channel_id, "status": ["draft"]}},
    )
    edges = data.get("data", {}).get("posts", {}).get("edges", [])
    return [e["node"] for e in edges if e.get("node")]


def fix_text(text: str) -> str:
    return text.replace(" — ", "—").replace(" – ", "–")


def update_post(post_id: str, text: str) -> bool:
    data = _gql(
        """
        mutation UpdatePost($input: UpdatePostInput!) {
          updatePost(input: $input) {
            ... on PostActionSuccess { post { id } }
            ... on MutationError { message }
          }
        }
        """,
        {"input": {"id": post_id, "text": text}},
    )
    result = data.get("data", {}).get("updatePost", {})
    if "message" in result:
        print(f"  ERROR: {result['message']}")
        return False
    return True


def main(apply: bool) -> None:
    if not BUFFER_API_KEY:
        sys.exit("BUFFER_API_KEY not set")

    org_id = get_org_id()
    channel_ids = get_channel_ids(org_id)

    changed = 0
    skipped = 0

    for channel_id in channel_ids:
        drafts = get_drafts(channel_id)
        for post in drafts:
            text = post.get("text", "")
            fixed = fix_text(text)
            if fixed == text:
                skipped += 1
                continue

            print(f"\nPost {post['id']}:")
            print(f"  BEFORE: {text[:120]!r}")
            print(f"  AFTER:  {fixed[:120]!r}")

            if apply:
                ok = update_post(post["id"], fixed)
                print(f"  {'Updated.' if ok else 'FAILED.'}")
            changed += 1

    print(f"\n{'Applied' if apply else 'Would change'} {changed} post(s). {skipped} already clean.")
    if not apply and changed:
        print("Run with --apply to write changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix em dash spacing in Buffer drafts")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = parser.parse_args()
    main(apply=args.apply)

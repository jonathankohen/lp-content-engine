"""
One-time helper: exchanges a short-lived User Access Token for a permanent
Love Productions Inc. Page Access Token and writes it to .env.

Usage:
  1. Generate a fresh User Access Token in Graph API Explorer with
     pages_read_engagement, pages_show_list, instagram_basic permissions
  2. Paste it into .env as META_USER_ACCESS_TOKEN
  3. Run: python get_page_token.py

The resulting Page Access Token does not expire.
"""
import os
import re
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

app_id = os.environ.get("META_APP_ID", "")
app_secret = os.environ.get("META_APP_SECRET", "")
user_token = os.environ.get("META_USER_ACCESS_TOKEN", "")

if not user_token:
    sys.exit("META_USER_ACCESS_TOKEN not set in .env")
if not app_id or not app_secret:
    sys.exit("META_APP_ID and META_APP_SECRET must be set in .env")

# Exchange short-lived user token for a 60-day long-lived user token
print("Exchanging for long-lived user token...")
resp = requests.get(
    "https://graph.facebook.com/v21.0/oauth/access_token",
    params={
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": user_token,
    },
    timeout=15,
)
if not resp.ok:
    sys.exit(f"Token exchange failed: {resp.text}")

long_lived_token = resp.json().get("access_token", "")
if not long_lived_token:
    sys.exit(f"No access_token in exchange response: {resp.text}")
print("Got long-lived user token.")

# Get the LP Page Access Token (derived from long-lived token = never expires)
resp = requests.get(
    "https://graph.facebook.com/v21.0/me/accounts",
    params={"fields": "id,name,access_token", "access_token": long_lived_token},
    timeout=15,
)
if not resp.ok:
    sys.exit(f"Accounts API error: {resp.text}")

pages = resp.json().get("data", [])
lp = next((p for p in pages if "Love Productions" in p.get("name", "")), None)
if not lp:
    print("Pages found:", [p["name"] for p in pages])
    sys.exit("Love Productions Inc. not found — make sure the token has pages_show_list permission")

page_token = lp["access_token"]
print(f"Found: {lp['name']} ({lp['id']})")

env_path = os.path.join(os.path.dirname(__file__), ".env")
with open(env_path) as f:
    env_content = f.read()

if "META_PAGE_ACCESS_TOKEN=" in env_content:
    env_content = re.sub(
        r"META_PAGE_ACCESS_TOKEN=.*", f"META_PAGE_ACCESS_TOKEN={page_token}", env_content
    )
else:
    env_content += f"\nMETA_PAGE_ACCESS_TOKEN={page_token}\n"

with open(env_path, "w") as f:
    f.write(env_content)

print("META_PAGE_ACCESS_TOKEN updated in .env (does not expire)")
print("Run: python main.py --test-analytics")

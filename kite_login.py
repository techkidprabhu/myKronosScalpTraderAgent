"""
Daily Zerodha Kite login. Kite access tokens expire every morning (~06:00
IST), so run this once a day before the NSE open:

    python kite_login.py

1. Open the printed URL, log in to Kite (password + 2FA).
2. Kite redirects to your app's redirect URL - copy that whole URL (or just
   its request_token=... value) and paste it here.
3. The access token is saved to data/kite_access_token (readable only by
   you). A running agent picks it up on its next reconnect attempt (every
   30 s) - no restart needed.

Needs KITE_API_KEY and KITE_API_SECRET in .env (from https://developers.kite.trade).
"""
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT / "data" / "kite_access_token"


def request_token_from(text: str) -> str:
    text = text.strip()
    if "request_token=" in text:
        return parse_qs(urlparse(text).query).get("request_token", [""])[0] or \
            text.split("request_token=", 1)[1].split("&", 1)[0]
    return text


def main():
    load_dotenv(ROOT / ".env")
    key, secret = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_API_SECRET")
    if not key or not secret:
        sys.exit("Set KITE_API_KEY and KITE_API_SECRET in .env first.")
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        sys.exit("pip install kiteconnect")
    kite = KiteConnect(api_key=key)
    print("\n1) Log in here:\n\n   " + kite.login_url() + "\n")
    token = request_token_from(input("2) Paste the redirect URL (or request_token): "))
    if not token:
        sys.exit("No request_token found.")
    session = kite.generate_session(token, api_secret=secret)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(session["access_token"])
    os.chmod(TOKEN_FILE, 0o600)
    print(f"\nSaved access token for {session.get('user_id', '?')} to {TOKEN_FILE.relative_to(ROOT)}.")
    print("The agent reconnects to Kite on its own within ~30 s.")


if __name__ == "__main__":
    main()

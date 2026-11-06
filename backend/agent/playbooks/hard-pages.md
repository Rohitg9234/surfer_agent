---
name: hard-pages
description: Cookie banners, waits, bot walls, empty snapshots. Load when the page blocks the task.
---

# Hard pages

- Cookie / chat overlay: click Accept/Close. Do not force-click a covered control. Common consent buttons are auto-dismissed when possible.
- Spinner / SPA: wait_for (text, gone, settled) rather than a tight click loop.
- Infinite list: scroll_until with the item text, or until the page stops changing.
- Bot wall / CAPTCHA: request_human_help mode=takeover. Do not retry.
- Login / 2FA: fill_credential or fill_totp from the vault; otherwise request_human_help mode=takeover. Never type_text secrets.
- Irreversible pay/delete/send: policy will ask; you can also request_human_help mode=approval.
- Empty or tiny snapshot: the UI may be inside an iframe or canvas. If `frames:` is listed, look for refs inside the snapshot; if none, stop guessing.
- Prefer a persistent profile (--user-data-dir) so logins survive runs. load_session is the JSON alternative.
- File input: upload_file with a local path. To save a file the page offers, click_download.
- Logged-in sites: pick your Chrome profile at startup (or --cdp) so cookies already exist.

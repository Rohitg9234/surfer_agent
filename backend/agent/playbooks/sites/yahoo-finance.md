---
name: yahoo-finance
description: Analyst ratings and quote pages on Yahoo Finance.
platform: Yahoo Finance
domains:
  - finance.yahoo.com
---

# Yahoo Finance

- Quote: `https://finance.yahoo.com/quote/<TICKER>`
- Analysts: `.../quote/<TICKER>/analysts` (rating, price target)
- Key statistics: `.../quote/<TICKER>/key-statistics`
- If you only have a company name, search the web for the ticker first.
- Analyst page is JS-heavy: wait_for_text for "Price Target" or similar, then read_element / note.
- Keep the rating label as shown ("Strong Buy"), not a guessed number.

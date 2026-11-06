---
name: e-commerce
description: Product listings, prices, variants, pagination and load-more.
---

# E-commerce

- Prefer on-page price/name text; save with note.
- Pagination: next, page numbers, ?page=N, ?offset=N, "showing X of Y". If the page says 200 products and you have 24, keep going.
- Infinite scroll: scroll down, search_page, repeat.
- Variants (size/color): open the dropdown with dropdown_options, then select_option.
- If HTML is a shell, look for a products JSON/API link on the snapshot; otherwise keep using refs.
- Bot-check / 404 on a listing: do not hammer reload; try another URL or done.

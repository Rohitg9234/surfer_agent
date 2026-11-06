---
name: structured-extraction
description: Pull facts into notes and a final answer without inventing data.
---

# Structured extraction

- Only report values seen on the page, in notes, or in tool output. If missing, say so.
- note each fact (price, name, URL) as you go.
- Numbers stay numbers in the final done answer (10.99 not "$10.99" unless quoting the page).
- For lists, keep extracting until pagination/scroll is exhausted or the user limit is hit.
- Call done with the complete answer on the last sub-goal, not subgoal_complete.

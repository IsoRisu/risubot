# Repo workflow (operator notes)

`/repo` (pull requests against allowlisted repos) and `!code` (sandbox repo) use **Gemini only**.
No local model, no other provider. The bot, not the model, reads files, applies edits and talks to git.

## Models and steps

| Step | Model | Notes |
|---|---|---|
| Plan: file tree + README/AGENTS.md + request, model picks up to 12 files | `gemini-3.8-flash` | JSON reply |
| Edit: files + plan, model returns edits | `gemini-3.8-flash` | max 4000 output tokens; `"continue": true` asks for another round |
| Edit fallback | `gemini-3.6-flash` | only when the edit model has no free lane at all; kept for the rest of that task |
| Repair: failed edits (bad anchor, syntax/JSON error) | `gemini-3.5-flash-lite` | at most one repair round per task |

Edits are applied by plain code, checked (`.py` compiles, `.json` parses), then committed to a bot branch and a PR is opened.
A task is all-or-nothing: if it fails, nothing is committed.

## Lanes

A lane is one Google account with one or more keys. Add one in `botkey.py`:

```python
GEMINI_LANES = [
    ["AIza...a1", "AIza...a2"],   # acct1
    ["AIza...b1"],                # acct2  <- new lane
]
```

Restart the bot. Rules: use keys from different accounts/projects (keys of one account share its quota),
and never put one key in two lanes. For each call the bot picks the lane with the most requests left today
for that model, and keeps each lane under its per-minute limit. Every call is logged as
`lane=<id> model=<m> status=<s>`. Without `GEMINI_LANES`, each key of `GEMINI_API_KEY` becomes its own lane.

## Daily counters

* Counted per (lane, model) in `gemini_quota.json` next to the bot, so a restart does not reset them.
* They reset at 00:00 America/Los_Angeles (Google's day), not at your local midnight.
* The per-minute window is kept in memory only, so right after a restart a lane may burst once.
* Limits are configuration, not code. Defaults: RPD 20 / 20 / 500 and RPM 5 / 5 / 15 for 3.8-flash / 3.6-flash / 3.5-flash-lite.
  Override with `REPO_LIMITS = {"rpd": {...}, "rpm": {...}}` in `botkey.py`.
* Google's real limits can differ from the configured ones; if Google answers 429 early, that lane is paused for the time Google names.

## When every lane is exhausted

The task fails with the time to retry (earliest lane or Pacific midnight) and the command to repeat. Nothing is committed. Options:

1. Wait for the time shown (daily limits return at Pacific midnight).
2. Add a lane (new Google account key) and restart.
3. If your accounts really have higher limits, raise them in `REPO_LIMITS`.

Plan and edit are 2 calls per task (plus continuations or repair), so 20 requests a day per lane is roughly 8-10 tasks.

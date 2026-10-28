# botkey.example.py -- copy to botkey.py (which is gitignored) and fill in your own values.
#
# Gemini-only repo workflow: the /repo and !code commands call only Gemini, through "lanes".
# A lane is one Google account (one quota pool) with one or more API keys. See README_REPO.md.
# llmclient.py (used by !ask) still understands the optional provider keys at the bottom.

# --- Discord ---------------------------------------------------------------
bot_key = "YOUR_DISCORD_BOT_TOKEN"
ALLOWED_GUILD_ID = 123456789012345678          # int or list of ints

# --- Spotify (optional) ----------------------------------------------------
spotify_client_id = "YOUR_SPOTIFY_CLIENT_ID"
spotify_client_secret = "YOUR_SPOTIFY_CLIENT_SECRET"
spotify_redirect_uri = "http://127.0.0.1:8888/callback"

# --- Gemini lanes for the repo workflow --------------------------------------
# One inner list per GOOGLE ACCOUNT. Keys inside one list share that account's quota, so the lane
# is the unit of rotation. Create keys of different accounts/projects to get more daily requests.
# The same key must not appear in two lanes. Lane ids are acct1, acct2, ... in this order.
GEMINI_LANES = [
    ["AIza...account1-key1", "AIza...account1-key2"],   # lane acct1
    ["AIza...account2-key1"],                           # lane acct2
]
# Fallback when GEMINI_LANES is not set: GEMINI_API_KEY = ["AIza...", "AIza..."] (one key = one lane).
# !ask also reads GEMINI_API_KEY (a string or a list of keys).
GEMINI_API_KEY = ["AIza...account1-key1", "AIza...account2-key1"]

# Optional: per-model limits, merged over the built-in defaults
#   RPD (requests per lane per day, reset 00:00 America/Los_Angeles):
#       gemini-3.8-flash 20, gemini-3.6-flash 20, gemini-3.5-flash-lite 500
#   RPM (requests per lane per sliding 60 s): 5, 5, 15
# Set these to the limits your accounts really have (they differ between free and paid tiers).
REPO_LIMITS = {
    "rpd": {"gemini-3.8-flash": 20, "gemini-3.6-flash": 20, "gemini-3.5-flash-lite": 500},
    "rpm": {"gemini-3.8-flash": 5, "gemini-3.6-flash": 5, "gemini-3.5-flash-lite": 15},
}

# --- GitHub / repo commands -----------------------------------------------------
GITHUB_TOKEN = "github_pat_..."                 # fine-grained PAT: Contents RW, Pull requests RW, Metadata R
REPO_OWNER_IDS = [123456789012345678]           # Discord user ids allowed to use the repo commands
REPO_ALLOWED_GUILD_IDS = [123456789012345678]   # falls back to ALLOWED_GUILD_ID
SANDBOX_REPO = "me/bot-sandbox"                 # private scratch repo, overwritten by every !code run
# SANDBOX_BRANCH = "main"
REPO_ALLOWLIST = {                              # repos the bot may open PRs against -> allowed base branches
    "me/app": ["main", "dev"],
}
# SANDBOX_OWNER_ONLY = True
# ALLOW_WORKFLOW_EDITS = False
# BOT_BRANCH_PREFIX = "ai/"

# --- Optional: other providers, used by !ask only (never by the repo workflow) ----------
# GROQ_API_KEY = "gsk_..."
# MISTRAL_API_KEY = "..."
# NVIDIA_API_KEY = "nvapi-..."
# OPENROUTER_API_KEY = "sk-or-..."
# PROVIDER_ORDER = ["gemini", "groq"]
# ROTATE_PROVIDERS = True

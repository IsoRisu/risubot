# Setting Up the Bot

Follow these steps to set up and run the bot:

## 1. Install Dependencies

Install Poetry (if you don't have it already):

```bash
pip install poetry
```

Then install the project's dependencies:

```bash
poetry install
```

If you're picking this project back up after a while, it's worth updating
dependencies too, since yt-dlp especially needs frequent updates to keep
working with YouTube:

```bash
poetry update
```

## 2. Configure `botkey.py`

Create a `botkey.py` file in the project root. It must define all three
of the following variables — the bot won't start (or AI features won't
work) without them:

```python
# Your Discord bot token (from the Discord Developer Portal)
bot_key = "YOUR_DISCORD_BOT_TOKEN"

# Your Google Gemini API key (https://aistudio.google.com/app/apikey)
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

# The Discord server (guild) ID where AI commands are allowed.
# Right-click your server in Discord (with Developer Mode enabled)
# and choose "Copy Server ID".
ALLOWED_GUILD_ID = 123456789012345678
```

Notes:

- `bot_key` is lowercase; `GEMINI_API_KEY` and `ALLOWED_GUILD_ID` are
  uppercase. Match the names exactly as used in the code.
- `ALLOWED_GUILD_ID` must be an `int`, not a string.
- This file is gitignored and won't be included if you clone the repo,
  so every fresh clone needs its own copy.

## 3. Run the App

Once dependencies are installed, run the bot:

```bash
poetry run python risubot.py
```

Your bot should now be up and running!

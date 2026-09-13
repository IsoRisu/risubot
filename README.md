
# Setting Up the Bot

Follow these steps to set up and run the bot:

## 3. Install Dependencies

Install Poetry (if you don't have it already):

\`\`\`bash
pip install poetry
\`\`\`

Then install the project's dependencies:

\`\`\`bash
poetry install
\`\`\`

If you're picking this project back up after a while, it's worth updating
dependencies too, since yt-dlp especially needs frequent updates to keep
working with YouTube:

\`\`\`bash
poetry update
\`\`\`

## 4. Set Up Your Bot Token

Create a `botkey.py` file in the project root with:

\`\`\`python
bot_key = "YOUR_DISCORD_BOT_TOKEN"
\`\`\`

This file is gitignored and won't be included if you clone the repo.

## 5. Run the App

Once dependencies are installed, run the bot:

\`\`\`bash
poetry run python risubot.py
\`\`\`

Your bot should now be up and running!

---


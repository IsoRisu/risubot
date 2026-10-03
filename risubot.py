import discord
from discord.ext import commands
import yt_dlp
import asyncio
import botkey  # Assuming botkey.py contains: bot_key = "YOUR_TOKEN"
from collections import deque
import llmclient
import latexrenderer
import graphrenderer
import io  # for discord.File
import mimetypes
import re
import shlex
import traceback
import os
import sys
import json
import threading
import requests
# SPOTIFY FUNCTIONALITY
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Bot Configuration ---
TOKEN = botkey.bot_key
PREFIX = '!'

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True

# Never let model output (or song titles) ping @everyone / roles / users.
# Only the replied-to author is pinged, as with a normal reply.
bot = commands.Bot(
    command_prefix=PREFIX,
    intents=intents,
    allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False, replied_user=True),
)

# Per-guild queues: {guild_id: deque([...])}
# A single global queue breaks as soon as the bot is in more than one server,
# since songs from different guilds would interleave and play in the wrong place.
music_queues: dict[int, deque] = {}

def get_queue(guild_id: int) -> deque:
    if guild_id not in music_queues:
        music_queues[guild_id] = deque()
    return music_queues[guild_id]

# --- YouTube-DL Options ---
YDL_OPTS = {
    'format': 'bestaudio/best',
    'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
    'restrictfilenames': True,
    'noplaylist': True,
    'nocheckcertificate': True,
    'ignoreerrors': False,
    'logtostderr': False,
    'verbose': False,       # was True — floods the console in normal operation
    'no_warnings': True,
    'default_search': 'auto',
    'source_address': '0.0.0.0',
    'geo_bypass': True,
}

# Base FFmpeg options; 'before_options' gets extended per-song with the
# correct HTTP headers below, since YouTube now frequently rejects stream
# requests that don't carry the same headers yt-dlp used to resolve the URL.
BASE_FFMPEG_OPTS = {
    'options': '-vn',
}


# --- Helper Function to Get Audio Source ---
async def get_audio_source(query_or_url):
    """
    Runs the blocking yt-dlp extraction in a thread executor so it doesn't
    freeze the bot's event loop (and therefore every other server/command)
    while resolving a song.
    """
    loop = asyncio.get_running_loop()

    def _extract():
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            try:
                return ydl.extract_info(query_or_url, download=False)
            except yt_dlp.utils.YoutubeDLError as e:
                print(f"Error extracting info with yt-dlp: {e}")
                return None

    info = await loop.run_in_executor(None, _extract)
    if info is None:
        return None, None, None

    if 'entries' in info:  # Search result or playlist
        entries = [e for e in info['entries'] if e]
        if not entries:
            return None, None, None
        info = entries[0]

    audio_url = info.get('url')
    title = info.get('title', 'Unknown Title')
    http_headers = info.get('http_headers', {})

    # Fallback for cases where 'url' isn't the direct audio stream
    if not audio_url:
        formats = info.get('formats', [])
        for f in formats:
            if f.get('acodec') != 'none' and f.get('vcodec') == 'none' and f.get('url'):
                audio_url = f['url']
                http_headers = f.get('http_headers', http_headers)
                break
        if not audio_url and formats:
            audio_url = formats[0].get('url')
            http_headers = formats[0].get('http_headers', http_headers)

    return audio_url, title, http_headers


def build_ffmpeg_opts(http_headers: dict) -> dict:
    """Attach a reconnect strategy and, critically, the original request
    headers so the CDN doesn't reject the stream mid-playback."""
    header_str = ''.join(f'{k}: {v}\r\n' for k, v in (http_headers or {}).items())
    before_options = '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5'
    if header_str:
        before_options += f' -headers {shlex.quote(header_str)}'
    return {
        'before_options': before_options,
        'options': BASE_FFMPEG_OPTS['options'],
    }


# --- Core Music Playback Logic ---
def _schedule(coro):
    """Thread-safe: run a coroutine on the bot loop from the player thread,
    logging (rather than silently swallowing) any exception it raises."""
    future = asyncio.run_coroutine_threadsafe(coro, bot.loop)

    def _done(f):
        try:
            f.result()
        except Exception as e:
            print(f"Scheduled coroutine failed: {type(e).__name__}: {e}")

    future.add_done_callback(_done)


async def play_next(ctx_param):
    guild_id = ctx_param.guild.id
    queue = get_queue(guild_id)

    if not queue:
        vc = discord.utils.get(bot.voice_clients, guild=ctx_param.guild)
        if vc and vc.is_connected():
            await ctx_param.send("Queue finished.")
        print(f"Music queue is empty for guild {guild_id}. Playback stopped.")
        return

    song_item = queue.popleft()
    song_url = song_item['url']
    song_title = song_item['title']
    song_headers = song_item.get('headers', {})
    original_ctx = song_item['ctx']

    vc = discord.utils.get(bot.voice_clients, guild=original_ctx.guild)

    if not vc or not vc.is_connected():
        print(f"Bot not connected in guild {guild_id} for song {song_title}. Clearing queue.")
        queue.clear()
        return

    if vc.is_playing() or vc.is_paused():
        print(f"play_next called for '{song_title}' while VC is already playing/paused. Re-queueing song.")
        queue.appendleft(song_item)
        return

    # Lazily-resolved entries (e.g. from !spotify) only carry a search query;
    # look up the YouTube stream now so the URL is fresh.
    if not song_url:
        try:
            song_url, resolved_title, song_headers = await get_audio_source(song_item['query'])
        except Exception as e:
            print(f"Lazy lookup failed for '{song_item['query']}': {e}")
            song_url = None
        if not song_url:
            await original_ctx.send(f"Couldn't find a YouTube match for **{song_item['title']}**, skipping.")
            await play_next(original_ctx)
            return
        song_title = resolved_title or song_title
        song_headers = song_headers or {}
        # The lookup awaited, so something else may have started playback or the bot may have left.
        vc = discord.utils.get(bot.voice_clients, guild=original_ctx.guild)
        if not vc or not vc.is_connected():
            queue.clear()
            return
        if vc.is_playing() or vc.is_paused():
            song_item.update(url=song_url, title=song_title, headers=song_headers)
            queue.appendleft(song_item)
            return

    try:
        ffmpeg_opts = build_ffmpeg_opts(song_headers)
        source = discord.FFmpegPCMAudio(song_url, **ffmpeg_opts)

        def after_playing_song_callback(error):
            if error:
                print(f'Player error in guild {guild_id} for song "{song_title}": {error}')
                _schedule(original_ctx.send(f"Playback error for '{song_title}': {error}"))
            # This runs on discord.py's player thread. Schedule and return;
            # blocking on the result would stall the thread.
            _schedule(play_next(original_ctx))

        vc.play(source, after=after_playing_song_callback)
        await original_ctx.send(f'Now playing: **{song_title}**')
    except Exception as e:
        await original_ctx.send(f"An error occurred before playing '{song_title}': {e}")
        print(f"Error in play_next setup for '{song_title}' in guild {guild_id}: {e}")
        await play_next(original_ctx)


# --- Bot Events ---
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    print(f'Bot prefix is: {PREFIX}')
    print('Bot is ready to play music!')
    print('------')


def chunk_text(text: str, limit: int = 1900) -> list:
    """Split text for Discord's 2000-char cap, preferring line breaks. If a cut
    lands inside a ``` code fence, close it and reopen it in the next chunk so
    the formatting doesn't break."""
    chunks, open_fence = [], None
    while text:
        prefix = open_fence + "\n" if open_fence else ""
        room = limit - len(prefix) - 4  # leave space for a closing "\n```"
        if len(text) <= room:
            chunks.append(prefix + text)
            break
        cut = text.rfind("\n", 0, room)
        if cut <= 0:
            cut = room
        body, text = prefix + text[:cut], text[cut:].lstrip("\n")
        state = None
        for line in body.splitlines():
            if line.startswith("```"):
                state = None if state else line.strip()
        if state:
            body += "\n```"
        open_fence = state
        chunks.append(body)
    return chunks


# --- Bot Commands ---
@commands.guild_only()
@bot.command(name='join', help='Tells the bot to join the voice channel you are in.')
async def join(ctx):
    if not ctx.author.voice:
        await ctx.send(f"{ctx.author.name} is not connected to a voice channel.")
        return

    channel = ctx.author.voice.channel
    if ctx.voice_client is not None:
        await ctx.voice_client.move_to(channel)
    else:
        await channel.connect()
    await ctx.send(f"Joined **{channel}**")


@commands.guild_only()
@bot.command(name='leave', aliases=['dc'], help='Tells the bot to leave the voice channel.')
async def leave(ctx):
    if ctx.voice_client is not None:
        get_queue(ctx.guild.id).clear()
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel and cleared the queue.")
    else:
        await ctx.send("I'm not in a voice channel.")


@commands.guild_only()
@bot.command(name='play', aliases=['p'], help='Plays a song or adds to queue. Usage: !play <URL or search query>')
async def play(ctx, *, query: str):
    if not ctx.author.voice:
        await ctx.send("You need to be in a voice channel to play music.")
        return

    channel = ctx.author.voice.channel
    vc = ctx.voice_client

    if vc is None:
        try:
            vc = await channel.connect()
            await ctx.send(f"Joined **{channel}**")
        except Exception as e:
            await ctx.send(f"Failed to join your voice channel: {e}")
            return
    elif vc.channel != channel:
        await ctx.send(f"You need to be in the same voice channel as me. I am in **{vc.channel}**.")
        return

    async with ctx.typing():
        audio_url, video_title, http_headers = await get_audio_source(query)

        if audio_url is None or video_title is None:
            await ctx.send(f"Could not find a playable audio source for '{query}'.")
            return

        queue = get_queue(ctx.guild.id)
        song_info = {'url': audio_url, 'title': video_title, 'headers': http_headers, 'ctx': ctx}
        queue.append(song_info)
        await ctx.send(f"Added to queue: **{video_title}** (Position: {len(queue)})")

    if not vc.is_playing() and not vc.is_paused():
        await play_next(ctx)


@commands.guild_only()
@bot.command(name='stop', help='Stops the music and clears the queue.')
async def stop(ctx):
    if ctx.voice_client:
        get_queue(ctx.guild.id).clear()
        if ctx.voice_client.is_playing() or ctx.voice_client.is_paused():
            ctx.voice_client.stop()
            await ctx.send("Music stopped and queue cleared.")
        else:
            await ctx.send("Nothing was playing, but queue has been cleared.")
    else:
        await ctx.send("I'm not in a voice channel.")
        get_queue(ctx.guild.id).clear()


@commands.guild_only()
@bot.command(name='skip', aliases=['s'], help='Skips the current song.')
async def skip(ctx):
    if ctx.voice_client and (ctx.voice_client.is_playing() or ctx.voice_client.is_paused()):
        await ctx.send("Skipping song...")
        ctx.voice_client.stop()
    else:
        await ctx.send("Nothing is currently playing to skip.")


@commands.guild_only()
@bot.command(name='pause', help='Pauses the current song.')
async def pause(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.pause()
        await ctx.send("Music paused.")
    elif ctx.voice_client and ctx.voice_client.is_paused():
        await ctx.send("Music is already paused.")
    else:
        await ctx.send("Nothing is currently playing to pause.")


@commands.guild_only()
@bot.command(name='resume', help='Resumes the paused song.')
async def resume(ctx):
    if ctx.voice_client and ctx.voice_client.is_paused():
        ctx.voice_client.resume()
        await ctx.send("Music resumed.")
    elif ctx.voice_client and ctx.voice_client.is_playing():
        await ctx.send("Music is already playing.")
    else:
        await ctx.send("Nothing to resume.")


@commands.guild_only()
@bot.command(name='queue', aliases=['q'], help='Displays the current music queue.')
async def queue_command(ctx):
    queue = get_queue(ctx.guild.id)
    if not queue:
        await ctx.send("The music queue is empty.")
        return

    lines = ["**🎶 Current Music Queue:**"]
    for i, song in enumerate(list(queue)):
        if i >= 15:
            lines.append(f"... and {len(queue) - i} more.")
            break
        lines.append(f"{i + 1}. {song['title']}")

    message = "\n".join(lines)
    # Discord hard-caps messages at 2000 characters
    if len(message) > 2000:
        message = message[:1990] + "\n..."
    await ctx.send(message)


# --- Error Handling for Commands ---
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Invalid command. Try `!help` to see available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing arguments for command `{ctx.command}`. Check `!help {ctx.command}` for usage.")
    elif isinstance(error, commands.CommandInvokeError):
        original_error = getattr(error, 'original', error)
        await ctx.send(f"An error occurred with the `{ctx.command}` command. Please check the console for details.")
        print(f"CommandInvokeError in command {ctx.command}: {original_error}")
        traceback.print_exception(type(original_error), original_error, original_error.__traceback__)
    elif isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"Slow down, try again in {error.retry_after:.0f}s.")
    elif isinstance(error, commands.CheckFailure):
        await ctx.send(f"You do not have the necessary permissions or conditions to run `{ctx.command}`.")
    else:
        await ctx.send(f"An unexpected error occurred: {error}")
        print(f"Unexpected error: {error}")

# Discord's own upload cap is higher, but Gemini's inline-data request
# limit is ~20 MB total, so leave headroom for base64 overhead (+33%).
MAX_ATTACHMENT_BYTES = 14 * 1024 * 1024


async def collect_attachments(message) -> tuple[list, list]:
    """Pull attachments off a message and, if it's a reply, off the message
    it replies to. Returns (attachments, skipped_filenames)."""
    sources = list(message.attachments)

    ref = message.reference
    if ref is not None:
        replied = ref.resolved
        if replied is None and ref.message_id:
            try:
                replied = await message.channel.fetch_message(ref.message_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                replied = None
        if isinstance(replied, discord.Message):
            sources.extend(replied.attachments)

    out, skipped, total = [], [], 0
    for a in sources:
        ctype = (a.content_type or "").split(";")[0]
        if not ctype:
            ctype = mimetypes.guess_type(a.filename)[0] or ""
        if not ctype.startswith(llmclient.SUPPORTED_PREFIXES):
            skipped.append(a.filename)
            continue
        if total + a.size > MAX_ATTACHMENT_BYTES:
            skipped.append(a.filename)
            continue
        data = await a.read()
        out.append(llmclient.Attachment(data, ctype, a.filename))
        total += a.size

    return out, skipped


@bot.command(name="ask")
async def ask_command(ctx, *, prompt: str = ""):
    """Usage: !ask <question>, optionally with an image/audio/video/PDF
    attached, or as a reply to a message that has one."""

    if ctx.guild is None or ctx.guild.id != getattr(botkey, "ALLOWED_GUILD_ID", None):
        await ctx.reply("This command isn't available here.")
        return

    async with ctx.typing():
        try:
            attachments, skipped = await collect_attachments(ctx.message)
        except discord.HTTPException as e:
            await ctx.reply(f"Couldn't download that attachment: `{e}`"[:1900])
            return

        if not prompt.strip() and not attachments:
            await ctx.reply("Give me something to work with — a question, or an image.")
            return

        if skipped:
            await ctx.send(f"Skipping (unsupported or too large): {', '.join(skipped)}"[:1900])

        try:
            reply = await asyncio.to_thread(llmclient.ask, prompt, attachments)
        except Exception as e:
            # Error bodies can be long; Discord rejects messages over 2000 chars.
            await ctx.reply(f"AI error: `{type(e).__name__}: {e}`"[:1900])
            return

        if not reply:
            await ctx.reply("(empty response)")
            return

        # Render off the event loop; matplotlib is CPU-bound and blocking.
        if latexrenderer.contains_latex(reply):
            try:
                segments = await asyncio.to_thread(latexrenderer.split_and_render, reply)
            except Exception as e:
                print(f"LaTeX rendering failed: {type(e).__name__}: {e}")
                segments = [("text", reply)]
        else:
            segments = [("text", reply)]

    # Merge consecutive text segments so we don't post a message per word.
    merged = []
    for kind, val in segments:
        if kind == "text" and merged and merged[-1][0] == "text":
            merged[-1] = ("text", merged[-1][1] + val)
        else:
            merged.append((kind, val))

    first_message = True

    async def send(**kwargs):
        nonlocal first_message
        if first_message:
            await ctx.reply(**kwargs)
            first_message = False
        else:
            await ctx.send(**kwargs)

    async def send_text(text: str):
        # Discord caps messages at 2000 chars; leave headroom.
        for chunk in chunk_text(text):
            if chunk.strip():  # Discord rejects empty / whitespace-only messages
                await send(content=chunk)

    # Consecutive equations go out together (Discord allows 10 files per message)
    # instead of one message each. Whitespace between them is dropped.
    pending = []

    async def flush_images():
        nonlocal pending
        for start in range(0, len(pending), 10):
            files = [
                discord.File(io.BytesIO(png), filename=f"math_{start + k}.png")
                for k, png in enumerate(pending[start:start + 10])
            ]
            await send(files=files)
        pending = []

    for kind, val in merged:
        if kind == "image":
            pending.append(val)
        elif val.strip():
            await flush_images()
            await send_text(val)
    await flush_images()


PLOT_USAGE = (
    "Usage: `!plot <function, equation or inequality>[; <next> ...] [from <a> to <b>]`\n"
    "Functions: `!plot x^2 - 2x + 1`, `!plot sin(x); cos(x) from -6.3 to 6.3`, `!plot ln(x) from 0.1 to 10`\n"
    "Curves: `!plot x²+y²=1`, `!plot x^2/4 + y^2 = 1`, `!plot x*y = 1`\n"
    "Regions (shaded; dashed edge for `<` `>`, solid for `<=` `>=`): `!plot x²+y²<1`, `!plot y > x^2`, "
    "`!plot x + y <= 2; x > 0`\n"
    "Available: sin cos tan asin acos atan sinh cosh tanh exp sqrt cbrt abs sign floor ceil ln log10 log2, "
    "constants pi and e, variables `x` and `y`. Up to 6 graphs, separated by `;`. "
    "The range applies to both axes; leave it out for curves and regions and the view fits automatically."
)


@bot.command(name="plot", aliases=["graph", "kuvaaja"],
             help="Draws graphs of functions and curves. Usage: !plot sin(x); x^2 from -5 to 5  or  !plot x^2+y^2<1")
@commands.cooldown(1, 5, commands.BucketType.user)
async def plot_command(ctx, *, spec: str):
    async with ctx.typing():
        try:
            png = await asyncio.to_thread(latexrenderer.render_plot_spec, spec)
        except Exception as e:
            await ctx.reply(f"Couldn't plot that: {e}\n{PLOT_USAGE}"[:1900])
            return
    await ctx.reply(file=discord.File(io.BytesIO(png), filename="plot.png"))


DIAGRAM_USAGE = (
    "Usage: `!diagram <description>`. Statements are separated by `;` or new lines.\n"
    "Graph: `!diagram A -- B; B -- C : 5; C -> A; highlight: A`\n"
    "DFA/NFA (`!dfa`, `!nfa`): `!dfa start: q0; accept: q1; q0 -0-> q0; q0 -1-> q1; q1 -0,1-> q1`\n"
    "Turing machine (`!tm`): `!tm start: q0; accept: qa; q0 -1/0,R-> q0; q0 -_/_,L-> qa; tape: 1 1 _; head: 0; state: q0`\n"
    "Tree (`!tree`): `!tree root -> a; root -> b; a -> c`\n"
    "Edges: `A -> B` directed, `A -- B` undirected, `A <-> B` both, `: 5` or `-label->` for labels. "
    "Also `start:`, `accept:`, `nodes:`, `highlight:`, `title:`, `layout: circle|spring|lr|tb|tree`, `tape:`, `head:`, `state:`. "
    "Write epsilon as `eps`. Up to 30 nodes and 100 edges."
)

_DIAGRAM_KINDS = {"diagram": "graph", "verkko": "graph", "puu": "tree", "automaatti": "automaton"}


@bot.command(name="diagram", aliases=["verkko", "automaton", "automaatti", "dfa", "nfa", "fsm", "tm", "turing", "pda", "tree", "puu"],
             help="Draws graphs, automata, Turing machines and trees. Usage: !dfa start: q0; accept: q1; q0 -a-> q1  (see !diagram for the full syntax)")
@commands.cooldown(1, 5, commands.BucketType.user)
async def diagram_command(ctx, *, spec: str):
    kind = _DIAGRAM_KINDS.get(ctx.invoked_with.lower(), ctx.invoked_with.lower())
    async with ctx.typing():
        try:
            png = await asyncio.to_thread(graphrenderer.render_graph_spec, spec, kind)
        except Exception as e:
            await ctx.reply(f"Couldn't draw that: {e}\n{DIAGRAM_USAGE}"[:1900])
            return
    await ctx.reply(file=discord.File(io.BytesIO(png), filename="diagram.png"))

# --- Spotify Setup ---
# Spotify's Feb 2026 Web API changes (Development Mode apps) matter here:
#   * playlist contents are only returned for playlists the *logged-in* account owns or collaborates on
#   * GET /playlists/{id}/tracks was renamed to /items and track -> item in responses
#   * GET /artists/{id}/top-tracks was removed
#   * search limit max is 10
SPOTIFY_SCOPE = "playlist-read-private playlist-read-collaborative"
SPOTIFY_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spotify_token_cache")
MAX_SPOTIFY_TRACKS = 300          # safety cap so one link can't flood the queue
# If Spotify's API refuses a playlist, try the public embed page as a last resort
# (public playlists only, ~100 tracks max, unofficial, may break at any time).
SPOTIFY_EMBED_FALLBACK = True

_spotify_client = None
_spotify_lock = threading.Lock()   # get_spotify_client() runs in worker threads, so a threading lock, not asyncio


def _make_spotify_auth(open_browser: bool = False):
    cid = getattr(botkey, "spotify_client_id", None)
    secret = getattr(botkey, "spotify_client_secret", None)
    redirect_uri = getattr(botkey, "spotify_redirect_uri", "http://127.0.0.1:8888/callback")
    if not cid or not secret:
        return None
    return SpotifyOAuth(
        client_id=cid,
        client_secret=secret,
        redirect_uri=redirect_uri,
        scope=SPOTIFY_SCOPE,
        cache_path=SPOTIFY_CACHE_PATH,   # absolute path: doesn't depend on the working directory
        open_browser=open_browser,
    )


def get_spotify_client():
    """Return a cached Spotify client, or None if credentials are missing.
    Never starts an interactive login (that would hang the bot inside a worker
    thread); raises RuntimeError telling you to run the one-time login instead."""
    global _spotify_client
    with _spotify_lock:
        if _spotify_client is not None:
            return _spotify_client

        auth = _make_spotify_auth()
        if auth is None:
            return None

        # validate_token() refreshes expired tokens and returns None when there is no
        # token or the cached one was issued with different scopes.
        if not auth.validate_token(auth.cache_handler.get_cached_token()):
            raise RuntimeError(
                "Spotify isn't authorized yet. Stop the bot and run "
                "`python risubot.py --spotify-auth` once, log in with the Spotify account "
                "that owns/collaborates on the playlists, then start the bot again."
            )

        _spotify_client = spotipy.Spotify(auth_manager=auth, requests_timeout=10, retries=3)
        return _spotify_client


def run_spotify_auth():
    """One-time interactive login (python risubot.py --spotify-auth)."""
    auth = _make_spotify_auth(open_browser=True)
    if auth is None:
        print("Add spotify_client_id and spotify_client_secret to botkey.py first.")
        return
    sp = spotipy.Spotify(auth_manager=auth, requests_timeout=10)
    me = sp.current_user()
    print(f"Authorized as: {me.get('display_name') or me['id']} (id: {me['id']})")
    print(f"Token cached at {SPOTIFY_CACHE_PATH}")


SPOTIFY_URL_RE = re.compile(
    r"(?:open\.spotify\.com/(?:intl-[a-z-]+/)?(?:user/[^/\s]+/)?|spotify:)"
    r"(playlist|album|track|artist)[/:]([A-Za-z0-9]+)"
)


def parse_spotify_url(url: str):
    """Return (kind, id) or (None, None). Accepts web links and spotify: URIs."""
    m = SPOTIFY_URL_RE.search(url)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _track_to_query(t: dict) -> str:
    name = t.get("name", "")
    artists = ", ".join(a["name"] for a in (t.get("artists") or []) if a.get("name"))
    return f"{artists} - {name}".strip(" -")


def _fetch_playlist_via_embed(playlist_id: str) -> list[str]:
    """Best-effort, unofficial: read the track list from the public embed page."""
    try:
        r = requests.get(
            f"https://open.spotify.com/embed/playlist/{playlist_id}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        r.raise_for_status()
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', r.text, re.S)
        if not m:
            return []
        entity = json.loads(m.group(1))["props"]["pageProps"]["state"]["data"]["entity"]
        out = []
        for t in entity.get("trackList", []):
            title = (t.get("title") or "").strip()
            artist = (t.get("subtitle") or "").replace("\xa0", " ").strip()
            if title:
                out.append(f"{artist} - {title}".strip(" -"))
        return out
    except Exception as e:
        print(f"[spotify] embed fallback failed: {type(e).__name__}: {e}")
        return []


def _fetch_playlist_tracks(sp, playlist_id: str) -> list[str]:
    tracks = []
    try:
        # Raw call: older spotipy versions' playlist_items() still hit the removed
        # /tracks endpoint, so go to /items directly.
        results = sp._get(f"playlists/{playlist_id}/items", limit=100, additional_types="track")
        while results:
            for entry in results.get("items", []):
                t = entry.get("item") or entry.get("track")   # 'item' since Feb 2026
                if not t or not t.get("name") or t.get("type") == "episode":
                    continue
                tracks.append(_track_to_query(t))
            results = sp.next(results) if results.get("next") else None
        return tracks
    except spotipy.SpotifyException as e:
        if e.http_status not in (403, 404):
            raise

    # 403/404: Spotify won't hand this playlist's contents to the logged-in account.
    if SPOTIFY_EMBED_FALLBACK:
        tracks = _fetch_playlist_via_embed(playlist_id)
        if tracks:
            return tracks

    try:
        me = sp.current_user()
        who = f"**{me.get('display_name') or me['id']}** (`{me['id']}`)"
    except Exception:
        who = "an unknown account"
    raise RuntimeError(
        "Spotify refused to return this playlist's tracks (403/404). The bot is logged in as "
        f"{who}. Spotify only returns tracks for playlists the *logged-in* account owns or "
        "collaborates on, so check that:\n"
        "• that exact account accepted the collaborator invite (if it's a different account "
        "from your personal one, re-run `python risubot.py --spotify-auth` with the right one),\n"
        "• that account is listed under User Management in your Spotify developer dashboard "
        "(Development Mode apps only work for allow-listed users),\n"
        "• the app owner's Spotify Premium is active.\n"
        "Public playlists you don't collaborate on can't be read through the API."
    )


def _fetch_spotify_tracks(kind: str, spotify_id: str) -> list[str]:
    """Blocking call — run in a thread. Returns ['Artist - Title', ...]."""
    sp = get_spotify_client()
    if sp is None:
        raise RuntimeError(
            "Spotify credentials missing. Add spotify_client_id and "
            "spotify_client_secret to botkey.py"
        )

    tracks = []

    if kind == "track":
        t = sp.track(spotify_id)
        if t:
            tracks.append(_track_to_query(t))

    elif kind == "album":
        results = sp.album_tracks(spotify_id, limit=50)
        while results:
            tracks.extend(_track_to_query(t) for t in results.get("items", []) if t)
            results = sp.next(results) if results.get("next") else None

    elif kind == "playlist":
        tracks = _fetch_playlist_tracks(sp, spotify_id)

    elif kind == "artist":
        # artist_top_tracks was removed in Feb 2026. Approximate with a track search
        # (max 10 results), keeping only tracks that actually credit this artist.
        artist = sp.artist(spotify_id)
        name = artist["name"]
        try:
            res = sp.search(q=f'artist:"{name}"', type="track", limit=10)
            for t in res.get("tracks", {}).get("items", []):
                if any(a.get("id") == spotify_id for a in t.get("artists", [])):
                    tracks.append(_track_to_query(t))
        except spotipy.SpotifyException as e:
            print(f"[spotify] artist search failed: {e}")
        if not tracks:
            tracks.append(name)   # at least let YouTube pick something

    else:
        raise ValueError(f"Unsupported Spotify link type: {kind}")

    # De-duplicate (keep order) and cap
    seen, unique = set(), []
    for q in tracks:
        if q and q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    return unique[:MAX_SPOTIFY_TRACKS]


@commands.guild_only()
@bot.command(
    name="spotify",
    aliases=["sp"],
    help="Queues a Spotify playlist/album/track/artist by searching YouTube. "
         "Usage: !spotify <spotify URL>",
)
async def spotify_command(ctx, *, url: str = ""):
    if not ctx.author.voice:
        await ctx.send("You need to be in a voice channel to play music.")
        return

    kind, spotify_id = parse_spotify_url(url.strip())
    if not kind:
        await ctx.send(
            "That doesn't look like a Spotify link. "
            "Paste a playlist, album, track, or artist URL from open.spotify.com."
        )
        return

    # Resolve the Spotify track list first (blocking → thread), so we don't
    # join voice for a link that turns out to be unreadable.
    async with ctx.typing():
        try:
            track_queries = await asyncio.to_thread(_fetch_spotify_tracks, kind, spotify_id)
        except spotipy.SpotifyException as e:
            await ctx.send(f"Spotify error (HTTP {e.http_status}): {e.msg}"[:1900])
            return
        except Exception as e:
            await ctx.send(f"Spotify error: {e}"[:1900])
            return

    if not track_queries:
        await ctx.send("No playable tracks found in that Spotify link.")
        return

    # Ensure the bot is in the caller's voice channel
    channel = ctx.author.voice.channel if ctx.author.voice else None
    if channel is None:
        await ctx.send("You left the voice channel — cancelling.")
        return
    vc = ctx.voice_client
    if vc is None:
        try:
            vc = await channel.connect()
            await ctx.send(f"Joined **{channel}**")
        except Exception as e:
            await ctx.send(f"Failed to join your voice channel: {e}")
            return
    elif vc.channel != channel:
        await ctx.send(
            f"You need to be in the same voice channel as me. I am in **{vc.channel}**."
        )
        return

    # Queue lazily: each entry stores only the search query. The YouTube lookup
    # happens in play_next() right before the song plays. That means playback starts
    # immediately, and stream URLs (which expire after a few hours) are never stale.
    queue = get_queue(ctx.guild.id)
    for query in track_queries:
        queue.append({"url": None, "query": query, "title": query, "headers": {}, "ctx": ctx})

    capped = " (capped)" if len(track_queries) >= MAX_SPOTIFY_TRACKS else ""
    await ctx.send(f"Queued **{len(track_queries)}** track(s) from Spotify{capped}.")

    current_vc = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    if current_vc and current_vc.is_connected() and not current_vc.is_playing() and not current_vc.is_paused():
        await play_next(ctx)


# --- Run the Bot ---
if __name__ == "__main__":
    if "--spotify-auth" in sys.argv:
        run_spotify_auth()
        sys.exit(0)
    if TOKEN == 'YOUR_DISCORD_BOT_TOKEN' or not TOKEN:
        print("ERROR: Please replace 'YOUR_DISCORD_BOT_TOKEN' with your actual bot token in the script or botkey.py.")
    else:
        try:
            bot.run(TOKEN)
        except discord.errors.LoginFailure:
            print("ERROR: Failed to log in. Make sure your bot token is correct and valid.")
        except Exception as e:
            print(f"An error occurred while trying to run the bot: {e}")


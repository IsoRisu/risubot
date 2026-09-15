import discord
from discord.ext import commands
import yt_dlp
import asyncio
import botkey  # Assuming botkey.py contains: bot_key = "YOUR_TOKEN"
from collections import deque
import risuai  

# --- Bot Configuration ---
TOKEN = botkey.bot_key
PREFIX = '!'

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents)

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
            except yt_dlp.utils.DownloadError as e:
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
        before_options += f' -headers "{header_str}"'
    return {
        'before_options': before_options,
        'options': BASE_FFMPEG_OPTS['options'],
    }


# --- Core Music Playback Logic ---
async def play_next(ctx_param):
    guild_id = ctx_param.guild.id
    queue = get_queue(guild_id)

    if not queue:
        vc = discord.utils.get(bot.voice_clients, guild=ctx_param.guild)
        if vc and vc.is_connected():
            asyncio.run_coroutine_threadsafe(ctx_param.send("Queue finished."), bot.loop)
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

    try:
        ffmpeg_opts = build_ffmpeg_opts(song_headers)
        source = discord.FFmpegPCMAudio(song_url, **ffmpeg_opts)

        def after_playing_song_callback(error):
            if error:
                print(f'Player error in guild {guild_id} for song "{song_title}": {error}')
                asyncio.run_coroutine_threadsafe(
                    original_ctx.send(f"Playback error for '{song_title}': {error}"),
                    bot.loop
                )
            future = asyncio.run_coroutine_threadsafe(play_next(original_ctx), bot.loop)
            try:
                future.result(timeout=5)
            except asyncio.TimeoutError:
                print(f"play_next call from after_playing_song_callback (guild {guild_id}) timed out on future.result().")
            except Exception as e:
                print(f"Error running/scheduling play_next from after_playing_song_callback (guild {guild_id}): {e}")

        vc.play(source, after=after_playing_song_callback)
        await original_ctx.send(f'Now playing: **{song_title}**')
    except Exception as e:
        await original_ctx.send(f"An error occurred before playing '{song_title}': {e}")
        print(f"Error in play_next setup for '{song_title}' in guild {guild_id}: {e}")
        asyncio.run_coroutine_threadsafe(play_next(original_ctx), bot.loop)


# --- Bot Events ---
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    print(f'Bot prefix is: {PREFIX}')
    print('Bot is ready to play music!')
    print('------')


# --- Bot Commands ---
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


@bot.command(name='leave', aliases=['dc'], help='Tells the bot to leave the voice channel.')
async def leave(ctx):
    if ctx.voice_client is not None:
        get_queue(ctx.guild.id).clear()
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel and cleared the queue.")
    else:
        await ctx.send("I'm not in a voice channel.")


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


@bot.command(name='skip', aliases=['s'], help='Skips the current song.')
async def skip(ctx):
    if ctx.voice_client and (ctx.voice_client.is_playing() or ctx.voice_client.is_paused()):
        await ctx.send("Skipping song...")
        ctx.voice_client.stop()
    else:
        await ctx.send("Nothing is currently playing to skip.")


@bot.command(name='pause', help='Pauses the current song.')
async def pause(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.pause()
        await ctx.send("Music paused.")
    elif ctx.voice_client and ctx.voice_client.is_paused():
        await ctx.send("Music is already paused.")
    else:
        await ctx.send("Nothing is currently playing to pause.")


@bot.command(name='resume', help='Resumes the paused song.')
async def resume(ctx):
    if ctx.voice_client and ctx.voice_client.is_paused():
        ctx.voice_client.resume()
        await ctx.send("Music resumed.")
    elif ctx.voice_client and ctx.voice_client.is_playing():
        await ctx.send("Music is already playing.")
    else:
        await ctx.send("Nothing to resume.")


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
        import traceback
        traceback.print_exception(type(original_error), original_error, original_error.__traceback__)
    elif isinstance(error, commands.CheckFailure):
        await ctx.send(f"You do not have the necessary permissions or conditions to run `{ctx.command}`.")
    else:
        await ctx.send(f"An unexpected error occurred: {error}")
        print(f"Unexpected error: {error}")

@bot.command(name="ask")
async def ask_command(ctx, *, prompt: str):
    """Usage: !ask <your question>"""

    # Checks for the server whitelist
    if ctx.guild is None or ctx.guild.id != botkey.ALLOWED_GUILD_ID:
        await ctx.reply("This command isn't available here.")
        return  # Silently ignore in other servers/DMs
    
    if not prompt.strip():
        await ctx.reply("Give me something to work with, e.g. `!ask what is 2+2?`")
        return

    # Show a typing indicator while we wait on the API
    async with ctx.typing():
        try:
            reply = await asyncio.to_thread(risuai.ask, prompt)
        except Exception as e:
            await ctx.reply(f"AI error: `{type(e).__name__}: {e}`")
            return

    # Discord messages cap at 2000 chars; split if needed
    for i in range(0, len(reply), 1900):
        chunk = reply[i:i + 1900]
        if i == 0:
            await ctx.reply(chunk)
        else:
            await ctx.send(chunk)

# --- Run the Bot ---
if __name__ == "__main__":
    if TOKEN == 'YOUR_DISCORD_BOT_TOKEN' or not TOKEN:
        print("ERROR: Please replace 'YOUR_DISCORD_BOT_TOKEN' with your actual bot token in the script or botkey.py.")
    else:
        try:
            bot.run(TOKEN)
        except discord.errors.LoginFailure:
            print("ERROR: Failed to log in. Make sure your bot token is correct and valid.")
        except Exception as e:
            print(f"An error occurred while trying to run the bot: {e}")
import discord
from discord.ext import commands
import yt_dlp
import asyncio
import botkey # Assuming botkey.py contains: bot_key = "YOUR_TOKEN"
from collections import deque

# --- Bot Configuration ---
TOKEN = botkey.bot_key
PREFIX = '!'

# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.voice_states = True

bot = commands.Bot(command_prefix=PREFIX, intents=intents)

music_queue = deque() # Using deque for efficient appends and poplefts

# --- YouTube-DL Options ---
YDL_OPTS = {
    'format': 'bestaudio/best', # Audio format code
    'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
    'restrictfilenames': True, # Restrict filenames to only ASCII characters, and avoid "&" and spaces in filenames
    'noplaylist': True,        # Process only single videos, not whole playlists
    'nocheckcertificate': True,
    'ignoreerrors': False, # If a video fails to download, tool won’t stop, just logs the error and moves on
    'logtostderr': False,
    'verbose': True, # Set to True for detailed yt-dlp debugging, False for cleaner console
    'no_warnings': True,
    'default_search': 'auto',
    'source_address': '0.0.0.0'
}
# FFmpeg options
FFMPEG_OPTS = {
    'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5', # reconnect 1 attempts to reconnect in case stream is lost, -streamed 1 attepmts to reconnect if audio source is lost, delay is the max time between attempts
    'options': '-vn' # Disable video processing
}

# --- Helper Function to Get Audio Source ---
async def get_audio_source(query_or_url):
    with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
        try:
            info = ydl.extract_info(query_or_url, download=False)
        except yt_dlp.utils.DownloadError as e:
            print(f"Error extracting info with yt-dlp: {e}")
            return None, None

        if 'entries' in info: # If it's a search result or a (non-disabled) playlist
            info = info['entries'][0] # Take the first result

        audio_url = info.get('url')
        title = info.get('title', 'Unknown Title')

        # Fallback for some cases where 'url' might not be the direct stream
        if not audio_url:
            formats = info.get('formats', [])
            for f in formats:
                if f.get('acodec') != 'none' and f.get('vcodec') == 'none' and f.get('url'):
                    audio_url = f['url']
                    break
            if not audio_url and formats: # If still no specific audio, take first format URL
                 audio_url = formats[0].get('url')

    return audio_url, title

# --- Core Music Playback Logic ---
async def play_next(ctx_param): # ctx_param is the context that triggered this or the song's original context
    if not music_queue:
        # Optional: Send a "Queue finished" message
        voice_client = discord.utils.get(bot.voice_clients, guild=ctx_param.guild)
        if voice_client and voice_client.is_connected():
            asyncio.run_coroutine_threadsafe(ctx_param.send("Queue finished."), bot.loop)
        print("Music queue is empty. Playback stopped.")
        return

    song_item = music_queue.popleft()
    song_url = song_item['url']
    song_title = song_item['title']
    original_ctx = song_item['ctx'] # Context from when the song was added

    vc = discord.utils.get(bot.voice_clients, guild=original_ctx.guild)

    if not vc or not vc.is_connected():
        print(f"Bot not connected in guild {original_ctx.guild.id} for song {song_title}. Clearing queue.")
        # asyncio.run_coroutine_threadsafe(original_ctx.send("I'm not connected to a voice channel. Clearing remaining queue."), bot.loop)
        music_queue.clear()
        return

    # Defensive check, though 'after' should handle this primarily
    if vc.is_playing() or vc.is_paused():
        print(f"play_next called for '{song_title}' while VC is already playing/paused. Re-queueing song.")
        music_queue.appendleft(song_item) # Put it back at the front
        return

    try:
        source = discord.FFmpegPCMAudio(song_url, **FFMPEG_OPTS)

        def after_playing_song_callback(error):
            guild_id = original_ctx.guild.id
            if error:
                print(f'Player error in guild {guild_id} for song "{song_title}": {error}')
                asyncio.run_coroutine_threadsafe(
                    original_ctx.send(f"Playback error for '{song_title}': {error}"),
                    bot.loop
                )
            # Schedule play_next to run in the bot's event loop
            future = asyncio.run_coroutine_threadsafe(play_next(original_ctx), bot.loop)
            try:
                future.result(timeout=5) # Check for immediate errors from scheduling
            except asyncio.TimeoutError:
                print(f"play_next call from after_playing_song_callback (guild {guild_id}) timed out on future.result().")
            except Exception as e:
                print(f"Error running/scheduling play_next from after_playing_song_callback (guild {guild_id}): {e}")

        vc.play(source, after=after_playing_song_callback)
        await original_ctx.send(f'Now playing: **{song_title}**')
    except Exception as e:
        await original_ctx.send(f"An error occurred before playing '{song_title}': {e}")
        print(f"Error in play_next setup for '{song_title}' in guild {original_ctx.guild.id}: {e}")
        # Try to play the next song to prevent queue stall if this one failed badly
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

@bot.command(name='leave',aliases=['dc'], help='Tells the bot to leave the voice channel.')
async def leave(ctx):
    if ctx.voice_client is not None:
        # Clear queue when leaving
        music_queue.clear()
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
        return # Or implement logic to move the bot if preferred

    async with ctx.typing():
        audio_url, video_title = await get_audio_source(query)

        if audio_url is None or video_title is None:
            await ctx.send(f"Could not find a playable audio source for '{query}'.")
            return

        song_info = {'url': audio_url, 'title': video_title, 'ctx': ctx} # Store original ctx
        music_queue.append(song_info)
        await ctx.send(f"Added to queue: **{video_title}** (Position: {len(music_queue)})")

    if not vc.is_playing() and not vc.is_paused():
        await play_next(ctx) # Pass current command's context to start the process

@bot.command(name='stop', help='Stops the music and clears the queue.')
async def stop(ctx):
    if ctx.voice_client:
        music_queue.clear() # Clear queue first
        if ctx.voice_client.is_playing() or ctx.voice_client.is_paused():
            ctx.voice_client.stop() # Triggers 'after' callback which will find an empty queue
            await ctx.send("Music stopped and queue cleared.")
        else:
            await ctx.send("Nothing was playing, but queue has been cleared.")
    else:
        await ctx.send("I'm not in a voice channel.")
        music_queue.clear() # Still clear queue as a safety measure

@bot.command(name='skip', aliases=['s'], help='Skips the current song.')
async def skip(ctx):
    if ctx.voice_client and (ctx.voice_client.is_playing() or ctx.voice_client.is_paused()):
        await ctx.send("Skipping song...")
        ctx.voice_client.stop() # Triggers the 'after' callback in play_next
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
async def queue_command(ctx): # Renamed to avoid conflict with the 'music_queue' variable
    if not music_queue:
        await ctx.send("The music queue is empty.")
        return

    message = "**🎶 Current Music Queue:**\n"
    # Display what's currently playing if possible
    if ctx.voice_client and ctx.voice_client.is_playing():
        # This part is tricky as we don't easily have the title of the *exact* song object playing
        # For simplicity, we'll just list the upcoming queue
        pass

    if not music_queue: # Check again in case it became empty
        await ctx.send("The music queue is empty (nothing upcoming).")
        return

    for i, song in enumerate(list(music_queue)): # Iterate a copy for display
        message += f"{i+1}. {song['title']}\n"
        if i > 15 : # Limit display length
            message += f"... and {len(music_queue) - (i+1)} more.\n"
            break
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
        # Detailed traceback for server console
        import traceback
        traceback.print_exception(type(original_error), original_error, original_error.__traceback__)
    elif isinstance(error, commands.CheckFailure):
        await ctx.send(f"You do not have the necessary permissions or conditions to run `{ctx.command}`.")
    else:
        await ctx.send(f"An unexpected error occurred: {error}")
        print(f"Unexpected error: {error}")

# --- Run the Bot ---
if __name__ == "__main__":
    if TOKEN == 'YOUR_DISCORD_BOT_TOKEN' or not TOKEN: # Check if token is placeholder or empty
        print("ERROR: Please replace 'YOUR_DISCORD_BOT_TOKEN' with your actual bot token in the script or botkey.py.")
    else:
        try:
            bot.run(TOKEN)
        except discord.errors.LoginFailure:
            print("ERROR: Failed to log in. Make sure your bot token is correct and valid.")
        except Exception as e:
            print(f"An error occurred while trying to run the bot: {e}")
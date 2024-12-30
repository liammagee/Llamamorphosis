import os
import discord
from dotenv import load_dotenv

load_dotenv()

# Get environment variables
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GUILD_ID = int(os.getenv('DISCORD_GUILD_ID', 0))  # Default to 0 if not set
CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', 0))  # Default to 0 if not set

# Check if the required environment variables are set
if not TOKEN or not GUILD_ID or not CHANNEL_ID:
    raise ValueError("Environment variables DISCORD_BOT_TOKEN, DISCORD_GUILD_ID, and DISCORD_CHANNEL_ID must be set in the .env file.")

# Intents are required to use certain Discord API features
intents = discord.Intents.default()
intents.messages = True


# Create a bot client
client = discord.Client(intents=intents)

# Event triggered when the bot is ready
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

    # Get the guild (server) and channel (replace with your IDs)
    guild_id = 123456789012345678  # Replace with your server ID
    channel_id = 123456789012345678  # Replace with your channel ID
    
    guild = discord.utils.get(client.guilds, id=GUILD_ID)
    if guild:
        print(f'Connected to guild: {guild.name}')
        channel = discord.utils.get(guild.text_channels, id=CHANNEL_ID)
        if channel:
            await channel.send('Hello!')
        else:
            print('Channel not found!')
    else:
        print('Guild not found!')

# Run the bot
client.run(TOKEN)
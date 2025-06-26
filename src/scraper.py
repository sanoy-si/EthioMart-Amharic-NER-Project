import os
import json
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.tl.types import Channel

load_dotenv()

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")

client = TelegramClient('telegram_session', API_ID, API_HASH)

async def fetch_messages(channel_username, limit):
    all_messages = []
    async with client:
        try:
            entity = await client.get_entity(channel_username)
            if not isinstance(entity, Channel):
                print(f"Error: '{channel_username}' is not a channel.")
                return []

            print(f"Fetching messages from {entity.title}...")
            async for message in client.iter_messages(entity, limit=limit):
                if message.text: # Only collect messages with text
                    all_messages.append({
                        "channel": channel_username,
                        "message_id": message.id,
                        "text": message.text,
                        "date": message.date.isoformat(),
                        "views": message.views,
                    })
        except Exception as e:
            print(f"Could not fetch messages from {channel_username}. Error: {e}")
            return []

    print(f"Fetched {len(all_messages)} messages.")
    return all_messages

def save_messages_to_json(messages, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

    print(f"Saved messages to {filename}")
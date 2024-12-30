from ultralytics import YOLO, YOLOWorld
import cv2
import time
import numpy as np
from ollama import chat
from ollama import ChatResponse


import asyncio
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
guild = None
channel = None

what_ive_seen = []




# Event triggered when the bot is ready
async def post(guild, channel, message):
    if channel:
        await channel.send(message)
    else:
        print('Guild not found!')


async def run_basic_detection():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Use default camera
    if not cap.isOpened():
        raise Exception("Could not open camera")
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Loading YOLO model...")
    model = YOLO('yolov8s')

    # Define colors for visualization
    COLORS = np.random.uniform(0, 255, size=(1, 3))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Run detection on frame
            results = model(frame, stream=True, verbose = False)  # stream=True for better performance
            
            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    return class_name
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")



async def run_realtime_detection(classes, guild, channel):
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Use default camera
    if not cap.isOpened():
        raise Exception("Could not open camera")
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Loading YOLO model...")
    model = YOLOWorld('yolov8s-worldv2')
    if classes != None:
        model.set_classes(classes)  # Set to detect only glasses
    # model.set_classes([classes])  # Set to detect only glasses
    # target_class = 'glasses'  # The class we want to detect
    
    # Define colors for visualization
    COLORS = np.random.uniform(0, 255, size=(1, 3))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Run detection on frame
            results = model(frame, stream=True, verbose = False)  # stream=True for better performance
            
            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    if class_name not in what_ive_seen:
                        what_ive_seen.append(class_name)
                        await post(guild, channel, f'I just saw {class_name}')
                        response_to_object = await ollama_approach_flee_ignore(class_name)
                        await post(guild, channel, f'and I am going to {response_to_object}')

                    # Only process if the detected object is glasses
                    # if class_name == target_class:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                    
                    # Draw bounding box
                    color = COLORS[0].tolist()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f'{class_name} {conf:.2f}'
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            
            # Display the frame
            cv2.imshow('Real-time Object Detection', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")


async def ollama_list(prompt):
    response: ChatResponse = chat(model='llama3.2:3b', messages=[
    {
        'role': 'user',
        'content': f'give me just a comma-delimited list of simple physical features often found on a {prompt}.',
    },
    ])
    return response.message.content


async def ollama_approach_flee_ignore(obj):
    response: ChatResponse = chat(model='llama3.2:3b', messages=[
    {
        'role': 'user',
        'content': f'I am an insect. Given this object – {obj} – output a single word response: "approach", "flee" or "ignore".',
    },
    ])
    return response.message.content


async def main(guild, channel):
    try:

        # Run the bot
        classes = []
        while len(classes) == 0:
            first_class = await run_basic_detection()  # Assuming this needs to be async
            print(first_class)
            terms = await ollama_list(first_class)    # Assuming this needs to be async
            print(terms)
            classes = terms.split(',')

        extended_classes = ['glasses'] + classes
        await run_realtime_detection(extended_classes, guild, channel)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    # Event triggered when the bot is ready
    @client.event
    async def on_ready():
        print(f'We have logged in as {client.user}')
        
        guild = discord.utils.get(client.guilds, id=GUILD_ID)
        if guild:
            print(f'Connected to guild: {guild.name}')
            channel = discord.utils.get(guild.text_channels, id=CHANNEL_ID)
            if channel:
                await channel.send('*Bzzzzzzzt-click-click-brrzzztt! 🤖⚙️🐜🐝🕷️*')

                await main(guild, channel)
            else:
                print('Channel not found!')
        else:
            print('Guild not found!')
    client.run(TOKEN)

"""
Modular robot control system with configurable settings and command-line options.
"""

import argparse
import asyncio
import threading
import time
import socket
import numpy as np
from datetime import datetime
import cv2
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import discord
from dotenv import load_dotenv
from ultralytics import YOLO, YOLOWorld
from ollama import chat, ChatResponse
import struct
import random
import math
import select
import signal
from discord import File

load_dotenv()

# Environment variables
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GUILD_ID = int(os.getenv('DISCORD_GUILD_ID', 0))
CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', 0))

# Discord client setup
intents = discord.Intents.default()
intents.messages = True
client_discord = discord.Client(intents=intents)
guild = None
channel = None
what_ive_seen = []

class COMMAND:
    CMD_MOVE = "CMD_MOVE"
    CMD_LED_MOD = "CMD_LED_MOD"
    CMD_LED = "CMD_LED"
    CMD_SONIC = "CMD_SONIC"
    CMD_BUZZER = "CMD_BUZZER"
    CMD_HEAD = "CMD_HEAD"
    CMD_BALANCE = "CMD_BALANCE"
    CMD_ATTITUDE = "CMD_ATTITUDE"
    CMD_POSITION = "CMD_POSITION"
    CMD_RELAX = "CMD_RELAX"
    CMD_POWER = "CMD_POWER"
    CMD_CALIBRATION = "CMD_CALIBRATION"
    CMD_CAMERA = "CMD_CAMERA"
    CMD_SERVOPOWER = "CMD_SERVOPOWER"

# InsectClient class
class InsectClient:
    def __init__(self):
        self.tcp_flag = False
        self.client_socket = None
        self.video_socket = None
        self.frame = None
        self.video_flag = True
        self.recording = False
        self.video_writer = None
        self.yolo_enabled = False
        self.detected_objects = []
        self.move_speed = "8"
        self.servo_power = True
        self.balance_mode = False
        self.head_position = {"vertical": 90, "horizontal": 90}
        self.last_sonar_reading = float('inf')
        self.min_safe_distance = 20
        self.sonar_lock = threading.Lock()
        self.exploring = False
        self.last_exploration_time = time.time()
        self.exploration_interval = 5  
        self.exploration_duration = 2  
        self.recording_start_time = None
        self.current_recording_file = None        
        self.buzzing = None        
        self.exploration_path = []  
        self.position = (0, 0)  
        self.orientation = 0  
        self.grid_resolution = 50  
        self.visited_cells = set()  
        self.spin_probability = 0.3  
        self.max_spin_ang = 90

    def connect(self, ip, port=5002, video_port=8002):
        self.last_connection_params = (ip, port, video_port)
        # Close any existing connections first
        self.disconnect()

        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((ip, port))
            self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_socket.connect((ip, video_port))                       
            self.tcp_flag = True
            print("Connected to insect robot!")
            self.video_thread = threading.Thread(target=self.receive_video, daemon=True)
            self.video_thread.start()
            # self.sonar_thread = threading.Thread(target=self.monitor_sonar, daemon=True)
            # self.sonar_thread.start()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
        
    def is_valid_image_4_bytes(self, buf):
        bValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                bValid = False
        else:
            try:
                Image.open(io.BytesIO(buf)).verify()
            except:
                bValid = False
        return bValid
    
    def receive_video(self):
        """Receives and processes video stream from robot."""
        print("Video reception thread started")
        frame_count = 0
        
        try:
            # Create file-like interface for reading
            self.connection = self.video_socket.makefile('rb')
            print("Created video stream reader")
            self.tcp_flag = True

            while self.tcp_flag:
                try:
                    # Read length bytes with timeout and size check
                    stream_bytes = self.connection.read(4)
                    if not stream_bytes or len(stream_bytes) != 4:
                        print(f"Invalid stream bytes length: {len(stream_bytes) if stream_bytes else 0}")
                        time.sleep(0.1)
                        continue

                    # Unpack length
                    try:
                        leng = struct.unpack('<L', stream_bytes)[0]
                        if leng <= 0 or leng > 1000000:  # Sanity check on length
                            print(f"Invalid frame length: {leng}")
                            continue
                    except struct.error as e:
                        print(f"Error unpacking stream length: {e}")
                        continue

                    # Read frame data
                    jpg = self.connection.read(leng)
                    if len(jpg) != leng:
                        print(f"Incomplete frame data. Expected {leng} bytes, got {len(jpg)}")
                        continue

                    if self.is_valid_image_4_bytes(jpg):
                        if self.video_flag:
                            decoded_frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if decoded_frame is not None:
                                self.frame = decoded_frame
                                self.image = self.frame
                                
                                # Write frame if recording
                                if self.recording and self.video_writer is not None:
                                    try:
                                        self.video_writer.write(self.frame)
                                    except Exception as e:
                                        print(f"Error writing video frame: {e}")
                                        self.stop_recording()
                                
                                self.video_flag = False  # Indicate frame is being processed
                            else:
                                print("Failed to decode frame")
                        self.video_flag = True  # Reset flag for next frame
                    else:
                        print("Invalid JPEG data")
                        
                except ConnectionError as e:
                    print(f"Connection error in video stream: {e}")
                    break
                except Exception as e:
                    print(f"Error processing video frame: {e}")
                    continue
                    
        except Exception as e:
            print(f"Video reception error: {e}")
        finally:
            print("Video reception stopped")
            # Ensure video writer is properly closed
            if self.recording:
                self.stop_recording()
            # Clean up connection
            try:
                if self.connection:
                    self.connection.close()
            except:
                pass

    def stop_recording(self):
        """Stop recording and ensure proper cleanup."""
        duration = None
        try:
            if self.video_writer is not None and self.recording_start_time is not None:
                duration = time.time() - self.recording_start_time
                print("Releasing video writer...")
                self.video_writer.release()
                print(f"Video saved to: {self.current_recording_file}")
        except Exception as e:
            print(f"Error stopping recording: {e}")
        finally:
            self.video_writer = None
            self.recording = False
            self.recording_start_time = None
        return duration

    def start_recording(self):
        """Start recording video with proper frame validation."""
        if self.frame is None or not isinstance(self.frame, np.ndarray) or self.frame.size == 0:
            print("No valid video frame available to start recording")
            return None
            
        try:
            if self.recording:
                print("Already recording. Stopping current recording first.")
                self.stop_recording()
                
            height, width = self.frame.shape[:2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_recording_file = f"recording_{timestamp}.avi"
            
            # Use MJPG codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(
                self.current_recording_file,
                fourcc,
                20.0,  # FPS
                (width, height)
            )
            
            if not self.video_writer.isOpened():
                print("Failed to create video writer")
                return None
                
            print(f"Started recording to {self.current_recording_file}")
            self.recording = True
            self.recording_start_time = time.time()
            return self.current_recording_file
        except Exception as e:
            print(f"Error starting recording: {e}")
            if self.video_writer is not None:
                self.video_writer.release()
            self.recording = False
            self.recording_start_time = None
            self.current_recording_file = None
            self.video_writer = None
            return None

    def get_recording_duration(self):
        """Get current recording duration in seconds."""
        if self.recording and self.recording_start_time:
            return time.time() - self.recording_start_time
        return 0
    

    def monitor_sonar(self):
        while self.tcp_flag:
            try:
                if self.client_socket:
                    self.send_command("CMD_SONIC\n")
                    time.sleep(0.1)
            except Exception as e:
                print(f"Sonar monitoring error: {e}")
                time.sleep(0.5)

    def process_sonar_response(self, data):
        try:
            distance = float(data)
            with self.sonar_lock:
                self.last_sonar_reading = distance
            return distance
        except ValueError:
            print(f"Invalid sonar data received: {data}")
            return None

    def is_path_clear(self):
        with self.sonar_lock:
            return self.last_sonar_reading > self.min_safe_distance

    def set_min_safe_distance(self, distance):
        self.min_safe_distance = max(5, min(100, distance))
        print(f"Minimum safe distance set to {self.min_safe_distance}cm")

    def send_command(self, command):
        """Send command to robot and receive response."""

        print(f"Sending command: {command.strip()}")  # Log the command being sent
        
        if not self.tcp_flag or not self.client_socket:
            print("Not connected to robot")
            return None
            
        try:
            # Store original blocking state
            original_blocking = self.client_socket.getblocking()
            
            try:
                # Send the command
                self.client_socket.send(command.encode('utf-8'))
                
                # Set socket to non-blocking temporarily
                self.client_socket.setblocking(0)
                
                # Wait for response with timeout
                ready = select.select([self.client_socket], [], [], 0.5)  # 0.5 second timeout
                
                if ready[0]:
                    # Socket has data
                    response = self.client_socket.recv(1024).decode('utf-8')
                    return response.strip()
                else:
                    # print("No response received (timeout)")
                    return None
                    
            finally:
                # Always restore original blocking state
                if original_blocking != self.client_socket.getblocking():
                    self.client_socket.setblocking(original_blocking)
                    
        except socket.error as e:
            if e.errno == 9:  # Bad file descriptor
                print("Socket is in invalid state, attempting to reconnect...")
                self.tcp_flag = False
                # Attempt to reconnect using last known good IP and ports
                if hasattr(self, 'last_connection_params'):
                    self.connect(*self.last_connection_params)
            else:
                print(f"Socket error: {e}")
            return None
        except Exception as e:
            print(f"Failed to send command: {e}")
            return None


    def disconnect(self):
        self.tcp_flag = False
        if self.recording:
            self.stop_recording()
        if self.client_socket:
            self.client_socket.close()
        if self.video_socket:
            self.video_socket.close()
        print("Disconnected from insect robot")

    def update_position(self, movement, duration):
        """Update the robot's estimated position and orientation."""
        move_speed = 20  # estimated cm/s when moving
        
        if movement[0] in ['w', 's']:  # Forward/backward movement
            distance = move_speed * duration * (1 if movement[0] == 'w' else -1)
            # Calculate new position based on current orientation
            angle_rad = math.radians(self.orientation)
            dx = distance * math.sin(angle_rad)
            dy = distance * math.cos(angle_rad)
            self.position = (self.position[0] + dx, self.position[1] + dy)
        elif movement[0] in ['a', 'd']:  # Left/right movement
            distance = move_speed * duration
            # Adjust for sideways movement
            angle_rad = math.radians(self.orientation + (90 if movement[0] == 'd' else -90))
            dx = distance * math.sin(angle_rad)
            dy = distance * math.cos(angle_rad)
            self.position = (self.position[0] + dx, self.position[1] + dy)
        
        # Update orientation for spin movements
        if movement[0] == 'spin':
            self.orientation = (self.orientation + movement[1]) % 360
            
        # Convert current position to grid cell
        cell = (int(self.position[0] / self.grid_resolution),
                int(self.position[1] / self.grid_resolution))
        self.visited_cells.add(cell)
        
        # Store movement in path
        self.exploration_path.append({
            'movement': movement,
            'position': self.position,
            'orientation': self.orientation,
            'timestamp': time.time()
        })
        
    def get_unvisited_directions(self):
        """Get list of available directions that lead to unvisited cells."""
        possible_moves = []
        current_cell = (int(self.position[0] / self.grid_resolution),
                       int(self.position[1] / self.grid_resolution))
        
        # Check each cardinal direction
        for move in ['w', 'a', 's', 'd']:
            # Calculate potential new position
            test_pos = self.simulate_movement(current_cell, move)
            test_cell = (int(test_pos[0] / self.grid_resolution),
                        int(test_pos[1] / self.grid_resolution))
            
            if test_cell not in self.visited_cells:
                possible_moves.append(move)
        
        return possible_moves
        
    def simulate_movement(self, current_cell, move):
        """Simulate movement to check potential new position."""
        x, y = current_cell
        if move == 'w': y += 1
        elif move == 's': y -= 1
        elif move == 'a': x -= 1
        elif move == 'd': x += 1
        return (x, y)

    async def random_explore(self):
        """Performs exploration by spinning randomly and moving forward."""
        if not self.tcp_flag or not self.servo_power:
            return

        try:
            # First spin a random amount
            spin_angle = random.randint(-90, 90)  # Full range of rotation
            command = f"CMD_MOVE#2#0#0#8#{spin_angle}\n"
            self.send_command(command)
            
            # Wait for spin to complete - scale wait time with angle
            spin_duration = abs(spin_angle) / 45  # Adjust this value based on actual robot speed
            await asyncio.sleep(spin_duration)
            
            # Update position to track the spin
            self.update_position(('spin', spin_angle), spin_duration)
            
            # Stop spinning
            self.send_command("CMD_MOVE#1#0#0#8#0\n")
            await asyncio.sleep(0.5)  # Brief pause after spin

            # Check if path is clear before moving forward
            if self.is_path_clear():
                # Move forward
                command = f"CMD_MOVE#1#0#35#8#0\n"  # Forward movement
                self.send_command(command)
                await asyncio.sleep(self.exploration_duration)
                
                # Update position with forward movement
                self.update_position(('w', None), self.exploration_duration)
                
                # Stop movement
                self.send_command("CMD_MOVE#1#0#0#8#0\n")
            else:
                print("Obstacle detected, skipping forward movement")
                
            # Add current position to visited cells
            cell = (int(self.position[0] / self.grid_resolution),
                int(self.position[1] / self.grid_resolution))
            self.visited_cells.add(cell)
            
            # Store movement in path
            self.exploration_path.append({
                'movement': 'explore',
                'spin_angle': spin_angle,
                'moved_forward': self.is_path_clear(),
                'position': self.position,
                'orientation': self.orientation,
                'timestamp': time.time()
            })

        except Exception as e:
            print(f"Error during exploration: {e}")
            # Ensure robot stops on error
            self.send_command("CMD_MOVE#1#0#0#8#0\n")            
    
    def start_exploration(self):
        """Enables random exploration mode."""
        self.exploring = True
        print("Random exploration enabled")

    def stop_exploration(self):
        """Disables random exploration mode."""
        self.exploring = False
        print("Random exploration disabled")        

class InsectControl:
    def __init__(self):
        self.client = InsectClient()
        self.cmd = COMMAND()
        self.current_led_mode = 0
        self.last_movement = None
        self.last_detection_time = 0 

    async def manage_exploration(self):
        """Manages random exploration behavior."""
        while True:
            if self.client.exploring:
                current_time = time.time()
                if (current_time - self.client.last_exploration_time) >= self.client.exploration_interval:
                    # Only explore if we haven't seen anything new recently
                    if not what_ive_seen or (current_time - self.last_detection_time) > 15:
                        await self.client.random_explore()
                        self.client.last_exploration_time = current_time
            await asyncio.sleep(1)        

    def handle_movement(self, direction):
        x, y = 0, 0
        if direction == 'w':
            # Check obstacle
            if not self.client.is_path_clear():
                print("Obstacle ahead, cannot move forward!")
                self.stop_movement()
                return
            y = 35
        elif direction == 's':
            y = -35
        elif direction == 'a':
            x = -35
        elif direction == 'd':
            x = 35

        self.last_movement = direction
        command = f"{self.cmd.CMD_MOVE}#1#{x}#{y}#{self.client.move_speed}#0\n"
        self.client.send_command(command)

    def handle_spin(self, direction):
        """Handle spinning movement."""
        x, y = 0, 0
        angle = 10 if direction == 'right' else -10
        
        command = f"{self.cmd.CMD_MOVE}#2#{x}#{y}#{self.client.move_speed}#{angle}\n"
        self.client.send_command(command)

    def stop_movement(self):
        # Send a stop command (for example: x=0, y=0)
        command = f"{self.cmd.CMD_MOVE}#1#0#0#{self.client.move_speed}#0\n"
        self.client.send_command(command)

    def toggle_servo_power(self):
        self.client.servo_power = not self.client.servo_power
        command = f"{self.cmd.CMD_SERVOPOWER}#{1 if self.client.servo_power else 0}\n"
        self.client.send_command(command)
        print(f"Servos {'powered' if self.client.servo_power else 'relaxed'}")

    def get_sonar(self):
        command = f"{self.cmd.CMD_SONIC}\n"
        self.client.send_command(command)

    def toggle_buzz(self):
        self.client.buzzing = not self.client.buzzing
        if self.client.buzzing:
            command = f"{self.cmd.CMD_BUZZER}#0\n"
        else:
            command = f"{self.cmd.CMD_BUZZER}#1\n"
        self.client.send_command(command)

    def toggle_balance(self):
        self.client.balance_mode = not self.client.balance_mode
        command = f"{self.cmd.CMD_BALANCE}#{1 if self.client.balance_mode else 0}\n"
        self.client.send_command(command)
        print(f"Balance mode {'enabled' if self.client.balance_mode else 'disabled'}")

    def move_head(self, direction):
        if direction == 'up':
            self.client.head_position["vertical"] = min(180, self.client.head_position["vertical"] + 5)
            axis = "0"
            angle = str(self.client.head_position["vertical"])
        elif direction == 'down':
            self.client.head_position["vertical"] = max(0, self.client.head_position["vertical"] - 5)
            axis = "0"
            angle = str(self.client.head_position["vertical"])
        elif direction == 'left':
            self.client.head_position["horizontal"] = min(180, self.client.head_position["horizontal"] + 5)
            axis = "1"
            angle = str(self.client.head_position["horizontal"])
        elif direction == 'right':
            self.client.head_position["horizontal"] = max(0, self.client.head_position["horizontal"] - 5)
            axis = "1"
            angle = str(self.client.head_position["horizontal"])
        
        command = f"{self.cmd.CMD_HEAD}#{axis}#{angle}\n"
        self.client.send_command(command)

    def cycle_led_mode(self):
        self.current_led_mode = (self.current_led_mode % 5) + 1
        command = f"{self.cmd.CMD_LED_MOD}#{self.current_led_mode}\n"
        self.client.send_command(command)
        print(f"LED Mode: {self.current_led_mode}")

    def custom_led(self):
        try:
            r = int(input("Enter RED value (0-255): "))
            g = int(input("Enter GREEN value (0-255): "))
            b = int(input("Enter BLUE value (0-255): "))
            command = f"{self.cmd.CMD_LED}#{r}#{g}#{b}\n"
            self.client.send_command(command)
        except ValueError:
            print("Invalid input. Please enter numbers between 0-255")

    def get_power_status(self):
        command = f"{self.cmd.CMD_POWER}\n"
        self.client.send_command(command)

    def show_preview(self):
        if self.client.frame is not None:
            cv2.imshow('Insect Robot Video', cv2.cvtColor(self.client.frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        else:
            print("No video frame available")

    def print_detections(self):
        if self.client.yolo_enabled and self.client.detected_objects:
            print("\nCurrent detections:")
            for obj in self.client.detected_objects:
                print(f"- {obj['class']}: {obj['confidence']:.2f}")
        else:
            print("No detections available (YOLO disabled or no objects detected)")


async def post(guild, channel, message):
    if channel:
        await channel.send(message)
    else:
        print('Guild not found!')

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
    return response.message.content.strip().lower()

async def run_basic_detection(insect_controller):
    # Use robot's video feed instead of local camera
    model = YOLO('yolov8s')
    
    try:
        while True:
            if insect_controller.client.frame is None:
                await asyncio.sleep(0.1)
                continue
                
            frame = insect_controller.client.frame
            results = model(frame, stream=True, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    return class_name
                    
            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse
            
    except Exception as e:
        print(f"Error in basic detection: {e}")
        return None

async def run_realtime_detection(classes, guild, channel, insect_controller, config):
    print("Starting realtime detection...")
   
    # Performance settings
    FRAME_SKIP = 1  # Process every Nth frame
    SCALE_FACTOR = 1.0 # 0.5  # Scale down frames by this factor (0.5 = half size)
    PROCESS_INTERVAL = 0.1  # Minimum time between processing frames
     
    frame_count = 0
    last_process_time = 0

    model_world = YOLOWorld(config.yolo.world_model)
    if classes:
        model_world.set_classes(classes)
    COLORS = np.random.uniform(0, 255, size=(1, 3))
    
    try:
        while True:
            current_time = time.time()
            
            # Skip if we're processing too frequently
            if current_time - last_process_time < PROCESS_INTERVAL:
                await asyncio.sleep(0.01)
                continue
            
            if insect_controller.client.frame is None:
                await asyncio.sleep(0.1)
                continue
                
            frame_count += 1
            
            if frame_count % FRAME_SKIP != 0:
                # Show original frame without detection
                if config.video_enabled:
                    cv2.imshow('Real-time Object Detection', insect_controller.client.frame)
                    # cv2.waitKey(1)
                continue
            
            # Make a copy and resize for processing
            original_frame = insect_controller.client.frame.copy()
            if SCALE_FACTOR != 1.0:
                height, width = original_frame.shape[:2]
                new_width = int(width * SCALE_FACTOR)
                new_height = int(height * SCALE_FACTOR)
                process_frame = cv2.resize(original_frame, (new_width, new_height))
            else:
                process_frame = original_frame
            original_frame = insect_controller.client.frame.copy()  # Make a copy to avoid modifying the original
           
            results = model_world(process_frame, stream=True, verbose=False)
            display_frame = original_frame.copy()  # For drawing detections
            post_image = False

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model_world.names[class_id]

                    if config.discord_integration:
                        if class_name not in what_ive_seen:
                            if not post_image:
                                post_image = True
                            what_ive_seen.append(class_name)
                            await post(guild, channel, f'I just saw {class_name}')
                            response_to_object = await ollama_approach_flee_ignore(class_name)
                            await post(guild, channel, f'I am going to {response_to_object}')

                            # Integrate with Insect Robot:
                            if response_to_object == "approach":
                                insect_controller.handle_movement('w')  # Move forward
                                await asyncio.sleep(1)  # Move for a second
                                insect_controller.stop_movement()
                            elif response_to_object == "flee":
                                insect_controller.handle_movement('s')  # Move backward
                                await asyncio.sleep(1)
                                insect_controller.stop_movement()
                        
                        # Resume exploration after handling the object
                        await asyncio.sleep(2)
                        insect_controller.client.exploring = True
                        
                    # Draw bounding box on frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = COLORS[0].tolist()
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {float(box.conf[0]):.2f}'
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Show processed frame
            if config.video_enabled:
                cv2.imshow('Real-time Object Detection', display_frame)
                cv2.waitKey(1)                    
            
            if config.discord_integration and channel and post_image:
                # Save the current frame as an image
                image_path = "current_frame.jpg"
                cv2.imwrite(image_path, display_frame)

                # Send the image to Discord
                with open(image_path, 'rb') as f:
                    discord_file = File(f)
                    await channel.send(file=discord_file)

                # Clean up the temporary image file
                os.remove(image_path)            

            last_process_time = current_time
            await asyncio.sleep(0.01)  # Small delay to prevent CPU overuse
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        insect_controller.client.stop_exploration()


@dataclass
class RobotSettings:
    """Configuration settings for the robot."""
    ip_address: str = "10.0.0.250"
    control_port: int = 5002
    video_port: int = 8002
    move_speed: int = 8
    min_safe_distance: float = 20.0
    exploration_interval: float = 5.0
    exploration_duration: float = 2.0
    head_default_vertical: int = 90
    head_default_horizontal: int = 90

@dataclass
class YOLOSettings:
    """Configuration settings for YOLO models."""
    base_model: str = 'yolov8s'
    world_model: str = 'yolov8s-worldv2'
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45

@dataclass
class DiscordSettings:
    """Configuration settings for Discord integration."""
    token: Optional[str] = None
    guild_id: Optional[int] = None
    channel_id: Optional[int] = None
    
    @classmethod
    def from_env(cls):
        """Create settings from environment variables."""
        load_dotenv()
        return cls(
            token=os.getenv('DISCORD_BOT_TOKEN'),
            guild_id=int(os.getenv('DISCORD_GUILD_ID', 0)),
            channel_id=int(os.getenv('DISCORD_CHANNEL_ID', 0))
        )

class ConfigManager:
    """Manages loading and saving of configuration settings."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.robot = RobotSettings()
        self.yolo = YOLOSettings()
        self.discord = DiscordSettings.from_env()
        self.explore = False
        self.discord_integration = False
        self.video_enabled = True
        self.detection_enabled = False

        if self.config_path.exists():
            self.load_config()

    def print_config(self):
        """Print the current configuration settings."""
        print("Robot Settings:")
        for key, value in self.robot.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nYOLO Settings:")
        for key, value in self.yolo.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nDiscord Settings:")
        for key, value in self.discord.__dict__.items():
            print(f"  {key}: {value}")

        print("\nAdditional Settings:")
        print(f"  Explore: {self.explore}")
        print(f"  Discord Integration: {self.discord_integration}")
        print(f"  Video Enabled: {self.video_enabled}")
        print(f"  Detection Enabled: {self.detection_enabled}")


    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                config = json.load(f)
                self.robot = RobotSettings(**config.get('robot', {}))
                self.yolo = YOLOSettings(**config.get('yolo', {}))
                self.explore = config.get('explore', self.explore)
                self.discord_integration = config.get('discord_integration', self.discord_integration)
                self.video_enabled = config.get('video_enabled', self.video_enabled)
                self.detection_enabled = config.get('detection_enabled', self.detection_enabled)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_config(self):
        """Save current configuration to JSON file."""
        config = {
            'robot': self.robot.__dict__,
            'yolo': self.yolo.__dict__,
            'explore': self.explore,
            'discord_integration': self.discord_integration,
            'video_enabled': self.video_enabled,
            'detection_enabled': self.detection_enabled
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)


class KeyboardController:

    def __init__(self, insect_control: 'InsectControl', active_tasks=None, discord_client=None):
        self.control = insect_control
        self.active_tasks = active_tasks or []
        self.discord_client = discord_client
        self.commands = {
            'w': ('Move forward', self.control.handle_movement),
            's': ('Move backward', self.control.handle_movement),
            'a': ('Turn left', self.control.handle_movement),
            'd': ('Turn right', self.control.handle_movement),
            'q': ('Spin left', lambda: self.control.handle_spin('left')),
            'e': ('Spin right', lambda: self.control.handle_spin('right')),
            't': ('Head up', lambda: self.control.move_head('up')),
            'g': ('Head down', lambda: self.control.move_head('down')),
            'f': ('Head left', lambda: self.control.move_head('left')),
            'h': ('Head right', lambda: self.control.move_head('right')),
            ' ': ('Stop movement', self.control.stop_movement),
            'r': ('Toggle servo power', self.control.toggle_servo_power),
            'b': ('Toggle balance mode', self.control.toggle_balance),
            'x': ('Toggle exploration mode', self.toggle_exploration),
            'n': ('Toggle sonar', self.control.get_sonar),
            'z': ('Buzz', self.control.toggle_buzz),
            'v': ('Toggle video recording mode', self.toggle_video_recording),
            'y': ('Preview', self.toggle_yolo),
            'o': ('Print detections', self.control.print_detections),
            '?y': ('Print help', self.print_help),
            'c': ('Get power status', self.control.get_power_status),
            'p': ('Preview', self.control.show_preview),
            '+': ('Increase speed', self.increase_speed),
            '-': ('Decrease speed', self.decrease_speed),
            '!': ('Quit', self.quit),
            'esc': ('Quit', self.quit)
        }
        self.running = True

    def quit(self):
        """Initiate graceful shutdown of all components."""
        if not self.running:  # Prevent multiple quit calls
            return
            
        print("\nInitiating shutdown...")
        self.running = False

        # Force stop all video windows
        cv2.destroyAllWindows()

        # Stop robot first
        self.control.stop_movement()
        self.control.client.tcp_flag = False  # Force stop video thread
        
        # Aggressively cancel all tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        
        # Stop the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.stop()
        except Exception as e:
            print(f"Error stopping event loop: {e}")
        

    def increase_speed(self):
        """Increase the robot's speed by one unit, ensuring it stays within the valid range."""
        current_speed = int(self.control.client.move_speed)
        if current_speed < 9:
            self.control.client.move_speed = str(current_speed + 1)
            print(f"Speed increased to {self.control.client.move_speed}")
        else:
            print("Speed is already at maximum")


    def decrease_speed(self):
        """Decrease the robot's speed by one unit, ensuring it stays within the valid range."""
        current_speed = int(self.control.client.move_speed)
        if current_speed > 1:
            self.control.client.move_speed = str(current_speed - 1)
            print(f"Speed decreased to {self.control.client.move_speed}")
        else:
            print("Speed is already at minimum")

    
        
    def handle_keypress(self, key):
        """Handle key press events including escape key."""
        if key == 'esc':  # Replace with actual escape key code for your environment
            self.quit()
            return True
        elif key in self.commands:
            _, func = self.commands[key]
            if key in ['w', 'a', 's', 'd']:
                func(key)
            elif key in ['sd']:
                func(key)
            else:
                func()
            return True
        return False
    

    def print_help(self):
        """Display available commands."""
        print("\nAvailable commands:")
        for key, (desc, _) in self.commands.items():
            print(f"{key}: {desc}")
    
    def toggle_exploration(self):
        if self.control.client.exploring:
            self.control.client.stop_exploration()
        else:
            self.control.client.start_exploration()
    
    async def preview_loop(self):
        """Continuously update video preview."""
        print("Starting video preview loop...")
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            if self.control.client.frame is not None:
                frame_count += 1
                if frame_count % 30 == 0:  # Print FPS every 30 frames
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    # print(f"Video FPS: {fps:.1f}")
                
                cv2.imshow('Insect Robot Video', self.control.client.frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.quit()
            else:
                print("Waiting for video frame...")
                await asyncio.sleep(1)  # Wait longer when no frame
                continue
                
            await asyncio.sleep(0.03)  # ~30 FPS
    
    async def input_loop(self):
        """Main input loop for keyboard control."""
        self.print_help()
        while self.running:
            cmd = await asyncio.get_event_loop().run_in_executor(None, input, "Enter command: ")
            if cmd in self.commands:
                _, func = self.commands[cmd]
                if cmd in ['w', 'a', 's', 'd']:
                    func(cmd)
                else:
                    func()
            elif cmd == 'h':
                self.print_help()

    def format_duration(self, seconds):
        """Format duration in seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def toggle_video_recording(self):
        """Toggle video recording state with enhanced feedback."""
        current_time = datetime.now().strftime("%H:%M:%S")

        # Proper frame check
        if self.control.client.frame is None or not isinstance(self.control.client.frame, np.ndarray):
            print(f"[{current_time}] ❌ Error: No video feed available. Cannot start recording.")
            return

        try:
            if not self.control.client.recording:
                # Start recording
                filename = self.control.client.start_recording()
                if filename:
                    print(f"[{current_time}] 📹 Recording started")
                    print(f"📁 Saving to: {filename}")
            else:
                # Stop recording and get duration
                duration = self.control.client.stop_recording()
                if duration:
                    formatted_duration = self.format_duration(duration)
                    print(f"[{current_time}] ⏹️ Recording stopped")
                    print(f"⏱️ Duration: {formatted_duration}")
                    print(f"💾 Saved as: {self.control.client.current_recording_file}")

        except Exception as e:
            print(f"[{current_time}] ❌ Error toggling video recording: {e}")
    
    def toggle_yolo(self):
        ### TODO: Move to InsectControl class
        self.control.client.yolo_enabled = not self.control.client.yolo_enabled
        print(f"YOLO detection {'enabled' if self.control.client.yolo_enabled else 'disabled'}")


    async def display_recording_status(self):
        """Periodically display recording duration in the preview window."""
        while self.running:
            try:
                # Proper frame check for recording status display
                if (self.control.client.recording and 
                    self.control.client.frame is not None and 
                    isinstance(self.control.client.frame, np.ndarray) and 
                    self.control.client.frame.size > 0):
                    
                    try:
                        frame = self.control.client.frame.copy()
                        duration = self.control.client.get_recording_duration()
                        formatted_duration = self.format_duration(duration)
                        
                        # Add recording indicator and duration to frame
                        cv2.putText(frame, 
                                f"REC {formatted_duration}", 
                                (10, 30),  # Position in top-left
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1,  # Font scale
                                (0, 0, 255),  # Red color
                                2)  # Thickness
                        
                        # Display red circle for recording indicator
                        cv2.circle(frame, (35, 25), 8, (0, 0, 255), -1)
                        
                        cv2.imshow('Insect Robot Video', frame)
                    except Exception as e:
                        print(f"Error modifying/displaying frame: {e}")
                        continue
                
            except Exception as e:
                print(f"Error in recording status display: {e}")
                
            await asyncio.sleep(0.1)  # Update every 100ms

            
def setup_discord(settings: DiscordSettings, insect_controller: 'InsectControl', config) -> Optional[discord.Client]:
    """Initialize Discord client with given settings."""
    if not all([settings.token, settings.guild_id, settings.channel_id]):
        print("Incomplete Discord settings, skipping Discord integration")
        return None

    intents = discord.Intents.default()
    intents.messages = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f'Logged in as {client.user}')
        guild = discord.utils.get(client.guilds, id=settings.guild_id)
        if guild:
            print(f'Connected to guild: {guild.name}')
            channel = discord.utils.get(guild.text_channels, id=settings.channel_id)
            if channel:
                await channel.send('*Bzzzzzzzt-click-click-brrzzztt! 🤖⚙️🐜🐝🕷️*')
                # Start the detection loop with Discord integration
                if insect_controller.client.yolo_enabled:
                    await run_realtime_detection([], guild, channel, insect_controller, config)
            else:
                print('Discord channel not found!')
        else:
            print('Discord guild not found!')

    return client

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Robot Control System')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--explore', action='store_true',
                      help='Enable autonomous exploration mode')
    parser.add_argument('--discord', action='store_true',
                      help='Enable Discord integration')
    parser.add_argument('--no-video', action='store_true',
                      help='Disable video feed')
    parser.add_argument('--detection', action='store_true',
                      help='Enable object detection')
    return parser.parse_args()

async def shutdown(tasks, insect_controller, discord_client):
    """Handle graceful shutdown of all components."""
    print("\nPerforming cleanup...")
    
    # Force stop all OpenCV windows first
    cv2.destroyAllWindows()
    
    # Handle robot shutdown
    if insect_controller:
        try:
            # Stop any movement
            insect_controller.stop_movement()
            
            # Stop recording if active
            if insect_controller.client.recording:
                insect_controller.client.stop_recording()
            
            # Force video thread to stop
            insect_controller.client.tcp_flag = False
            
            # Disconnect from robot
            insect_controller.client.disconnect()
        except Exception as e:
            print(f"Error during robot cleanup: {e}")
    
    # Close Discord client
    if discord_client:
        try:
            await discord_client.close()
        except Exception as e:
            print(f"Error closing Discord client: {e}")
    
    # Cancel all tasks immediately
    for task in tasks:
        try:
            if not task.done():
                task.cancel()
        except Exception as e:
            print(f"Error cancelling task: {e}")
    
    # Wait briefly for tasks to complete
    if tasks:
        try:
            await asyncio.wait(tasks, timeout=0.5)
        except Exception as e:
            print(f"Error during task cleanup: {e}")
    
    # Stop the event loop
    try:
        loop = asyncio.get_running_loop()
        loop.stop()
    except Exception as e:
        print(f"Error stopping event loop: {e}")

async def main():
    args = parse_args()
    config = ConfigManager(args.config)
    
    # Use config values by default, fall back to CLI args if not set
    config.explore = getattr(config, 'explore', False) or args.explore
    config.discord_integration = getattr(config, 'discord_integration', False) or args.discord
    config.video_enabled = getattr(config, 'video_enabled', True) and not args.no_video
    config.detection_enabled = getattr(config, 'detection_enabled', False) or args.detection
    config.print_config()

    # Initialize robot controller
    insect_controller = InsectControl()
    
    # Connect to robot
    if not insect_controller.client.connect(config.robot.ip_address,
                                          config.robot.control_port,
                                          config.robot.video_port):
        print("Failed to connect to robot")
        return

    # Store tasks and clients for cleanup
    active_tasks = set()
    discord_client = None
    keyboard = None
    
    try:
        # Set up keyboard control first
        keyboard = KeyboardController(insect_controller, active_tasks, discord_client)
        
        # Create and add tasks one by one
        if config.video_enabled:
            preview_task = asyncio.create_task(keyboard.preview_loop())
            status_task = asyncio.create_task(keyboard.display_recording_status())
            active_tasks.update({preview_task, status_task})
        
        input_task = asyncio.create_task(keyboard.input_loop())
        active_tasks.add(input_task)
        
        if config.explore:
            exploration_task = asyncio.create_task(insect_controller.manage_exploration())
            active_tasks.add(exploration_task)
        
        if config.detection_enabled:
            insect_controller.client.yolo_enabled = True
            detection_task = asyncio.create_task(
                run_realtime_detection([], None, None, insect_controller, config)
            )
            active_tasks.add(detection_task)
        
        if config.discord_integration:
            discord_client = setup_discord(config.discord, insect_controller, config)
            if discord_client:
                discord_task = asyncio.create_task(discord_client.start(config.discord.token))
                active_tasks.add(discord_task)
                keyboard.discord_client = discord_client  # Update keyboard's discord client

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, keyboard.quit)
        
        # Wait for completion or interruption
        try:
            await asyncio.gather(*active_tasks)
        except asyncio.CancelledError:
            print("Tasks cancelled")
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
        
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        if keyboard:
            keyboard.quit()
        await shutdown(active_tasks, insect_controller, discord_client)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Force close any remaining windows
        cv2.destroyAllWindows()
        print("\nShutdown complete.")
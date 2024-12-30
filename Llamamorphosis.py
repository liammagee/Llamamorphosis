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
    def __init__(self):
        pass

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
        self.model = YOLO('yolov8m_ncnn_model')  # Adjust path if needed
        self.last_sonar_reading = float('inf')
        self.min_safe_distance = 20
        self.sonar_lock = threading.Lock()
        self.exploring = False
        self.last_exploration_time = time.time()
        self.exploration_interval = 5  # seconds between exploration moves
        self.exploration_duration = 2  # seconds per move

        # Add new attributes for path tracking
        self.exploration_path = []  # Store sequence of movements
        self.position = (0, 0)  # Track relative position (x, y)
        self.orientation = 0  # Track orientation in degrees (0 = initial direction)
        self.grid_resolution = 50  # cm per grid cell
        self.visited_cells = set()  # Track visited grid cells
        self.spin_probability = 0.3  # 30% chance to spin before moving
        self.max_spin_ang


    def connect(self, ip, port=5002, video_port=8002):
        # Close any existing connections first
        self.disconnect()
        
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((ip, port))
            self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_socket.connect((ip, video_port))

                        
            self.tcp_flag = True
            print("Connected to insect robot!")
            
            # Start threads (optional if video/sonar needed)
            self.video_thread = threading.Thread(target=self.receive_video, daemon=True)
            self.video_thread.start()
            
            self.sonar_thread = threading.Thread(target=self.monitor_sonar, daemon=True)
            self.sonar_thread.start()
            
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
                    stream_bytes= self.connection.read(4)
                    leng=struct.unpack('<L', stream_bytes[:4])
                    jpg=self.connection.read(leng[0])
                    if self.is_valid_image_4_bytes(jpg):
                        
                        if self.video_flag:
                            self.frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            self.image = self.frame
                            self.video_flag = False  # Indicate frame is being processed
                        self.video_flag = True  # Reset flag for next frame
                        
                except Exception as e:
                    print(f"Error processing video frame: {e}")
                    continue
                    
        except Exception as e:
            print(f"Video reception error: {e}")
        finally:
            print("Video reception stopped")
            if self.recording:
                self.stop_recording()

    def start_recording(self):
        if self.frame is not None:
            height, width = self.frame.shape[:2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            self.recording = True
            print(f"Started recording to {filename}")
        else:
            print("No video frame available to start recording")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Stopped recording")

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
        if self.tcp_flag:
            try:
                self.client_socket.send(command.encode('utf-8'))
                return True
            except Exception as e:
                print(f"Failed to send command: {e}")
                return False
        else:
            print("Not connected to robot")
            return False

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
        """Performs intelligent random exploration with spinning."""
        if not self.tcp_flag or not self.servo_power:
            return

        # First, decide if we should spin
        if random.random() < self.spin_probability:
            # Choose a random spin angle
            spin_angle = random.uniform(-self.max_spin_angle, self.max_spin_angle)
            command = f"CMD_MOVE#2#0#0#8#{spin_angle}\n"
            self.send_command(command)
            await asyncio.sleep(abs(spin_angle) / 45)  # Adjust sleep based on spin angle
            self.update_position(('spin', spin_angle), abs(spin_angle) / 45)
            self.send_command("CMD_MOVE#1#0#0#8#0\n")  # Stop spinning
            await asyncio.sleep(0.5)  # Brief pause after spin

        # Get unvisited directions
        available_moves = self.get_unvisited_directions()
        
        # If all adjacent cells have been visited, choose any valid movement
        if not available_moves:
            available_moves = ['w', 's', 'a', 'd']
            if not self.is_path_clear():
                available_moves.remove('w')  # Remove forward movement if obstacle detected

        if available_moves:
            movement = random.choice(available_moves)
            x, y = 0, 0
            if movement == 'w': y = 35
            elif movement == 's': y = -35
            elif movement == 'a': x = -35
            elif movement == 'd': x = 35

            command = f"CMD_MOVE#1#{x}#{y}#8#0\n"
            self.send_command(command)
            await asyncio.sleep(self.exploration_duration)
            self.update_position((movement, None), self.exploration_duration)
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
        self.exploration_task = None

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

    def main_loop(self):

        self.print_help()
        
        while self.running:
            try:
                cmd = input("Enter command: ").lower()
                
                if cmd in ['w', 'a', 's', 'd']:
                    self.handle_movement(cmd)
                elif cmd == ' ':  # space key
                    self.stop_movement()
                elif cmd == 'r':
                    self.toggle_servo_power()
                elif cmd == 'b':
                    self.toggle_balance()
                elif cmd == 'n':
                    self.get_sonar()
                elif cmd.startswith('sd'):  # Set minimum safe distance
                    try:
                        distance = int(cmd[2:])
                        self.client.set_min_safe_distance(distance)
                    except ValueError:
                        print("Invalid distance. Usage: sd30 (sets safe distance to 30cm)")
                elif cmd == 'z':
                    command = f"{self.cmd.CMD_BUZZER}#1\n"
                    self.client.send_command(command)
                elif cmd == 't':
                    self.move_head('up')
                elif cmd == 'g':
                    self.move_head('down')
                elif cmd == 'f':
                    self.move_head('left')
                elif cmd == 'h':
                    self.move_head('right')
                elif cmd == 'l':
                    self.custom_led()
                elif cmd == 'm':
                    self.cycle_led_mode()
                elif cmd == 'c':
                    self.get_power_status()
                elif cmd == 'v':
                    if not self.client.recording:
                        self.client.start_recording()
                    else:
                        self.client.stop_recording()
                elif cmd == 'p':
                    self.show_preview()
                elif cmd == 'y':
                    self.client.yolo_enabled = not self.client.yolo_enabled
                    print(f"YOLO detection {'enabled' if self.client.yolo_enabled else 'disabled'}")
                elif cmd == 'o':
                    self.print_detections()
                elif cmd == 'h':
                    self.print_help()
                elif cmd == 'q':
                    self.running = False
                elif cmd.isdigit() and 1 <= int(cmd) <= 9:
                    self.client.move_speed = cmd
                    print(f"Speed set to {cmd}")
                else:
                    print("Unknown command. Type 'h' for help")

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"Error: {e}")

        cv2.destroyAllWindows()
        self.client.disconnect()



load_dotenv()

TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GUILD_ID = int(os.getenv('DISCORD_GUILD_ID', 0))
CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', 0))

intents = discord.Intents.default()
intents.messages = True

client_discord = discord.Client(intents=intents)
guild = None
channel = None
what_ive_seen = []

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

async def run_realtime_detection(classes, guild, channel, insect_controller):
    print("Starting realtime detection...")
    
    model_world = YOLOWorld('yolov8s-worldv2')
    if classes:
        model_world.set_classes(classes)
    COLORS = np.random.uniform(0, 255, size=(1, 3))
    
    try:
        while True:
            if insect_controller.client.frame is None:
                await asyncio.sleep(0.1)
                continue
                
            frame = insect_controller.client.frame.copy()  # Make a copy to avoid modifying the original
            results = model_world(frame, stream=True, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model_world.names[class_id]
                    
                    if class_name not in what_ive_seen:
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {float(box.conf[0]):.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Real-time Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse
                
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
        
        if self.config_path.exists():
            self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path) as f:
                config = json.load(f)
                self.robot = RobotSettings(**config.get('robot', {}))
                self.yolo = YOLOSettings(**config.get('yolo', {}))
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def save_config(self):
        """Save current configuration to JSON file."""
        config = {
            'robot': self.robot.__dict__,
            'yolo': self.yolo.__dict__
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

class KeyboardController:
    """Handles keyboard input for robot control."""
    
    def __init__(self, insect_control: 'InsectControl'):
        self.control = insect_control
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
            'esc': ('Quit', self.quit)
        }
        self.running = True
        
    def handle_keypress(self, key):
        """Handle key press events including escape key."""
        if key == 'esc':  # Replace with actual escape key code for your environment
            self.quit()
            return True
        elif key in self.commands:
            _, func = self.commands[key]
            if key in ['w', 'a', 's', 'd']:
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
    
    def quit(self):
        """Properly shut down all subsystems."""
        print("Shutting down...")
        
        # Stop any active movement
        self.control.stop_movement()
        
        # Stop exploration if active
        if self.control.client.exploring:
            self.control.client.stop_exploration()
            
        # Disconnect from robot
        if self.control.client.tcp_flag:
            if self.control.client.recording:
                self.control.client.stop_recording()
            self.control.client.disconnect()
            
        # Close any open windows
        cv2.destroyAllWindows()
        
        # Set running to false to stop input loop
        self.running = False
        
        print("Shutdown complete")
    
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

def setup_discord(settings: DiscordSettings, insect_controller: 'InsectControl') -> Optional[discord.Client]:
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
                    await run_realtime_detection([], guild, channel, insect_controller)
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

async def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Initialize robot controller
    insect_controller = InsectControl()
    
    # Connect to robot
    if not insect_controller.client.connect(config.robot.ip_address,
                                          config.robot.control_port,
                                          config.robot.video_port):
        print("Failed to connect to robot")
        return
    
    try:
        # Set up keyboard control
        keyboard = KeyboardController(insect_controller)
        
        # Start tasks based on command line arguments
        tasks = [keyboard.input_loop()]
        
        # Add video preview task if video is enabled
        if not args.no_video:
            tasks.append(keyboard.preview_loop())
        
        if args.explore:
            insect_controller.client.start_exploration()
            tasks.append(insect_controller.manage_exploration())
        
        if args.detection:
            insect_controller.client.yolo_enabled = True
            tasks.append(run_realtime_detection([], None, None, insect_controller))
        
        if args.discord:
            discord_client = setup_discord(config.discord, insect_controller)
            if discord_client:
                tasks.append(discord_client.start(config.discord.token))
        
        # Run all tasks
        await asyncio.gather(*tasks)
        
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        insect_controller.client.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
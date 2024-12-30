# Modular Robot Control System

This repository contains a modular robot control system for an
insect-like robot. It includes a configurable set of classes, utilities,
and command-line tools for controlling the robot, capturing its video
feed, running YOLO-based object detection, enabling autonomous
exploration, and optionally integrating with a Discord bot for real-time
notifications.

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Requirements](#requirements)
4.  [Installation](#installation)
5.  [Configuration](#configuration)
    -   [Config File
        (`config``.json`)](#config-file-configjson)
    -   [Environment Variables (Discord
        Integration)](#environment-variables-discord-integration)
6.  [Usage](#usage)
    -   [Command-line
        Arguments](#command-line-arguments)
    -   [Keyboard Controls](#keyboard-controls)
7.  [Classes & Modules](#classes--modules)
    -   [InsectClient](#insectclient)
    -   [InsectControl](#insectcontrol)
    -   [ConfigManager](#configmanager)
    -   [KeyboardController](#keyboardcontroller)
    -   [Discord Integration](#discord-integration)
8.  [Autonomous Exploration](#autonomous-exploration)
9.  [YOLO-based Object
    Detection](#yolo-based-object-detection)
10. [Running the Code](#running-the-code)
11. [Contributing](#contributing)
12. [License](#license)

------------------------------------------------------------------------

## 1. Overview

This project provides a control system for a modular insect-like robot.
The robot can be operated either from a command-line interface or
integrated with Discord for real-time updates on detected objects or
events.

Key capabilities include:

-   **Movement and heading control** (forward, backward, turning,
    spinning).
-   **Video capture** from the robot's camera feed.
-   **Video recording** (with on-screen overlays for duration).
-   **Distance sensing** via an ultrasonic sensor (sonar).
-   **Autonomous exploration** and path tracking.
-   **YOLO-based object detection** (local real-time detection).
-   **Discord integration** for posting notifications and receiving
    commands.

------------------------------------------------------------------------

## 2. Features

-   **TCP Connectivity**: Establishes a socket connection for control
    commands and a separate socket for video streaming.
-   **Head/LED Control**: Move the robot's head (pan/tilt) and toggle or
    customize its LED colors.
-   **Servo Power & Balance**: Toggle servo power and balance mode at
    runtime.
-   **Exploration**: Random or semi-intelligent exploration using sonar
    data to avoid obstacles, spin at random intervals, and track visited
    cells in a grid.
-   **Real-time Video**: Streams live video, optionally displayed in an
    `OpenCV` preview window.
-   **Recording**: Capture and save .avi video files, complete with
    overlays for recording status and duration.
-   **Object Detection**: Supports YOLO (Ultralytics) to detect objects
    in the video feed, potentially responding (approach/flee/ignore) to
    recognized objects.
-   **Discord Integration**: Can post detections and robot state updates
    to a specified Discord channel, and can be extended to receive
    commands via Discord.
-   **Configurable**: Includes a JSON-based config manager and
    environment variable overrides for Discord tokens, etc.

------------------------------------------------------------------------

## 3. Requirements

-   **Python 3.8+**
-   **pip** package manager

**Python Dependencies** (typical set):

-   `argparse`
-   `asyncio`
-   `numpy`
-   `opencv-python` (`cv2`)
-   `Pillow` (implied by some image checks, if needed)
-   `discord.py`
-   `python-dotenv`
-   `ultralytics` (for YOLO)
-   `ollama` (for language model interactions, optional)

Make sure you also have the correct YOLO model files in place (e.g.,
`yolov8s`, `yolov8s-worldv2`, or NCNN-converted YOLO models).

------------------------------------------------------------------------

## 4. Installation

1.  **Clone or download** this repository.

2.  **Install dependencies**:

    ``` !overflow-visible
    bashCopy codepip install -r requirements.txt
    ```

3.  **Set up model files**: Ensure your YOLO model files (`.pt` or
    `.onnx` or NCNN variants) are in the correct paths, e.g., `yolov8s`
    in the working directory or update `YOLOSettings` accordingly.

------------------------------------------------------------------------

## 5. Configuration

The system uses both a JSON configuration file and environment variables
(for Discord integration).

### Config File (`config.json`)

A typical `config.json` looks like this:

``` !overflow-visible
jsonCopy code{
  "robot": {
    "ip_address": "10.0.0.250",
    "control_port": 5002,
    "video_port": 8002,
    "move_speed": 8,
    "min_safe_distance": 20.0,
    "exploration_interval": 5.0,
    "exploration_duration": 2.0,
    "head_default_vertical": 90,
    "head_default_horizontal": 90
  },
  "yolo": {
    "base_model": "yolov8s",
    "world_model": "yolov8s-worldv2",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }
}
```

-   `robot.ip_address`: IP of the robot's server.
-   `robot.control_port`: Port for sending control commands (default
    `5002`).
-   `robot.video_port`: Port for receiving video stream (default
    `8002`).
-   `robot.move_speed`: Default movement speed (1-9).
-   `robot.min_safe_distance`: Minimum distance for obstacle avoidance
    (in cm).
-   `robot.exploration_interval`: Time (seconds) between exploration
    moves.
-   `robot.exploration_duration`: Duration (seconds) to move for each
    exploration step.
-   `robot.head_default_vertical` / `head_default_horizontal`: Default
    servo angles for the robot's head.
-   `yolo.base_model`: Name/path of YOLO base model.
-   `yolo.world_model`: Name/path of YOLO \"world\" model.
-   `yolo.confidence_threshold` / `yolo.iou_threshold`: Detection
    thresholds for YOLO.

### Environment Variables (Discord Integration)

In a `.env` file or your environment, set:

``` !overflow-visible
bashCopy codeDISCORD_BOT_TOKEN="YOUR_BOT_TOKEN"
DISCORD_GUILD_ID="123456789012345678"
DISCORD_CHANNEL_ID="123456789012345678"
```

-   `DISCORD_BOT_TOKEN` : Your bot's token from the [Discord Developer
    Portal]{rel="noopener" target="_new"}.
-   `DISCORD_GUILD_ID` : Numeric ID of the server (guild) you want to
    connect to.
-   `DISCORD_CHANNEL_ID` : Numeric ID of the channel in which the bot
    should post messages.

If these variables are missing or invalid, Discord integration will be
skipped.

------------------------------------------------------------------------

## 6. Usage

### Command-line Arguments

``` !overflow-visible
bashCopy codepython main.py [--config CONFIG] [--explore] [--discord] [--no-video] [--detection]
```

  Argument        Description
  --------------- ----------------------------------------------------------
  `--config`      Path to the configuration file (default: `config.json`).
  `--explore`     Enable autonomous exploration mode.
  `--discord`     Enable Discord integration (requires `.env` setup).
  `--no-video`    Disable video feed display.
  `--detection`   Enable YOLO object detection.

### Keyboard Controls

Once the script is running, you can control the robot via the keyboard
(from within the console input or a dedicated keyboard listener).
Typical commands (case-insensitive):

  Key            Action
  -------------- -------------------------
  `w`            Move forward
  `s`            Move backward
  `a`            Move left
  `d`            Move right
  `q`            Spin left
  `e`            Spin right
  `space`        Stop movement
  `t`            Move head up
  `g`            Move head down
  `f`            Move head left
  `h`            Move head right
  `r`            Toggle servo power
  `b`            Toggle balance mode
  `x`            Toggle exploration mode
  `v`            Toggle video recording
  `esc` or `!`   Quit

You can also press `h` (in some modes) to display a help message with
the available commands.

------------------------------------------------------------------------

## 7. Classes & Modules

### InsectClient

-   **Purpose**: Handles low-level robot communication (TCP sockets for
    control & video).
-   **Key Methods**:
    -   `connect(ip, port, video_port)`: Connect to the robot.
    -   `disconnect()`: Disconnect from the robot.
    -   `send_command(command)`: Send a string command to the robot.
    -   `receive_video()`: Threaded function receiving video frames and
        storing them in `self.frame`.
    -   `start_recording()`, `stop_recording()`: Start or stop .avi
        video capture.
    -   `monitor_sonar()`: Continuously requests and updates sonar
        distance readings.
    -   `random_explore()`: Intelligent random movement with path
        tracking.

### InsectControl

-   **Purpose**: High-level control layer that interacts with
    `InsectClient`.
-   **Key Methods**:
    -   `handle_movement(direction)`: Forward/backward/left/right
        movement.
    -   `handle_spin(direction)`: Spin left or right.
    -   `move_head(direction)`: Tilt or pan the robot's head.
    -   `toggle_servo_power()`, `toggle_balance()`, `cycle_led_mode()`:
        Toggle features.
    -   `show_preview()`: Display the current video frame (OpenCV
        window).
    -   `start_exploration()`, `stop_exploration()`: Turn on or off
        random exploration.

### ConfigManager

-   **Purpose**: Loads and saves configuration from a JSON file.
-   **Attributes**:
    -   `robot`: An instance of `RobotSettings`.
    -   `yolo`: An instance of `YOLOSettings`.
    -   `discord`: An instance of `DiscordSettings`.
-   **Key Methods**:
    -   `load_config()`: Reads JSON and populates settings.
    -   `save_config()`: Writes the current in-memory settings to JSON.

### KeyboardController

-   **Purpose**: Handles user input for controlling the robot (e.g.,
    arrow keys, WASD).
-   **Key Methods**:
    -   `handle_keypress(key)`: Dispatch key commands.
    -   `print_help()`: Lists all available keys and actions.
    -   `toggle_video_recording()`: Start or stop .avi recording.
    -   `toggle_exploration()`: Enable or disable the random exploration
        mode.
    -   `quit()`: Graceful shutdown (stop movement, disconnect sockets,
        close windows).

### Discord Integration

-   **Setup**:
    `setup_discord(settings: DiscordSettings, insect_controller: 'InsectControl')`
    initializes the Discord client.
-   **Behavior**: On startup, the bot attempts to join a guild and
    channel as specified by `DISCORD_GUILD_ID` and `DISCORD_CHANNEL_ID`.
-   **Detection**: If YOLO detection is enabled, the bot can post
    messages like \"I just saw `<object>`\" and decide how to respond
    (approach, flee, ignore).

------------------------------------------------------------------------

## 8. Autonomous Exploration

The robot can randomly explore its environment with basic obstacle
avoidance:

-   `extraction_interval`: The robot waits a certain interval before
    performing a random move.
-   `spin_probability`: Chance the robot will spin before moving.
-   `min_safe_distance`: The robot checks sonar data to decide if
    forward movement is blocked.
-   `visited_cells`: Tracks (x, y) grid cells the robot has already
    visited to reduce redundant exploration.

Use:

-   `--explore` at startup
-   Or toggle in real time by pressing `x` to start/stop.

------------------------------------------------------------------------

## 9. YOLO-based Object Detection

-   By default, uses Ultralytics `yolov8s`.
-   For real-time detection:
    -   Enable `--detection` at startup (or set
        `insect_controller.client.yolo_enabled = True`).
    -   A separate async task runs inference on each frame
        (`run_realtime_detection`).
-   **Discord**: If integrated, the system will post messages when new
    objects are detected (once per object name).

------------------------------------------------------------------------

## 10. Running the Code

1.  **Prepare `.env`** (optional, for Discord):

    ``` !overflow-visible
    bashCopy codeDISCORD_BOT_TOKEN="YourDiscordBotToken"
    DISCORD_GUILD_ID="123456789012345678"
    DISCORD_CHANNEL_ID="123456789012345678"
    ```

2.  **Launch**:

    ``` !overflow-visible
    bashCopy codepython main.py --explore --discord --detection
    ```

    (Modify flags based on your needs; see [Command-line
    Arguments](#command-line-arguments){rel="noopener"})

3.  **Control the robot**:

    -   Use the console input commands (e.g. `w` to move forward).
    -   Or in some environments, use the `KeyboardController` to press
        keys in real time.

------------------------------------------------------------------------

## 11. Contributing

We welcome contributions! Feel free to open issues or PRs for
improvements, new features, or bugfixes.\
To contribute:

1.  Fork or clone the repo.
2.  Create a feature branch.
3.  Commit changes with clear messages.
4.  Open a pull request describing your changes.

------------------------------------------------------------------------

## 12. License

This project is made available under an open-source license (MIT or
similar). See [LICENSE]{rel="noopener"} for details.

------------------------------------------------------------------------

### Questions / Contact

For questions, open an issue in this repository or reach out to the main
contributors.

Enjoy exploring and controlling your insect-like robot!

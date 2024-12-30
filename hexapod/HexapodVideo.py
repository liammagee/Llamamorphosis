# -*- coding: utf-8 -*-
import sys
import cv2
import time
import threading
import signal
import socket
import struct
import io
import numpy as np
from PIL import Image
import select

running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("\nTerminating...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class Client:
    def __init__(self):
        self.tcp_flag = False
        self.video_flag = True
        self.image = None

    def turn_on_client(self, ip):
        self.client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip

    def turn_off_client(self):
        try:
            self.client_socket.shutdown(2)
            self.client_socket1.shutdown(2)
            self.client_socket.close()
            self.client_socket1.close()
        except Exception as e:
            print(e)

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

    def receiving_video(self, ip):
        try:
            self.client_socket.connect((ip, 8002))
            self.connection = self.client_socket.makefile('rb')
        except Exception as e:
            print("Video connection failed:", e)
            return
        while True:
            try:
                stream_bytes = self.connection.read(4)
                if len(stream_bytes) < 4:
                    break
                leng = struct.unpack('<L', stream_bytes[:4])[0]
                jpg = self.connection.read(leng)
                if not jpg:
                    break
                if self.is_valid_image_4_bytes(jpg):
                    if self.video_flag:
                        frame = cv2.iamdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        self.image = frame
                        self.video_flag = False  # Indicate frame is being processed
                    self.video_flag = True  # Reset flag for next frame
            except Exception as e:
                print("Video receive error:", e)
                break

    def send_data(self, data):
        if self.tcp_flag:
            try:
                self.client_socket1.send(data.encode('utf-8'))
            except Exception as e:
                print(e)

    def receive_data(self):
        data = ""
        try:
            data = self.client_socket1.recv(1024).decode('utf-8')
        except Exception as e:
            print(e)
        return data

def receive_instructions(client, ip):
    """
    Minimal instruction thread. Just connect and read data.
    """
    try:
        client.client_socket1.connect((ip, 5002))
        client.tcp_flag = True
        print("Instruction connection successful!")
    except Exception as e:
        print("Failed to connect to server instructions:", e)
        client.tcp_flag = False

    while client.tcp_flag:
        try:
            alldata = client.receive_data()
        except Exception as e:
            print("Instruction receive error:", e)
            client.tcp_flag = False
            break

        if alldata == '':
            break
        else:
            cmdArray = alldata.split('\n')
            if cmdArray[-1] != "":
                cmdArray = cmdArray[:-1]
            for oneCmd in cmdArray:
                data = oneCmd.split("#")
                # Just print the received instructions
                print("Received instruction:", data)

def main():
    global running

    # Read IP from IP.txt
    try:
        with open('IP.txt', 'r') as file:
            ip = file.readline().strip()
    except:
        print("Could not read IP from IP.txt.")
        return

    # Initialize client and connect
    client = Client()
    client.turn_on_client(ip)

    # Start threads for video and instructions
    video_thread = threading.Thread(target=client.receiving_video, args=(ip,))
    instruction_thread = threading.Thread(target=receive_instructions, args=(client, ip))

    video_thread.start()
    instruction_thread.start()

    print("Press 'r' to start recording, 's' to stop recording, 'q' to quit.")
    print("Waiting for video frames...")

    record = False
    out = None
    fps = 20
    frame_size_initialized = False

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    while running:
        if client.image is not None:
            cv2.imshow('Video', client.image)

            # If recording is active and writer is ready, write the frame
            if record and out is not None:
                out.write(client.image)

        # Handle keyboard input
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            command = sys.stdin.readline().strip()
            if command == 'r':
                if client.image is None:
                    print("No frames received yet. Please wait until the video feed is active before recording.")
                else:
                    if not record:
                        # Initialize the video writer with the actual frame size
                        height, width = client.image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
                        record = True
                        print(f"Recording started. Video dimensions: {width}x{height}")
                    else:
                        print("Already recording.")
            elif command == 's':
                if record:
                    record = False
                    if out is not None:
                        out.release()
                        out = None
                    print("Recording stopped.")
                else:
                    print("Not currently recording.")
            elif command == 'q':
                running = False
                break
            else:
                print("Unknown command. Use 'r' to record, 's' to stop, 'q' to quit.")

        # Allow OpenCV window to update
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            running = False
            break

        time.sleep(0.05)

    # Cleanup
    if record and out is not None:
        out.release()

    client.tcp_flag = False
    client.turn_off_client()

    cv2.destroyAllWindows()
    print("Disconnected and exited.")

if __name__ == '__main__':
    main()
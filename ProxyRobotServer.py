import io
import socket
import struct
import threading
import time
import cv2
import numpy as np

class ProxyServer:
    def __init__(self):
        self.tcp_flag = False
        self.camera = None
        self.running = True
        
    def turn_on_server(self, host='0.0.0.0'):
        # Video transmission port
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, 8002))
        self.server_socket.listen(1)
        
        # Command reception port
        self.server_socket1 = socket.socket()
        self.server_socket1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket1.bind((host, 5002))
        self.server_socket1.listen(1)
        print(f'Server address: {host}')
        
    def turn_off_server(self):
        self.running = False
        try:
            if self.camera is not None:
                self.camera.release()
            if hasattr(self, 'connection'):
                self.connection.close()
            if hasattr(self, 'connection1'):
                self.connection1.close()
        except Exception as e:
            print(f'Error during shutdown: {e}')
    
    def reset_server(self):
        self.turn_off_server()
        time.sleep(1)  # Give sockets time to close
        self.running = True
        self.turn_on_server()
        self.video = threading.Thread(target=self.transmission_video)
        self.instruction = threading.Thread(target=self.receive_instruction)
        self.video.start()
        self.instruction.start()

    def send_data(self, connect, data):
        try:
            connect.send(data.encode('utf-8'))
            print(f"Sending response: {data}")
        except Exception as e:
            print(f"Error sending data: {e}")

    def transmission_video(self):
        while self.running:
            try:
                print("Waiting for video connection...")
                self.connection, self.client_address = self.server_socket.accept()
                self.connection = self.connection.makefile('wb')
                print("Video socket connected...")
                
                # Initialize webcam
                if self.camera is None:
                    self.camera = cv2.VideoCapture(0)
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
                
                while self.running:
                    ret, frame = self.camera.read()
                    if not ret:
                        print("Failed to grab frame")
                        break
                        
                    try:
                        # Convert frame to JPEG
                        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        jpeg_bytes = jpeg.tobytes()
                        
                        # Send frame size followed by frame data
                        size = len(jpeg_bytes)
                        size_data = struct.pack('<I', size)
                        self.connection.write(size_data)
                        self.connection.write(jpeg_bytes)
                        self.connection.flush()  # Ensure data is sent
                        
                    except (BrokenPipeError, ConnectionResetError) as e:
                        print("Client disconnected, waiting for new connection...")
                        break
                    except Exception as e:
                        print(f"Error in video transmission: {e}")
                        break
                    
                    time.sleep(0.033)  # Limit to ~30 FPS
                    
            except Exception as e:
                print(f"Video connection error: {e}")
            finally:
                try:
                    self.connection.close()
                except:
                    pass
                
            if not self.running:
                break
            
            time.sleep(1)  # Wait before accepting new connections
            
        if self.camera is not None:
            self.camera.release()

    def receive_instruction(self):
        while self.running:
            try:
                print("Waiting for command connection...")
                self.connection1, self.client_address1 = self.server_socket1.accept()
                print("Client connection successful!")
                
                while self.running:
                    try:
                        all_data = self.connection1.recv(1024).decode('utf-8')
                        if not all_data:
                            print("Client disconnected")
                            break
                            
                        cmd_array = all_data.split('\n')
                        print(f"Received commands: {cmd_array}")
                        
                        for one_cmd in cmd_array:
                            if not one_cmd:
                                continue
                                
                            data = one_cmd.split("#")
                            print(f"Processing command: {data}")
                            
                            # Handle specific commands with mock responses
                            if "POWER" in data[0]:
                                command = f"POWER#7.4#7.4\n"
                                self.send_data(self.connection1, command)
                            elif "SONIC" in data[0]:
                                command = f"SONIC#50.0\n"
                                self.send_data(self.connection1, command)
                            else:
                                print(f"Received command: {data}")
                                
                    except (ConnectionResetError, BrokenPipeError):
                        print("Command connection lost")
                        break
                    except Exception as e:
                        print(f"Error receiving data: {e}")
                        break
                        
            except Exception as e:
                print(f"Command socket error: {e}")
            finally:
                try:
                    self.connection1.close()
                except:
                    pass
            
            if not self.running:
                break
                
            time.sleep(1)  # Wait before accepting new connections

def main():
    server = ProxyServer()
    server.turn_on_server()
    server.video = threading.Thread(target=server.transmission_video)
    server.instruction = threading.Thread(target=server.receive_instruction)
    server.video.start()
    server.instruction.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.turn_off_server()

if __name__ == '__main__':
    main()
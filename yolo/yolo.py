from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time
import numpy as np

def test_camera_and_yolo():
    print("Initializing camera...")
    picam2 = Picamera2()
    # Configure for RGB output
    config = picam2.create_preview_configuration(main={"format": 'RGB888'})
    picam2.configure(config)
    picam2.start()
    
    print("Loading YOLO model...")
    #model = YOLO('yolov8m.pt')  # loads the smallest model
    model = YOLO('yolov8m')  # loads the smallest model
    
    print("Capturing frame...")
    # Wait a moment for camera to initialize
    time.sleep(2)
    frame = picam2.capture_array()

    # Save original frame
    cv2.imwrite('original_capture.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("Saved original frame as 'original_capture.jpg'")

    
    print(f"Frame shape: {frame.shape}")  # Let's see what we're getting
    
    print("Running detection...")
    results = model(frame)
    
    print("\nDetections found:")
    for r in results:
        for box in r.boxes:
            # Get class name and confidence
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            print(f"- {class_name}: {confidence:.2f}")
    
    # Clean up
    picam2.stop()
    print("\nTest completed!")

if __name__ == "__main__":
    try:
        test_camera_and_yolo()
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error type: {type(e)}")
from ultralytics import YOLO, YOLOWorld
import cv2
import time
import numpy as np

def run_realtime_detection():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Use default camera
    if not cap.isOpened():
        raise Exception("Could not open camera")
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Loading YOLO model...")
    model = YOLOWorld('yolov8s-worldv2')
    # model.set_classes(['head', 'hair', 'man'])  # Set to detect only glasses
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
            results = model(frame, stream=True)  # stream=True for better performance
            
            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # print(class_name)
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

if __name__ == "__main__":
    try:
        run_realtime_detection()
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error type: {type(e)}")
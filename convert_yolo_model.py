from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # loads the smallest model
model.export(format="ncnn")

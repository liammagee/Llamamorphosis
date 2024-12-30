from ultralytics import YOLO

model = YOLO('yolo11n')  # loads the smallest model
model.export(format="ncnn")

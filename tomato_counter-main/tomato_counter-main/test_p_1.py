from ultralytics import YOLO
model=YOLO('yolov8n.pt')


model.predict(source="testing/t2.avif",save=True)



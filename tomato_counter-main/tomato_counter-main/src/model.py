from ultralytics import YOLO

model=YOLO("yolo11n.pt")
train_results=model.train(
    data="C:/Users/Dileep Sathya/OneDrive/Desktop/obj_det/dataset/data.yaml",
    epochs=10,
    imgsz=640,
)
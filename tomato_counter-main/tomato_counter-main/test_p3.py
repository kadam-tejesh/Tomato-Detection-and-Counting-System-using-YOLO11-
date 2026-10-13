from ultralytics import YOLO

# Load a pre-trained model (e.g., on COCO, but you'll need to fine-tune on tomatoes)
model = YOLO('C:/Users/Dileep Sathya/OneDrive/Desktop/obj_det/runs/detect/train3/weights/best.pt')  # or your custom trained model

# Run inference
results = model('testing/t6.jpg',conf=0.3,save=True)

# Count detections
count = len(results[0].boxes)
print(f"Number of tomatoes: {count}")
from ultralytics import YOLO

# Load a pre-trained model (e.g., on COCO, but you'll need to fine-tune on tomatoes)
model = YOLO('C:/Users/Dileep Sathya/OneDrive/Desktop/obj_det/runs/detect/train3/weights/best.pt')  # or your custom trained model

# Run inference
results = model('C:/Users/Dileep Sathya/OneDrive/Desktop/obj_det/dataset/test/images',conf=0.3,save=True)

count = 0
for r in results:
    count += len(r.boxes)

print(count)

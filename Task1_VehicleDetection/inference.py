from ultralytics import YOLO
from collections import Counter
import os

# Load the trained model
model_path = 'results/car_detection/weights/best.pt'  # Make sure this path is correct
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}")

model = YOLO(model_path)

# Run inference on an input image
results = model.predict(
    source="sample_inputs/image3.png",  # Replace with your test image path
    conf=0.5,                           # Confidence threshold
    save=True,                          # Save annotated image
    save_txt=True,                      # Save results in YOLO txt format
    project="runs/detect",              # Output folder
    name="image_inference",             # Subfolder for this run
    imgsz=640                           # Input image size
)

# Count detections by class
names = model.names  # e.g., {0: 'empty', 1: 'occupied'}

for r in results:
    cls = r.boxes.cls.tolist()
    counts = Counter(cls)
    print("\n--- Detection Summary ---")
    for class_id, count in counts.items():
        label = names.get(int(class_id), f"Class {int(class_id)}")
        print(f"{label}: {count}")

# Optional: Show the output image path
output_dir = os.path.join("runs", "detect", "image_inference")
print(f"\nAnnotated image and results saved to: {output_dir}")

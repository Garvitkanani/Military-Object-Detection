from ultralytics import YOLO
import os

# Load trained model
model = YOLO('models/best.pt')

# Run detection on a folder of images
results = model.predict(
    source='data/test_images/',  # path to test images
    imgsz=640,
    conf=0.25,
    save=True,
    device=0
)

# Save directory printed in results
print(f"Inference results saved to: {results[0].save_dir}")

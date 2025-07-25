🛰️ Military Object Detection with YOLOv8
This project uses a custom-trained YOLOv8 model to detect military-related assets like tanks, aircraft, vehicles, and personnel in aerial or ground images.

📁 Project Structure
military_object_detection/
├── data/
│   └── data.yaml          # Dataset config file (paths to images & labels)
│
├── models/
│   ├── yolov8m.pt        # Base pretrained YOLOv8 model
│   ├── best.pt           # Final trained model weights
│   └── last.pt           # Last epoch checkpoint
│
├── results/
│   ├── metrics.csv       # mAP, precision, recall scores
│   ├── graphs/           # Loss curves, PR, confusion matrix
│   └── predictions/      # Sample validation predictions
│
├── scripts/
│   ├── train.py          # Custom training script
│   ├── val.py            # Re-evaluation script
│   ├── detect.py         # Run inference on test images
│   └── app.py            # Streamlit demo (optional)
│
├── requirements.txt      # Dependencies
├── .gitignore            # Git ignored files
└── README.md             # You're here!


📦 Requirements
Install dependencies using:
pip install -r requirements.txt

Or manually:
pip install ultralytics streamlit opencv-python


🧠 Training
You can train the model using:
yolo detect train \
  model=models/yolov8m.pt \
  data=data/data.yaml \
  epochs=100 imgsz=640 batch=8 \
  device=0 optimizer=AdamW name=military_yolov8m


📊 Validation
yolo val model=models/best.pt data=data/data.yaml


🔍 Inference
Option 1 – CLI Script:
python scripts/detect.py --source path/to/image.jpg

Option 2 – Streamlit Demo (Optional UI):
streamlit run scripts/app.py

Upload an image → view detections → download the output.


📁 Dataset
📌 Due to size, the dataset is not uploaded here.You can download it from:🔗 Kaggle – Military Assets Dataset (12 Classes, YOLOv8 Format)
Make sure to update data/data.yaml after download.


📈 Results (v1.0)
Metric                  Value

mAP50                   54.6%

Best Class              Military Aircraft (85.9% Precision)

Classes                 12 (tanks, aircraft, etc.)

Training Time           ~32 hours (RTX 3050)




🧾 Notes
🔧 Model used: YOLOv8m (Ultralytics)
🧠 Classes: 12 military-related categories
💾 Format: YOLOv8-compatible
🧪 Live testing: Planned for v2.0


✍️ Author
Garvit Kanani
📬 Email: garvitkanani@gmail.com
🔗 LinkedIn: www.linkedin.com/in/garvit-kanani
📁 GitHub: github.com/Garvitkanani


🚀 Version
v1.0: Core model training & detection
v2.0 (Planned): Web app deployment, new dataset classes, model export

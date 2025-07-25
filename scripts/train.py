from ultralytics import YOLO

# Load pretrained model
model = YOLO('models/yolov8m.pt')

# Train the model
model.train(
    data='data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    name='military_yolov8m_best',
    optimizer='AdamW',
    workers=2,
    patience=25,
    verbose=True
)

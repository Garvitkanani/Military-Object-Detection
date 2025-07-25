from ultralytics import YOLO

# Load trained model
model = YOLO('models/best.pt')

# Export to multiple formats (e.g., onnx, torchscript, openvino, etc.)
model.export(format='onnx')  # you can change to: torchscript, openvino, etc.

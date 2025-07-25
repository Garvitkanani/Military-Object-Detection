import streamlit as st
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO

# 👇 Get root directory of the current script
ROOT_DIR = Path(__file__).resolve().parent.parent
model_path = ROOT_DIR / "models" / "best.pt"
model = YOLO(str(model_path))


# Streamlit app config
st.set_page_config(page_title="Military Object Detection", layout="centered")
st.title("🔍 Military Object Detection using YOLOv8")

# Upload an image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save image temporarily for YOLO inference
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    # Inference button
    if st.button("🚀 Detect Objects"):
        with st.spinner("Running detection..."):
            results = model(temp_image_path, conf=0.25)
            result_image = results[0].plot()

            # Create results directory
            output_dir = Path("results/predictions")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save and show the result image
            output_path = output_dir / f"pred_{Path(uploaded_file.name).stem}.jpg"
            Image.fromarray(result_image).save(output_path)

            st.image(result_image, caption="Detection Result", use_container_width=True)

            # Download option
            with open(output_path, "rb") as f:
                st.download_button("📥 Download Result", f, file_name=output_path.name)

        # Clean up temp file
        os.remove(temp_image_path)
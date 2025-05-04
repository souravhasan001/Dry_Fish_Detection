import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import torch

# Set page configuration
st.set_page_config(
    page_title="Dry Fish Detection with XAI",
    page_icon="🐟",
    layout="wide"
)

# Title of the app
st.title("Dry Fish Detection using YOLOv Models + EigenCAM")
st.sidebar.title("⚙️ Settings")

# Model selection dropdown
model_options = {
    "YOLOv9": "yolov9.pt",
    "YOLOv10": "yolov10.pt",
    "YOLOv11": "yolov11.pt",
    "YOLOv12": "yolov12.pt"
}
selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_path = model_options[selected_model_name]

# Load YOLO model with caching
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)
st.success(f"✅ Model `{model_path}` loaded successfully.")

# Draw bounding boxes around detections
# Draw bounding boxes around detections, now showing the predicted fish name
def draw_boxes(image, results, model):
    annotated_img = image.copy()
    names = model.names  # dict: {class_idx: class_name}
    if results and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            fish_name = names.get(cls_idx, "unknown")
            label = f"{fish_name}: {conf:.2f}"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
    return annotated_img


# Image upload section
st.subheader("📷 Upload an Image to Detect Dry Fish")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if st.button("🔍 Detect Dry Fish"):
        with st.spinner("Processing..."):
            try:
                results = model(image_np)
                result_image = draw_boxes(image_np, results[0], model)


                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("Detection Result")
                    st.image(result_image, use_column_width=True)

                count = len(results[0].boxes)
                if count > 0:
                    st.success(f"Detected {count} Dry Fish instance(s).")
                else:
                    st.info("No Dry Fish detected.")

                # ============ 🔍 EigenCAM XAI Visualization ============
                st.subheader("📊 EigenCAM Visualization")

                img_resized = cv2.resize(image_np, (640, 640))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_norm = np.float32(img_resized) / 255.0

                # Select appropriate convolution layer
                target_layers = [model.model.model[-2]]

                cam = EigenCAM(model.model, target_layers)
                grayscale_cam = cam(input_tensor=img_tensor, eigen_smooth=True)[0, :, :]
                cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

                st.image(cam_image, caption="EigenCAM Attention Map", use_column_width=True)
                combined = np.hstack((cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), cam_image))
                st.image(combined, caption="Original + CAM", use_column_width=True)
                # ========================================================

            except Exception as e:
                st.error(f"Error during detection: {e}")

# About section
with st.expander("About this App"):
    st.write("""
    ### Dry Fish Detection App (Image Upload + XAI)
    This app uses YOLOv Models trained for detecting dry fish from images.

    #### Features:
    - Upload an image for dry fish detection
    - Bounding boxes with confidence scores
    - XAI visualization using EigenCAM

    #### How it works:
    The model processes the uploaded image and detects regions containing dry fish. CAMs show model focus areas.

    #### Use cases:
    - Quality control in seafood processing
    - Marine life classification
    - Research and monitoring in fisheries
    """)

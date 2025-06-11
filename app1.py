
import streamlit as st
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import json
import gdown
import os
import torch._classes
import requests

SERVICE_ACCOUNT_PATH = "streamlit_uploader1.json"

@st.cache_resource
def load_drive():
    with open(SERVICE_ACCOUNT_PATH) as source:
        creds_dict = json.load(source)
    creds = service_account.Credentials.from_service_account_info(creds_dict)
    return build("drive", "v3", credentials=creds)

drive_service = load_drive()


# ✅ Your Google Drive folder IDs here:
ORIGINAL_FOLDER_ID = "1nAoIUoP_4V06uMzkL802Zao4xoI6kxU3"
MASK_FOLDER_ID = "1H3jM5blTOzfifEWmGYoL7K3Z263o-mZL"
FINAL_FOLDER_ID = "12H5zu3Gjdh3sGvL_am8A7lEmHVvVOkVD"




@st.cache_resource
def load_model():
    """
    Loads and caches the Mask R-CNN model. Downloads from Hugging Face if not present locally.
    """
    st.info("⏳ Loading model... This may take a moment.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_maskrcnn_model.pth"

    # Hugging Face model download URL
    hf_url = "https://huggingface.co/mahigodike/scratch_detection/resolve/main/best_maskrcnn_model.pth"

    # If model not found locally, download it
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Hugging Face (one-time operation)...")

        try:
            hf_token = "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"  # 🔐 Replace with your real token or use secrets
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.get(hf_url, headers=headers)

            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                st.success("✅ Model downloaded successfully.")
            else:
                st.error(f"Failed to download model. Status code: {response.status_code}")
                return None, None

        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None, None

    # Load the model architecture
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=True)

    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    st.success("✅ Model loaded and ready!")

    return model, device

model, device = load_model()

def predict_image(image, threshold=0.2):
    image_tensor = F.to_tensor(image).to(device)
    original = np.array(image)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes, masks, scores = prediction['boxes'], prediction['masks'], prediction['scores']
    final_masks = []
    valid_scores = []

    for i in range(len(scores)):
        if scores[i] > threshold:
            mask = masks[i, 0].cpu().numpy()
            final_masks.append(mask)
            valid_scores.append(float(scores[i]))

    return original, final_masks, valid_scores

def upload_to_drive(image: Image.Image, folder_id, filename):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    media = MediaIoBaseUpload(buffer, mimetype='image/jpeg')
    drive_service.files().create(body={"name": filename, "parents": [folder_id]}, media_body=media).execute()

def create_mask_overlay(original_img, masks, scores):
    h, w = original_img.shape[:2]
    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    total_pixels = 0

    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8)
        red = np.zeros_like(mask_overlay)
        red[:, :, 2] = binary * 255
        mask_overlay = cv2.addWeighted(mask_overlay, 1.0, red, 0.5, 0)
        total_pixels += np.sum(binary)

    overlayed = cv2.addWeighted(original_img, 1.0, mask_overlay, 0.6, 0)

    # Single bounding box
    all_y, all_x = np.where(np.sum(masks, axis=0) > 0.5)
    if len(all_x) > 0 and len(all_y) > 0:
        x1, y1, x2, y2 = int(np.min(all_x)), int(np.min(all_y)), int(np.max(all_x)), int(np.max(all_y))
        cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 5)

    severity = (total_pixels / (h * w)) * 100
    confidence = np.mean(scores) * 100

    text = f"Severity: {severity:.1f}% | Confidence: {confidence:.1f}% "
    text1=f" Mask Pixels: {total_pixels}"
    cv2.putText(overlayed, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 3)
    cv2.putText(overlayed, text1, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 3)


    return Image.fromarray(mask_overlay), Image.fromarray(overlayed)

# Streamlit UI
st.title("🚗 Vehicle Scratch Detection - PRO Version")

uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    st.image(image, caption="Original", use_column_width=True)

    if st.button("Run Detection"):
        original_name = f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        upload_to_drive(image, ORIGINAL_FOLDER_ID, original_name)

        with st.spinner("Processing..."):
            original_np, masks, scores = predict_image(image)

        if masks:
            mask_img, overlay_img = create_mask_overlay(original_np, masks, scores)

            col1, col2 = st.columns(2)
            col1.image(mask_img, caption="Scratch Mask", use_column_width=True)
            col2.image(overlay_img, caption="Result Overlay", use_column_width=True)

            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            upload_to_drive(mask_img, MASK_FOLDER_ID, f"mask_{now}.jpg")
            upload_to_drive(overlay_img, FINAL_FOLDER_ID, f"overlay_{now}.jpg")
            st.success("✅ All images uploaded to Google Drive.")
        else:
            st.warning("No scratches detected.")


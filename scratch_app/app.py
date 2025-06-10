import streamlit as st

# First command
st.set_page_config(page_title="🚗 Scratch Detection App", layout="wide")

# Imports
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import cv2
import os
import tempfile
import gdown

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

# =============================
# Load Model from Google Drive
# =============================
@st.cache_resource
def load_model():
    model_path = "best_m_model.pth"
    if not os.path.exists(model_path):
        # Replace this ID with your actual Google Drive file ID
        file_id = "1LC5YJ_nigk0u8YOVgYvjjGCtagKTnvT_"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 2)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(256, 256, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# =============================
# Upload to Google Drive
# =============================
@st.cache_resource
def connect_service_drive():
    scope = ['https://www.googleapis.com/auth/drive']

    creds_json_str = st.secrets["gdrive_service_account"]["json"]
    creds_dict = json.loads(creds_json_str)

    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

    gauth = GoogleAuth()
    gauth.credentials = creds
    return GoogleDrive(gauth)

def upload_to_drive(file_path, filename, folder_id=None):
    file_drive = drive.CreateFile({
        'title': filename,
        'parents': [{"kind": "drive#fileLink", "id": folder_id}] if folder_id else []
    })
    file_drive.SetContentFile(file_path)
    file_drive.Upload()

# =============================
# Prediction Function
# =============================
def predict_image(image, threshold=0.5):
    original = np.array(image.convert("RGB"))
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes = prediction['boxes']
    scores = prediction['scores']
    masks = prediction['masks']

    final_boxes, final_masks, severities, final_scores = [], [], [], []

    for i in range(len(scores)):
        if scores[i] > threshold:
            final_boxes.append(boxes[i].cpu().numpy())
            mask_np = masks[i, 0].cpu().numpy()
            final_masks.append(mask_np)

            mask_area = np.sum(mask_np > 0.5)
            total_area = original.shape[0] * original.shape[1]
            severities.append((mask_area / total_area) * 100)
            final_scores.append(scores[i].item())

    return original, final_boxes, final_masks, severities, final_scores

# =============================
# Visualization Function
# =============================
def visualize_results(image, masks, severities, scores):
    overlay_img = image.copy()
    mask_only = np.zeros_like(image)
    all_mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = np.array([0, 0, 255])
        mask_bin = (mask > 0.5).astype(np.uint8)
        mask_rgb = np.stack([mask_bin * c for c in color], axis=-1)

        mask_only = np.where(mask_bin[..., None], mask_rgb, mask_only)
        overlay_img = np.where(mask_bin[..., None], 0.6 * overlay_img + 0.4 * mask_rgb, overlay_img)
        all_mask_combined = np.logical_or(all_mask_combined, mask_bin).astype(np.uint8)

    ys, xs = np.where(all_mask_combined)
    if len(xs) > 0 and len(ys) > 0:
        x1, y1, x2, y2 = int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))
        width = x2 - x1
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(overlay_img, f"Width: {width}px", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    overall = sum(severities)
    avg_conf = np.mean(scores) * 100 if scores else 0.0

    cv2.putText(overlay_img, f"Overall Severity: {overall:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(overlay_img, f"Avg Confidence: {avg_conf:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return overlay_img.astype(np.uint8), mask_only.astype(np.uint8)

# =============================
# Streamlit UI
# =============================
st.title("🚗 Scratch Detection App using Mask R-CNN")

st.markdown("""
Upload a vehicle surface image to detect scratches and calculate severity level using deep learning.
""")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
threshold = st.slider("🔧 Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("Processing image..."):
            image_np, boxes, masks, severities, scores = predict_image(img, threshold=threshold)
            overlay, mask_only = visualize_results(image_np, masks, severities, scores)

            col1, col2 = st.columns(2)
            col1.image(mask_only, caption="🟥 Scratch Mask", use_column_width=True)
            col2.image(overlay, caption="📊 Detection Results", use_column_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                cv2.imwrite(tmpfile.name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                st.success("✅ Detection Complete!")

                # Upload to Google Drive (set your folder ID)
                upload_to_drive(tmpfile.name, uploaded_file.name, folder_id="your_google_drive_folder_id")

                with open(tmpfile.name, "rb") as f:
                    st.download_button("📥 Download Result Image", f, file_name="scratch_detection_result.jpg", mime="image/jpeg")

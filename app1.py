import pandas as pd
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
import time
from pytz import timezone

@st.cache_resource
def load_drive():
    # Load service account info from Streamlit secrets
    creds_dict = {
        "type": st.secrets["gcp_service_account"]["type"],
        "project_id": st.secrets["gcp_service_account"]["project_id"],
        "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
        "private_key": st.secrets["gcp_service_account"]["private_key"],
        "client_email": st.secrets["gcp_service_account"]["client_email"],
        "client_id": st.secrets["gcp_service_account"]["client_id"],
        "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
        "token_uri": st.secrets["gcp_service_account"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
        "universe_domain": st.secrets["gcp_service_account"]["universe_domain"]
    }

    creds = service_account.Credentials.from_service_account_info(creds_dict)
    return build("drive", "v3", credentials=creds)

# Call the function to create drive service
drive_service = load_drive()


# ✅ Your Google Drive folder IDs here:
ORIGINAL_FOLDER_ID = "1nAoIUoP_4V06uMzkL802Zao4xoI6kxU3"
MASK_FOLDER_ID = "1H3jM5blTOzfifEWmGYoL7K3Z263o-mZL"
FINAL_FOLDER_ID = "12H5zu3Gjdh3sGvL_am8A7lEmHVvVOkVD"

def save_data_to_csv_drive(filename, severity, confidence, pixels, folder_id, csv_name="scratch_data.csv"):
    # Step 1: Search for existing CSV file in Drive
    response = drive_service.files().list(
        q=f"name='{csv_name}' and '{folder_id}' in parents and mimeType='text/csv'",
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    files = response.get('files', [])
    
    file_id = files[0]['id'] if files else None

    # Step 2: Load existing data if file exists
    if file_id:
        request = drive_service.files().get_media(fileId=file_id)
        content = request.execute()
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.DataFrame(columns=["filename", "severity", "confidence", "pixels", "timestamp"])

    # Step 3: Append new row
    new_row = {
        "filename": filename,
        "severity": severity if severity != "none" else None,
        "confidence": confidence if confidence != "none" else None,
        "pixels": pixels if pixels != "none" else None,
        "timestamp": datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Step 4: Save and upload to Drive
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    media = MediaIoBaseUpload(buffer, mimetype='text/csv')

    if file_id:
        drive_service.files().update(fileId=file_id, media_body=media).execute()
    else:
        drive_service.files().create(
            body={"name": csv_name, "parents": [folder_id]},
            media_body=media
        ).execute()





@st.cache_resource
def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_maskrcnn_model.pth"

    # Hugging Face model download URL
    hf_url = "https://huggingface.co/babbilibhavani/scartch_detection/resolve/main/best_maskrcnn_model_k.pth"

    # If model not found locally, download it
    if not os.path.exists(model_path):
        

        try:
            hf_token = "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"  # 🔐 Replace with your real token or use secrets
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.get(hf_url, headers=headers)

            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                
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
        cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 3)

    severity = (total_pixels / (h * w)) * 100
    confidence = np.mean(scores) * 100
    h,w=overlayed.shape[:2]
    font_scale=w/1000
    font_thickness=int(w/1000)
    

    text = f"Severity: {severity:.1f}% | Confidence: {confidence:.1f}% "
    text1=f" Mask Pixels: {total_pixels}"
    cv2.putText(overlayed, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    cv2.putText(overlayed, text1, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255, 0), 2)


    return Image.fromarray(mask_overlay), Image.fromarray(overlayed), severity, total_pixels,confidence
# Streamlit UI
st.title("🚗 Vehicle Scratch Detection System")

uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    st.image(image, caption="Original", use_container_width=True)

    if st.button("Run Detection"):
        original_name = f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        upload_to_drive(image, ORIGINAL_FOLDER_ID, original_name)

        with st.spinner("Detecting Scratches..."):
            start_time = time.time()
            original_np, masks, scores = predict_image(image)
            end_time = time.time()
            detection_time = end_time - start_time


        if masks:
            mask_img, overlay_img,severity,total_pixels,confidence = create_mask_overlay(original_np, masks, scores)
            st.subheader("Results:")
            st.write(f"🕒shown in: {detection_time:.2f} seconds")
            col1, col2 = st.columns(2)
            col1.image(mask_img, caption="Scratch Mask", use_container_width=True)
            col2.image(overlay_img, caption="Result Overlay", use_container_width=True)
            st.subheader("Scratches Detected:")
            st.write(f"sevierity:{severity:.1f}%") 
            st.write(f"Pixels:{total_pixels}pxs")

           
            ist = timezone('Asia/Kolkata')
            now = datetime.now(ist).strftime('%Y%m%d_%H%M%S')

            upload_to_drive(mask_img, MASK_FOLDER_ID, f"mask_{now}.jpg")
            upload_to_drive(overlay_img, FINAL_FOLDER_ID, f"overlay_{now}.jpg")
            st.success("✅ All images saved sucessfully.")
            st.write("_____________________________________")
            st.markdown("""
            <span style='color: white;'>___________________@</span>
            <span style='color: orange; font-weight: bold;'>Valkontek Embedded Services</span>
            """, unsafe_allow_html=True)
            save_data_to_csv_drive(filename= f"overlay_{now}.jpg", severity=severity, confidence=confidence, pixels=total_pixels, folder_id="1QIIdYHFt-iWAd-jvE1KAESEzBjWkb3gv")
        
        else:
            st.warning("No scratches detected.")
            ist = timezone('Asia/Kolkata')
            now = datetime.now(ist).strftime('%Y%m%d_%H%M%S')
            save_data_to_csv_drive(filename= f"overlay_{now}.jpg", severity="none", confidence="none", pixels="none", folder_id="1QIIdYHFt-iWAd-jvE1KAESEzBjWkb3gv")
        
            


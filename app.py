import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from pygame import mixer
import time
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Driver Safety System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR "COOL" UI ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .safe { background-color: #00CC96; color: white; }
    .danger { background-color: #FF4B4B; color: white; }
    .warning { background-color: #FFA15A; color: white; }
</style>
""", unsafe_allow_html=True)

# --- MODEL DEFINITION (Must match training) ---
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- INITIALIZATION & CACHING ---
@st.cache_resource
def load_resources():
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrowsinessCNN().to(device)
    try:
        model.load_state_dict(torch.load('models\\drowsiness_model.pth', map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Model not found! Error: {e}")
        return None, None, None

    # Load Sound
    mixer.init()
    try:
        sound = mixer.Sound('alarm.wav')
    except:
        sound = None
        
    # Load Face Detector
    haar_path = 'haar cascade files\\haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(haar_path)
    
    return model, sound, face_cascade, device

model, sound, face_cascade, device = load_resources()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((145, 145)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

LABELS = ['Open', 'yawn', 'Closed', 'no_yawn']

# --- SIDEBAR SETTINGS ---
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Configure the sensitivity of the system.")

ALARM_THRESHOLD = st.sidebar.slider("Alarm Threshold (Frames)", 5, 50, 15)
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold (%)", 50, 100, 70)
ENABLE_SOUND = st.sidebar.checkbox("Enable Audio Alarm", value=True)
CAMERA_SOURCE = st.sidebar.selectbox("Select Camera", [0, 1])

st.sidebar.markdown("---")
st.sidebar.info("Developed with PyTorch & Streamlit")

# --- MAIN UI LAYOUT ---
st.markdown('<p class="main-header">🚗 Driver Drowsiness Detection System</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Live Feed")
    video_placeholder = st.empty()
    camera_status = st.empty()

with col2:
    st.markdown("### Real-Time Analytics")
    status_placeholder = st.empty()
    score_metric = st.empty()
    chart_placeholder = st.empty()
    
    # Start/Stop Button
    run = st.checkbox('Start Monitoring', value=False)

# --- APP LOGIC ---
if run:
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    score = 0
    alarm_on = False
    
    # For chart data
    chart_data = []
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not access camera.")
            break
            
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        status_text = "Active"
        color_status = (0, 255, 0)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
            
            roi_color = frame[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(roi_rgb)
            
            try:
                input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    class_idx = pred.item()
                    
                # Logic: 0=Open, 1=Yawn, 2=Closed, 3=No Yawn
                if (class_idx == 1 or class_idx == 2) and (conf.item() * 100 > CONFIDENCE_THRESHOLD):
                    score += 1
                    status_text = "DROWSY"
                    color_status = (255, 0, 0)
                else:
                    score -= 1
                    status_text = "Active"
                    color_status = (0, 255, 0)
                    
                if score < 0: score = 0
                if score > 50: score = 50
                
                # Display status on frame
                cv2.putText(frame, f"{LABELS[class_idx]} ({int(conf*100)}%)", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
                           
            except Exception as e:
                pass

        # Alarm Logic
        if score > ALARM_THRESHOLD:
            cv2.putText(frame, "WAKE UP!", (width//2-100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            status_text = "DANGER!"
            if ENABLE_SOUND and sound and not alarm_on:
                sound.play(-1)
                alarm_on = True
        else:
            if alarm_on:
                sound.stop()
                alarm_on = False

        # --- UPDATE UI ---
        
        # 1. Update Video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # 2. Update Status Box
        if score > ALARM_THRESHOLD:
            status_html = f'<div class="status-card danger">🚨 WAKE UP! 🚨</div>'
        elif score > ALARM_THRESHOLD / 2:
            status_html = f'<div class="status-card warning">😴 Drowsy</div>'
        else:
            status_html = f'<div class="status-card safe">😊 Active</div>'
        status_placeholder.markdown(status_html, unsafe_allow_html=True)
        
        # 3. Update Metric
        score_metric.metric("Fatigue Score", score, delta=score-ALARM_THRESHOLD, delta_color="inverse")
        
        # 4. Update Chart (Rolling Window)
        chart_data.append(score)
        if len(chart_data) > 50: chart_data.pop(0)
        chart_placeholder.line_chart(chart_data)

        # Stop button logic handled by Streamlit checkbox 'run' state
        
    # Cleanup when loop ends
    cap.release()
    if alarm_on: sound.stop()
    cv2.destroyAllWindows()
else:
    st.info("Check 'Start Monitoring' to begin.")
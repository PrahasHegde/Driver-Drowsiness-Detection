import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pygame import mixer

# --- CONFIGURATION ---
ALARM_SOUND_PATH = 'alarm.wav'
MODEL_PATH = 'models\\drowsiness_model.pth'
Haar_PATH = 'haar cascade files\\haarcascade_frontalface_alt.xml'

# Updated Labels based on your testing
# 0: Open, 1: Yawn, 2: Closed, 3: No Yawn
LABELS = ['Open', 'yawn', 'Closed', 'no_yawn']

# --- MODEL DEFINITION ---
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

# --- INITIALIZATION ---
mixer.init()
try:
    sound = mixer.Sound(ALARM_SOUND_PATH)
except:
    print("Warning: alarm.wav not found.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrowsinessCNN().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("PyTorch Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

preprocess = transforms.Compose([
    transforms.Resize((145, 145)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

face_cascade = cv2.CascadeClassifier(Haar_PATH)
cap = cv2.VideoCapture(0)

score = 0
alarm_on = False

print("System Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw black background for score
    cv2.rectangle(frame, (0, height-50), (width, height), (0, 0, 0), -1)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
        
        roi_color = frame[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        
        try:
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                class_index = predicted.item()
                state = LABELS[class_index]

            # --- UPDATED LOGIC HERE ---
            # 0=Open (Active), 1=Yawn (Drowsy), 2=Closed (Drowsy), 3=No Yawn (Active)
            
            if class_index == 1 or class_index == 2:
                score += 1
                color = (0, 0, 255) # Red for danger
                text_status = "DROWSY"
            else:
                score -= 1
                color = (0, 255, 0) # Green for safe
                text_status = "ACTIVE"
                
            # Cap the score so it doesn't get too high or low
            if score < 0: score = 0
            if score > 40: score = 40
            
            # Show status above face
            cv2.putText(frame, f"{state}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        except Exception as e:
            print(f"Error: {e}")

    # --- ALARM LOGIC ---
    # Trigger if score > 15 (Adjust this number if it's too sensitive)
    if score > 15:
        cv2.putText(frame, "WAKE UP!", (width//2-100, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        if not alarm_on:
            try:
                sound.play(-1)
                alarm_on = True
            except:
                pass
    else:
        if alarm_on:
            sound.stop()
            alarm_on = False

    # Display Score
    cv2.putText(frame, f'Fatigue Score: {score}', (10, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Driver Drowsiness System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
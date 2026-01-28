import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = 'models\\drowsiness_model.pth'
TEST_PATH = 'dataset_new\\test' 
BATCH_SIZE = 32

# 1. SETUP DEVICE & MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the exact same model architecture
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

# Load Model
model = DrowsinessCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Important: set to eval mode for testing!

# 2. PREPARE TEST DATA
transform = transforms.Compose([
    transforms.Resize((145, 145)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Loading Test Data...")
test_data = datasets.ImageFolder(root=TEST_PATH, transform=transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Get class names
class_names = test_data.classes
print(f"Classes: {class_names}")

# 3. RUN PREDICTIONS
all_preds = []
all_labels = []

print("Running Evaluation...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 4. GENERATE METRICS

# A. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# B. Classification Report (Precision, Recall, F1)
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

# C. Class-wise Accuracy Bar Chart
# Calculate accuracy per class
class_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(8, 5))
plt.bar(class_names, class_accuracy, color=['blue', 'orange', 'green', 'red'])
plt.ylabel('Accuracy')
plt.title('Accuracy per Class')
plt.ylim(0, 1)
for i, v in enumerate(class_accuracy):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()
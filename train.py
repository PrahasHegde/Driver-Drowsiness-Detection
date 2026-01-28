import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# --- CONFIGURATION ---
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = (145, 145)
TRAIN_PATH = 'dataset_new\\train'
TEST_PATH = 'dataset_new\\test'
MODEL_SAVE_PATH = 'models\\drowsiness_model.pth'

# 1. DEFINE TRANSFORMS (Preprocessing)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(), # Converts image to (C, H, W) and scales to 0-1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to range [-1, 1]
])

# 2. LOAD DATASET
# PyTorch's ImageFolder expects structure: dataset/class_name/image.jpg
train_data = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
test_data = datasets.ImageFolder(root=TEST_PATH, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes found: {train_data.class_to_idx}")
# Expected: {'Closed': 0, 'No_Yawn': 1, 'Open': 2, 'Yawn': 3} (Check output!)

# 3. DEFINE THE CNN MODEL
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Fully Connected Layers
        # Calculation for input size to linear layer:
        # 145 -> pool -> 72 -> pool -> 36 -> pool -> 18
        # Final feature map size: 128 channels * 18 * 18
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4) # 4 Output Classes

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 18 * 18) # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. TRAINING LOOP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = DrowsinessCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# 5. SAVE MODEL
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
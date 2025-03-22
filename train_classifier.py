import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DDPMScheduler
from sklearn.metrics import accuracy_score
from classifier import SimpleCNN

# ======================= CONFIG =======================
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
MODEL_PATH = './models/classifier.pth'
NUM_CLASSES = 10
TIME_EMBEDDING_DIM = 128
NUM_TIMESTEPS = 1000
IMAGE_SIZE = 32

# ===================== DEVICE =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== TRANSFORMS =====================
classifier_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===================== DATASETS =======================
train_data = datasets.MNIST(root='data', train=True, download=True, transform=classifier_transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=classifier_transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ===================== TRAIN ==========================
def train():
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS, beta_schedule="linear")

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            timesteps = torch.randint(0, NUM_TIMESTEPS, (data.size(0),), device=device).long()
            noise = torch.randn_like(data).to(device)
            noisy_images = noise_scheduler.add_noise(data, noise, timesteps)
            outputs = model(noisy_images, timesteps)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.2f}%")

    end_time = time.time()
    print(f"Training complete in {(end_time - start_time) / 60:.2f} minutes.")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ===================== TEST ===========================
def test():
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            timesteps = torch.zeros(data.shape[0], dtype=torch.long, device=device)
            outputs = model(data, timesteps)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

train()
test()
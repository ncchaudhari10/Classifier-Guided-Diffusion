import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from diffusers import DDPMScheduler, UNet2DModel
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
IMAGE_SIZE = 32
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_TIMESTEPS = 1000
MODEL_PATH = './models/ddpm_unet.pth'
SAMPLE_OUTPUT = './sample_ddpm_output.png'

# ===================== DEVICE ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== DATASET =======================
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 2) - 1)  # Scale to [-1, 1]
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================== MODEL =========================
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512)
).to(device)

scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS, beta_schedule="linear")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# ===================== TRAIN ==========================
model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        noise = torch.randn_like(images).to(device)
        timesteps = torch.randint(0, NUM_TIMESTEPS, (images.shape[0],), device=device).long()
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps).sample
        loss = criterion(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# =================== INFERENCE ========================
# Load model and generate one sample
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    num_samples = 8
    samples = torch.randn((num_samples, 1, IMAGE_SIZE, IMAGE_SIZE)).to(device)

    for t in range(NUM_TIMESTEPS - 1, -1, -1):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(samples, t_batch).sample
        samples = scheduler.step(noise_pred, t, samples).prev_sample

    samples = (samples.clamp(-1, 1) + 1) / 2
    samples = samples.cpu().numpy() * 255.0

    # Plot and save output image
    plt.figure(figsize=(12, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(samples[i, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(SAMPLE_OUTPUT)
    plt.close()
    print(f"Sample image saved to {SAMPLE_OUTPUT}")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DModel
from classifier import SimpleCNN

# ======================= CONFIG =======================
IMAGE_SIZE = 32
NUM_TIMESTEPS = 1000
NUM_SAMPLES = 8
TARGET_CLASS = 3
GUIDANCE_SCALE = 15.0
CLASSIFIER_PATH = './models/classifier.pth'
DDPM_PATH = './models/ddpm_unet.pth'
OUTPUT_IMAGE_PATH = './guided_output.png'


# ===================== DEVICE =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== LOAD MODELS =====================
# Classifier
classifier = SimpleCNN()
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device,weights_only=True))
classifier.to(device)
classifier.eval()

# DDPM UNet
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512)
)
model.load_state_dict(torch.load(DDPM_PATH, map_location=device,weights_only=True))
model.to(device)
model.eval()

# Scheduler
scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS, beta_schedule="linear")

# ============ CLASSIFIER-GUIDED SAMPLING FUNCTION ============
def classifier_guided_sampling(model, scheduler, classifier, num_samples, guidance_scale, target_class, image_size):
    model.eval()
    classifier.eval()
    with torch.no_grad():
        samples = torch.randn((num_samples, 1, image_size, image_size)).to(device)

        for t in range(scheduler.config.num_train_timesteps - 1, -1, -1):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(samples, timesteps).sample

            if t > 0:
                samples.requires_grad_(True)
                with torch.enable_grad():
                    logits = classifier(samples, timesteps)
                    log_prob = F.log_softmax(logits, dim=1)[:, target_class].sum()
                    grad = torch.autograd.grad(log_prob, samples)[0]

                noise_pred = noise_pred - guidance_scale * grad
                samples = samples.detach()

            samples = scheduler.step(noise_pred, t, samples).prev_sample

        samples = (samples.clamp(-1, 1) + 1) / 2
        samples = samples.cpu().numpy() * 255.0

    return samples

# ===================== RUN INFERENCE ====================
print("Running classifier-guided sampling...")
images = classifier_guided_sampling(
    model, scheduler, classifier,
    num_samples=NUM_SAMPLES,
    guidance_scale=GUIDANCE_SCALE,
    target_class=TARGET_CLASS,
    image_size=IMAGE_SIZE
)

# ===================== SAVE IMAGES ======================
def plot_images(images, output_path, nrow=4):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for i, img in enumerate(images):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved guided samples to {output_path}")

plot_images(images, OUTPUT_IMAGE_PATH)

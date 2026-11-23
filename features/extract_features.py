import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pretrained Frozen ResNet-18
weights = ResNet18_Weights.IMAGENET1K_V1
base_model = resnet18(weights=weights)
encoder = nn.Sequential(*list(base_model.children())[:-1])
encoder = encoder.to(device).half().eval()
for param in encoder.parameters():
    param.requires_grad = False

CROP_SIZE = 41
HALF_CROP = 20
RESIZE_TO = 224

# Normalization constants on GPU
norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
norm_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)

def extract_crops_and_embed(img_rgb, centroids, batch_size=512):
    """
    Extracts 41x41 crops, resizes to 224x224 via GPU bilinear interpolation,
    and extracts 512-dim FP16 embeddings with frozen ResNet-18.
    """
    N = len(centroids)
    if N == 0:
        return np.empty((0, 512), dtype=np.float16)

    pad = HALF_CROP + 5
    img_padded = cv2.copyMakeBorder(img_rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    all_embeddings = []
    
    # Process in batches
    for i in range(0, N, batch_size):
        batch_centroids = centroids[i:i+batch_size]
        batch_crops = []
        
        for c in batch_centroids:
            x, y = int(round(c[0])), int(round(c[1]))
            x_p, y_p = x + pad, y + pad
            crop = img_padded[y_p - HALF_CROP : y_p + HALF_CROP + 1,
                              x_p - HALF_CROP : x_p + HALF_CROP + 1]
            if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            batch_crops.append(crop)
            
        # Fast transfer to GPU
        batch_np = np.stack(batch_crops, axis=0).astype(np.float16) / 255.0
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device) # (B, 3, 41, 41)
        
        # GPU Bilinear Resize to 224x224
        batch_resized = F.interpolate(batch_tensor, size=(RESIZE_TO, RESIZE_TO), mode='bilinear', align_corners=False)
        # Normalization
        batch_norm = (batch_resized - norm_mean) / norm_std
        
        with torch.no_grad():
            emb = encoder(batch_norm).squeeze(-1).squeeze(-1) # (B, 512)
            all_embeddings.append(emb.cpu().numpy().astype(np.float16))
            
    return np.vstack(all_embeddings)

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from metrics.evaluation import compute_macro_f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LIZARD_MAP = {2: 0, 1: 1, 3: 1, 4: 1, 5: 1, 6: 2}

class FastCropDataset(Dataset):
    def __init__(self, crops, labels, is_train=True):
        self.crops = crops # (N, 41, 41, 3) uint8
        self.labels = labels # (N,) int64
        self.is_train = is_train

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx] # (41, 41, 3)
        if self.is_train:
            if np.random.rand() > 0.5:
                crop = np.fliplr(crop)
            if np.random.rand() > 0.5:
                crop = np.flipud(crop)
            k_rot = np.random.choice([0, 1, 2, 3])
            if k_rot > 0:
                crop = np.rot90(crop, k_rot)

        # Convert to tensor (3, 41, 41) float32 in [0, 1]
        t = torch.from_numpy(crop.copy().astype(np.float32) / 255.0).permute(2, 0, 1)
        return t, torch.tensor(self.labels[idx], dtype=torch.long)

def build_crop_pool(items, max_samples=25000, seed=42):
    print(f"Building crop pool (target: {max_samples})...", flush=True)
    all_records = []
    
    # 1. Collect candidate coordinates
    for it in items:
        img_p = it['ImagePath']
        mat_p = it['LabelPath']
        mat = sio.loadmat(mat_p)
        classes = mat['class'].flatten()
        centroids = mat['centroid']

        for c, (x, y) in zip(classes, centroids):
            lbl = LIZARD_MAP.get(int(c), -1)
            if lbl >= 0:
                all_records.append((img_p, float(x), float(y), lbl))

    # Stratified subsampling
    df = pd.DataFrame(all_records, columns=['img_path', 'x', 'y', 'label'])
    if len(df) > max_samples:
        per_class = max_samples // 3
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), per_class), random_state=seed)
        ).reset_index(drop=True)

    print(f"Sampling {len(df)} crops across images...", flush=True)

    # 2. Extract 41x41 crops per image
    grouped = df.groupby('img_path')
    crops_list = []
    labels_list = []

    HALF_CROP = 20
    CROP_SIZE = 41
    pad = 25

    for img_p, grp in grouped:
        im = cv2.imread(img_p)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_padded = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        xs = grp['x'].values
        ys = grp['y'].values
        lbls = grp['label'].values

        for x, y, lbl in zip(xs, ys, lbls):
            xp, yp = int(round(x)) + pad, int(round(y)) + pad
            cr = im_padded[yp - HALF_CROP : yp + HALF_CROP + 1, xp - HALF_CROP : xp + HALF_CROP + 1]
            if cr.shape[0] != CROP_SIZE or cr.shape[1] != CROP_SIZE:
                cr = cv2.resize(cr, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            crops_list.append(cr)
            labels_list.append(lbl)

    crops_arr = np.stack(crops_list, axis=0) # (N, 41, 41, 3) uint8
    labels_arr = np.array(labels_list, dtype=np.int64)
    print(f"Extracted {len(crops_arr)} crops, size: {crops_arr.nbytes / (1024*1024):.1f} MB", flush=True)
    return crops_arr, labels_arr

def train_m1_model(seed=42, max_epochs=8, lr=1e-4, batch_size=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    with open('data/splits/splits.json') as f:
        splits = json.load(f)

    train_items = splits['details']['train']
    val_items = splits['details']['val']

    train_crops, train_lbls = build_crop_pool(train_items, max_samples=25000, seed=seed)
    val_crops, val_lbls = build_crop_pool(val_items, max_samples=4000, seed=seed)

    train_ds = FastCropDataset(train_crops, train_lbls, is_train=True)
    val_ds = FastCropDataset(val_crops, val_lbls, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ResNet-18 model
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, 3)
    model = model.to(device)

    # Class weights
    counts = np.bincount(train_lbls, minlength=3)
    c_weights = len(train_lbls) / (3.0 * counts.astype(np.float32))
    class_weights = torch.tensor(c_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # GPU normalization constants
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    best_val_f1 = -1.0
    best_weights = None
    best_epoch = 0

    print(f"\n[M1 ResNet-18 Seed {seed}] Starting training for {max_epochs} epochs...", flush=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for crops, targets in train_loader:
            crops = crops.to(device) # (B, 3, 41, 41)
            targets = targets.to(device)

            # GPU Bilinear resize to 224x224
            crops_224 = F.interpolate(crops, size=(224, 224), mode='bilinear', align_corners=False)
            crops_norm = (crops_224 - norm_mean) / norm_std

            optimizer.zero_grad()
            logits = model(crops_norm)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for crops, targets in val_loader:
                crops = crops.to(device)
                crops_224 = F.interpolate(crops, size=(224, 224), mode='bilinear', align_corners=False)
                crops_norm = (crops_224 - norm_mean) / norm_std

                logits = model(crops_norm)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                v_preds.append(preds)
                v_targets.append(targets.numpy())

        v_targets_cat = np.concatenate(v_targets)
        v_preds_cat = np.concatenate(v_preds)
        val_f1 = compute_macro_f1(v_targets_cat, v_preds_cat)

        print(f"[M1 Seed {seed}] Epoch {epoch}/{max_epochs} - Loss: {total_loss/len(train_loader):.4f} - Val Macro-F1: {val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    print(f"[M1 Seed {seed}] Best Val Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}", flush=True)
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(best_weights, f"checkpoints/m1_resnet18_s{seed}_best.pt")

    # =========================================================================
    # Test Inference on Lizard and CoNSeP across all 5 conditions
    # =========================================================================
    print(f"[M1 Seed {seed}] Running Inference on Test Sets...", flush=True)
    model.eval()
    out_dir = 'outputs/predictions'
    os.makedirs(out_dir, exist_ok=True)

    conditions = ['clean', 'macenko_mild', 'macenko_moderate', 'macenko_strong', 'randstainna']
    HALF_CROP = 20
    CROP_SIZE = 41
    pad = 25

    def eval_image_crops(img_p, centroids, batch_size=256):
        im = cv2.imread(img_p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_padded = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        all_probs = []
        for i in range(0, len(centroids), batch_size):
            b_c = centroids[i:i+batch_size]
            b_crops = []
            for (x, y) in b_c:
                xp, yp = int(round(x)) + pad, int(round(y)) + pad
                cr = im_padded[yp - HALF_CROP : yp + HALF_CROP + 1, xp - HALF_CROP : xp + HALF_CROP + 1]
                if cr.shape[0] != CROP_SIZE or cr.shape[1] != CROP_SIZE:
                    cr = cv2.resize(cr, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                b_crops.append(cr)

            b_np = np.stack(b_crops, axis=0).astype(np.float32) / 255.0
            b_tensor = torch.from_numpy(b_np).permute(0, 3, 1, 2).to(device)
            b_224 = F.interpolate(b_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            b_norm = (b_224 - norm_mean) / norm_std

            with torch.no_grad():
                logits = model(b_norm)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    with h5py.File('data/processed/features_cache.h5', 'r') as h5:
        # 1. Lizard Test
        liz_keys = list(h5['test_lizard'].keys())
        for cond in conditions:
            all_probs, all_preds, all_targets, all_centroids, all_tiles = [], [], [], [], []
            for fn in liz_keys:
                grp = h5['test_lizard'][fn]
                centroids = grp['centroids'][:]
                labels = grp['labels'][:]
                img_p = f"data/processed/perturbed_test/lizard_{fn}_{cond}.png"

                probs = eval_image_crops(img_p, centroids)
                preds = np.argmax(probs, axis=-1)

                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(labels)
                all_centroids.append(centroids)
                all_tiles.extend([fn] * len(labels))

            save_p = os.path.join(out_dir, f"m1_resnet18_s{seed}_lizard_{cond}.npz")
            np.savez_compressed(
                save_p,
                probs=np.vstack(all_probs),
                preds=np.concatenate(all_preds),
                targets=np.concatenate(all_targets),
                centroids=np.vstack(all_centroids),
                tile_ids=np.array(all_tiles),
                model_type='m1',
                gnn_type='resnet18',
                seed=seed,
                dataset='lizard',
                condition=cond
            )

        # 2. CoNSeP Test
        consep_keys = list(h5['test_consep'].keys())
        for cond in conditions:
            all_probs, all_preds, all_targets, all_centroids, all_tiles = [], [], [], [], []
            for t_name in consep_keys:
                grp = h5['test_consep'][t_name]
                centroids = grp['centroids'][:]
                labels = grp['labels'][:]
                img_p = f"data/processed/perturbed_test/consep_{t_name}_{cond}.png"

                probs = eval_image_crops(img_p, centroids)
                preds = np.argmax(probs, axis=-1)

                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(labels)
                all_centroids.append(centroids)
                all_tiles.extend([t_name] * len(labels))

            save_p = os.path.join(out_dir, f"m1_resnet18_s{seed}_consep_{cond}.npz")
            np.savez_compressed(
                save_p,
                probs=np.vstack(all_probs),
                preds=np.concatenate(all_preds),
                targets=np.concatenate(all_targets),
                centroids=np.vstack(all_centroids),
                tile_ids=np.array(all_tiles),
                model_type='m1',
                gnn_type='resnet18',
                seed=seed,
                dataset='consep',
                condition=cond
            )

    print(f"[M1 Seed {seed}] Completed inference for all conditions on Lizard and CoNSeP.", flush=True)

if __name__ == '__main__':
    for s in [42, 43, 44]:
        train_m1_model(seed=s)
    print("\n=======================================================", flush=True)
    print("ALL M1 RUNS COMPLETE! Predictions saved to outputs/predictions/", flush=True)
    print("=======================================================", flush=True)

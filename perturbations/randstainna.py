import cv2
import numpy as np
import json
import os

class RandStainNA:
    def __init__(self):
        self.stats = None

    def fit_from_images(self, image_paths, sample_per_image=5000):
        print(f"Fitting RandStainNA on {len(image_paths)} training images...")
        means_l, means_a, means_b = [], [], []
        stds_l, stds_a, stds_b = [], [], []

        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Subsample to speed up
            h, w, _ = lab.shape
            pts_h = np.random.randint(0, h, size=min(sample_per_image, h*w))
            pts_w = np.random.randint(0, w, size=min(sample_per_image, h*w))
            pixels = lab[pts_h, pts_w]

            m = np.mean(pixels, axis=0)
            s = np.std(pixels, axis=0)

            means_l.append(m[0])
            means_a.append(m[1])
            means_b.append(m[2])
            stds_l.append(s[0])
            stds_a.append(s[1])
            stds_b.append(s[2])

        self.stats = {
            'mean_l': {'avg': float(np.mean(means_l)), 'std': float(np.std(means_l))},
            'mean_a': {'avg': float(np.mean(means_a)), 'std': float(np.std(means_a))},
            'mean_b': {'avg': float(np.mean(means_b)), 'std': float(np.std(means_b))},
            'std_l': {'avg': float(np.mean(stds_l)), 'std': float(np.std(stds_l))},
            'std_a': {'avg': float(np.mean(stds_a)), 'std': float(np.std(stds_a))},
            'std_b': {'avg': float(np.mean(stds_b)), 'std': float(np.std(stds_b))}
        }
        print("RandStainNA fitting complete.")
        return self.stats

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            self.stats = json.load(f)

    def perturb(self, img_rgb, seed=None, fixed_params=None):
        if self.stats is None:
            raise ValueError("RandStainNA statistics not fitted or loaded!")

        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        if fixed_params is not None:
            target_mean = np.array(fixed_params['target_mean'], dtype=np.float32)
            target_std = np.array(fixed_params['target_std'], dtype=np.float32)
        else:
            target_mean = np.array([
                rng.normal(self.stats['mean_l']['avg'], self.stats['mean_l']['std']),
                rng.normal(self.stats['mean_a']['avg'], self.stats['mean_a']['std']),
                rng.normal(self.stats['mean_b']['avg'], self.stats['mean_b']['std'])
            ], dtype=np.float32)
            target_std = np.array([
                max(5.0, rng.normal(self.stats['std_l']['avg'], self.stats['std_l']['std'])),
                max(2.0, rng.normal(self.stats['std_a']['avg'], self.stats['std_a']['std'])),
                max(2.0, rng.normal(self.stats['std_b']['avg'], self.stats['std_b']['std']))
            ], dtype=np.float32)

        params = {
            'target_mean': target_mean.tolist(),
            'target_std': target_std.tolist()
        }

        # Convert to LAB
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        curr_mean = np.mean(lab, axis=(0, 1))
        curr_std = np.std(lab, axis=(0, 1))
        curr_std[curr_std < 1e-5] = 1e-5

        # Normalize and scale
        lab_norm = (lab - curr_mean) / curr_std
        lab_target = lab_norm * target_std + target_mean
        lab_target = np.clip(lab_target, 0, 255).astype(np.uint8)

        # Convert back to RGB
        img_pert = cv2.cvtColor(lab_target, cv2.COLOR_LAB2RGB)
        return img_pert, params

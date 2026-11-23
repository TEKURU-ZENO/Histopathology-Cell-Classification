import numpy as np

# Standard reference H&E stain matrix
DEFAULT_STAIN_MATRIX = np.array([
    [0.65, 0.70, 0.29],  # Hematoxylin
    [0.07, 0.99, 0.11]   # Eosin
], dtype=np.float64)

class MacenkoPerturbator:
    def __init__(self, alpha=1.0, beta=0.15):
        self.alpha = alpha
        self.beta = beta
        # Severity configuration
        self.severities = {
            'mild': {'conc_range': 0.10, 'rot_deg': 5.0},
            'moderate': {'conc_range': 0.20, 'rot_deg': 10.0},
            'strong': {'conc_range': 0.30, 'rot_deg': 15.0}
        }

    def rgb_to_od(self, img_rgb):
        img_rgb = img_rgb.astype(np.float64)
        return -np.log10((img_rgb + 1.0) / 255.0)

    def od_to_rgb(self, od):
        rgb = 255.0 * (10.0 ** (-od)) - 1.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def extract_stain_matrix(self, img_rgb):
        od = self.rgb_to_od(img_rgb).reshape((-1, 3))
        # Filter transparent/background pixels
        mask = (od > self.beta).any(axis=1)
        od_hat = od[mask]
        
        if len(od_hat) < 100:
            return DEFAULT_STAIN_MATRIX.copy()

        # SVD on covariance
        try:
            cov = np.cov(od_hat, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Two largest eigenvectors
            plane = eigvecs[:, [1, 2]]
            
            # Project onto 2D plane
            proj = np.dot(od_hat, plane)
            # Normalize projection
            norm = np.linalg.norm(proj, axis=1, keepdims=True)
            norm[norm == 0] = 1e-8
            proj_norm = proj / norm
            
            angles = np.arctan2(proj_norm[:, 1], proj_norm[:, 0])
            phi_min = np.percentile(angles, self.alpha)
            phi_max = np.percentile(angles, 100 - self.alpha)
            
            v_min = np.dot(plane, np.array([np.cos(phi_min), np.sin(phi_min)]))
            v_max = np.dot(plane, np.array([np.cos(phi_max), np.sin(phi_max)]))
            
            # Hematoxylin first, Eosin second (based on red absorption)
            if v_min[0] > v_max[0]:
                he = np.array([v_min, v_max])
            else:
                he = np.array([v_max, v_min])
                
            # Normalize stain vectors
            he = he / np.linalg.norm(he, axis=1, keepdims=True)
            return he
        except Exception:
            return DEFAULT_STAIN_MATRIX.copy()

    def perturb(self, img_rgb, severity='moderate', seed=None, fixed_params=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        sev = self.severities[severity]
        conc_range = sev['conc_range']
        rot_deg = sev['rot_deg']
        rot_rad = np.radians(rot_deg)

        if fixed_params is not None:
            scale_h = fixed_params['scale_h']
            scale_e = fixed_params['scale_e']
            rot_h = fixed_params['rot_h']
            rot_e = fixed_params['rot_e']
        else:
            scale_h = rng.uniform(1.0 - conc_range, 1.0 + conc_range)
            scale_e = rng.uniform(1.0 - conc_range, 1.0 + conc_range)
            rot_h = rng.uniform(-rot_rad, rot_rad)
            rot_e = rng.uniform(-rot_rad, rot_rad)

        params = {
            'severity': severity,
            'scale_h': float(scale_h),
            'scale_e': float(scale_e),
            'rot_h': float(rot_h),
            'rot_e': float(rot_e)
        }

        # Original stain matrix
        he = self.extract_stain_matrix(img_rgb)
        
        # Perturb stain vectors via rotation in 3D
        def rotate_vector(vec, angle):
            # Rotate in arbitrary orthogonal plane
            # Small random perturbation
            axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(vec, axis)) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            rot_axis = np.cross(vec, axis)
            rot_axis /= np.linalg.norm(rot_axis)
            # Rodrigues rotation
            v_rot = (vec * np.cos(angle) + 
                     np.cross(rot_axis, vec) * np.sin(angle) + 
                     rot_axis * np.dot(rot_axis, vec) * (1.0 - np.cos(angle)))
            return v_rot / np.linalg.norm(v_rot)

        he_pert = np.zeros_like(he)
        he_pert[0] = rotate_vector(he[0], rot_h)
        he_pert[1] = rotate_vector(he[1], rot_e)

        # Deconvolve concentrations: OD = C * V  => C = OD * pinv(V)
        od = self.rgb_to_od(img_rgb).reshape((-1, 3))
        # he is (2, 3), so pinv(he) is (3, 2)
        pinv = np.linalg.pinv(he)
        c = np.dot(od, pinv)
        
        # Scale concentrations
        c[:, 0] *= scale_h
        c[:, 1] *= scale_e
        c = np.clip(c, 0, None)

        # Reconstruct OD: OD_pert = C * V_pert
        od_pert = np.dot(c, he_pert)
        od_pert = od_pert.reshape(img_rgb.shape)
        
        img_pert = self.od_to_rgb(od_pert)
        return img_pert, params

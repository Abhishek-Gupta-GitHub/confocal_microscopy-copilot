# copilot/detection_deeptrack.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np

try:
    import deeptrack as dt  # DeepTrack2
    HAS_DEEPTRACK = True
except Exception:
    HAS_DEEPTRACK = False


@dataclass
class DeepTrackConfig:
    image_shape: tuple[int, int] = (64, 64)
    crop_z: Optional[int] = None       # for 3D stacks: number of slices or None
    particle_radius_px: float = 3.0
    particle_intensity: float = 1.0
    n_particles_min: int = 5
    n_particles_max: int = 20
    noise_level: float = 0.05
    backend: str = "cnn_localizer"     # placeholder flag (“classical” vs “cnn”)


class DeepTrackDetector:
    """
    Thin wrapper around DeepTrack2 for:
    - synthetic data generation (2D / 3D)
    - feature localization on image stacks
    """

    def __init__(self, cfg: Optional[DeepTrackConfig] = None):
        self.cfg = cfg or DeepTrackConfig()
        self.available = HAS_DEEPTRACK

    # ----------------------------
    # Synthetic data (DeepTrack)
    # ----------------------------
    def generate_synthetic_2d_t(self, n_frames: int = 100) -> np.ndarray:
        """
        Return a 3D array (T, H, W) of synthetic 2D+t data.
        Uses a simple DeepTrack PointParticle pipeline if available,
        otherwise falls back to a Gaussian-blob generator.
        """
        H, W = self.cfg.image_shape
        if not self.available:
            return self._fallback_synthetic_2d_t(n_frames, H, W)

        # Minimal DeepTrack-style pipeline: PointParticle + noise + PSF
        particle = dt.features.PointParticle(
            position=lambda: np.random.rand(2) * np.array([H, W]),
            intensity=self.cfg.particle_intensity,
        )
        psf = dt.features.Gaussian(sigma=self.cfg.particle_radius_px)
        noise = dt.features.PoissonNoise(intensity=1.0)

        image = (particle + psf + noise)(
            size=(H, W),
            number_of_particles=lambda: np.random.randint(
                self.cfg.n_particles_min, self.cfg.n_particles_max + 1
            )
        )

        stack = [image() for _ in range(n_frames)]
        return np.stack(stack, axis=0)

    def _fallback_synthetic_2d_t(self, n_frames: int, H: int, W: int) -> np.ndarray:
        """Simple Gaussian blobs as a backup when DeepTrack is not installed."""
        stack = np.zeros((n_frames, H, W), dtype=np.float32)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        n_particles = np.random.randint(self.cfg.n_particles_min, self.cfg.n_particles_max + 1)

        for _ in range(n_particles):
            x0 = np.random.uniform(0, W)
            y0 = np.random.uniform(0, H)
            r2 = (xx - x0) ** 2 + (yy - y0) ** 2
            sigma2 = (self.cfg.particle_radius_px ** 2)
            blob = np.exp(-0.5 * r2 / sigma2)
            for t in range(n_frames):
                stack[t] += self.cfg.particle_intensity * blob

        stack += self.cfg.noise_level * np.random.randn(*stack.shape)
        stack -= stack.min()
        stack /= (stack.max() + 1e-8)
        return stack

    # ----------------------------
    # Detection on given stack
    # ----------------------------
    def detect_2d_t(
        self,
        stack_2d_t: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Localize particles frame-by-frame in a 2D+t stack.
        Returns a dict with 'positions' (N, 4): [frame, y, x, intensity].
        """
        T, H, W = stack_2d_t.shape
        positions = []

        if not self.available:
            # simple threshold + centroid fallback
            for t in range(T):
                img = stack_2d_t[t]
                mask = img > threshold * img.max()
                ys, xs = np.where(mask)
                for y, x in zip(ys, xs):
                    positions.append([t, float(y), float(x), float(img[y, x])])
        else:
            # Using DeepTrack pipeline as a “localizer” (conceptual; adapt to your network)
            # Here we show per-frame inference conceptually; training is done in DT tutorials.
            for t in range(T):
                img = stack_2d_t[t]
                # Example: a trained CNN model loaded via DeepTrack (pseudo-code)
                # prediction = my_trained_model(img[None, ..., None])
                # peaks = some_peak_finder(prediction)
                # for each peak: append [t, y, x, score]
                # For now, behave like fallback.
                mask = img > threshold * img.max()
                ys, xs = np.where(mask)
                for y, x in zip(ys, xs):
                    positions.append([t, float(y), float(x), float(img[y, x])])

        positions = np.array(positions, dtype=np.float32)
        return {
            "positions": positions,
            "meta": {
                "T": T,
                "H": H,
                "W": W,
                "backend": "deeptrack" if self.available else "threshold_fallback",
            },
        }

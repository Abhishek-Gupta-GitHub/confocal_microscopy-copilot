# golden_dataset.py
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tifffile as tiff
import json

from copilot.config import DEFAULT_META


def simulate_brownian_3d_stack(
    n_frames: int = 100,
    n_particles: int = 30,
    box_size_xyz: Tuple[int, int, int] = (64, 64, 32),
    diffusion_coeff: float = 0.05,
    dt: float = 0.1,
    psf_sigma_xy: float = 1.5,
    psf_sigma_z: float = 1.5,
    background_level: float = 20.0,
    particle_intensity: float = 200.0,
    read_noise_sigma: float = 3.0,
    bit_depth: int = 12,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate a synthetic 4D confocal-style stack:
      shape (T, Z, Y, X), Brownian particles in a 3D box,
      convolved with a simple Gaussian PSF and with realistic noise.

    Returns
    -------
    stack : np.ndarray
        Integer array, shape (T, Z, Y, X).
    meta : Dict[str, Any]
        Minimal metadata with voxel size, frame time, etc.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = box_size_xyz[0], box_size_xyz[1], box_size_xyz[2]
    T = n_frames

    # Discrete step size from diffusion coefficient: <Δr^2> = 6 D dt  (3D)
    step_std = np.sqrt(2 * diffusion_coeff * dt)

    # Initialize positions uniformly in the box (continuous coordinates)
    # Coordinates are in voxels.
    x = rng.uniform(0.0, nx - 1.0, size=n_particles)
    y = rng.uniform(0.0, ny - 1.0, size=n_particles)
    z = rng.uniform(0.0, nz - 1.0, size=n_particles)

    stack = np.zeros((T, nz, ny, nx), dtype=np.float32)

    # Precompute simple 3D Gaussian kernel around each particle
    # For speed, use a small local patch of radius ~3 sigma
    rad_xy = int(3 * psf_sigma_xy)
    rad_z = int(3 * psf_sigma_z)
    x_grid = np.arange(-rad_xy, rad_xy + 1)
    y_grid = np.arange(-rad_xy, rad_xy + 1)
    z_grid = np.arange(-rad_z, rad_z + 1)
    Xg, Yg, Zg = np.meshgrid(x_grid, y_grid, z_grid, indexing="xy")

    psf = np.exp(
        -(
            (Xg**2) / (2 * psf_sigma_xy**2)
            + (Yg**2) / (2 * psf_sigma_xy**2)
            + (Zg**2) / (2 * psf_sigma_z**2)
        )
    )
    psf /= psf.sum()

    for t in range(T):
        # Background
        frame = np.full((nz, ny, nx), background_level, dtype=np.float32)

        # Brownian step with reflecting boundaries
        x += rng.normal(0.0, step_std, size=n_particles)
        y += rng.normal(0.0, step_std, size=n_particles)
        z += rng.normal(0.0, step_std, size=n_particles)

        # Reflecting walls in each dimension
        for arr, bound in ((x, nx - 1), (y, ny - 1), (z, nz - 1)):
            arr[arr < 0] = -arr[arr < 0]
            arr[arr > bound] = 2 * bound - arr[arr > bound]

        # Render particles
        for px, py, pz in zip(x, y, z):
            cx, cy, cz = int(round(px)), int(round(py)), int(round(pz))
            x_min = max(cx - rad_xy, 0)
            x_max = min(cx + rad_xy, nx - 1)
            y_min = max(py - rad_xy, 0)
            y_max = min(py + rad_xy, ny - 1)
            z_min = max(cz - rad_z, 0)
            z_max = min(cz + rad_z, nz - 1)

            # Corresponding PSF indices
            px_min = x_min - (cx - rad_xy)
            px_max = px_min + (x_max - x_min)
            py_min = y_min - (cy - rad_xy)
            py_max = py_min + (y_max - y_min)
            pz_min = z_min - (cz - rad_z)
            pz_max = pz_min + (z_max - z_min)

            frame[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] += (
                particle_intensity
                * psf[
                    py_min : py_max + 1,
                    px_min : px_max + 1,
                    pz_min : pz_max + 1,
                ].transpose(2, 0, 1)
            )

        # Add Gaussian read noise
        frame += rng.normal(0.0, read_noise_sigma, size=frame.shape)

        # Store
        stack[t] = frame

    # Quantize to mimick camera bit depth
    max_val = float(2**bit_depth - 1)
    stack = np.clip(stack, 0, max_val).astype(np.uint16)

    meta = DEFAULT_META.copy()
    meta.update(
        {
            "n_frames": int(T),
            "pixel_size_um": 0.1,
            "z_step_um": 0.2,
            "dt_s": float(dt),
            "description": "Synthetic 3D Brownian confocal stack (golden dataset).",
            "diffusion_coeff_um2_s": float(diffusion_coeff),
            "n_particles": int(n_particles),
            "box_size_xyz": [nx, ny, nz],
        }
    )
    return stack, meta


def ensure_golden_dataset(data_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate (and cache) the golden Brownian dataset under data/golden_brownian_3d.
    """
    folder = data_dir / "golden_brownian_3d"
    folder.mkdir(exist_ok=True, parents=True)
    stack_path = folder / "stack.tif"
    meta_path = folder / "metadata.json"

    if stack_path.exists() and meta_path.exists():
        stack = tiff.imread(str(stack_path))
        meta = json.loads(meta_path.read_text())
        return stack, meta

    stack, meta = simulate_brownian_3d_stack()
    tiff.imwrite(str(stack_path), stack, imagej=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    return stack, meta

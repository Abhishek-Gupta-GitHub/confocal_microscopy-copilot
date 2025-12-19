# generate_brownian_data.py
from pathlib import Path
from typing import Dict, Any, Tuple

import json
import numpy as np
import tifffile as tiff

from copilot.config import DEFAULT_META  # uses your existing config


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
    Synthetic 3D Brownian confocal-style stack: shape (T, Z, Y, X).

    - 3D Brownian motion with reflecting boundaries.
    - Simple Gaussian PSF.
    - Additive Gaussian read noise and camera quantization.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = box_size_xyz
    T = n_frames

    # <Δr^2> = 6 D dt in 3D for Brownian motion
    step_std = np.sqrt(2 * diffusion_coeff * dt)

    # Initial positions in voxel coordinates
    x = rng.uniform(0.0, nx - 1.0, size=n_particles)
    y = rng.uniform(0.0, ny - 1.0, size=n_particles)
    z = rng.uniform(0.0, nz - 1.0, size=n_particles)

    stack = np.zeros((T, nz, ny, nx), dtype=np.float32)

    # PSF support
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
        frame = np.full((nz, ny, nx), background_level, dtype=np.float32)

        # Brownian step
        x += rng.normal(0.0, step_std, size=n_particles)
        y += rng.normal(0.0, step_std, size=n_particles)
        z += rng.normal(0.0, step_std, size=n_particles)

        # Reflective boundaries
        for arr, bound in ((x, nx - 1), (y, ny - 1), (z, nz - 1)):
            arr[arr < 0] = -arr[arr < 0]
            arr[arr > bound] = 2 * bound - arr[arr > bound]

        # Render each particle with local PSF patch
        for px, py, pz in zip(x, y, z):
            cx, cy, cz = int(round(px)), int(round(py)), int(round(pz))

            x_min = max(cx - rad_xy, 0)
            x_max = min(cx + rad_xy, nx - 1)
            y_min = max(cy - rad_xy, 0)
            y_max = min(cy + rad_xy, ny - 1)
            z_min = max(cz - rad_z, 0)
            z_max = min(cz + rad_z, nz - 1)

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

        frame += rng.normal(0.0, read_noise_sigma, size=frame.shape)
        stack[t] = frame

    max_val = float(2**bit_depth - 1)
    stack = np.clip(stack, 0, max_val).astype(np.uint16)

    meta = DEFAULT_META.copy()
    meta.update(
        {
            "n_frames": int(T),
            "pixel_size_um": 0.1,
            "z_step_um": 0.2,
            "dt_s": float(dt),
            "description": "Brownian data: 3D particles in confocal-style image frames (specified frame size and time step).",
            "diffusion_coeff_um2_s": float(diffusion_coeff),
            "n_particles": int(n_particles),
            "box_size_xyz": [nx, ny, nz],
        }
    )
    return stack, meta

def simulate_colloidal_3d_spheres(
    n_frames: int = 100,
    n_particles: int = 30,
    box_size_xyz: Tuple[int, int, int] = (64, 64, 32),  # (X, Y, Z)
    diffusion_coeff: float = 0.05,
    dt: float = 0.1,
    sphere_radius_px: float = 3.0,      # controls apparent diameter in voxels
    background_level: float = 20.0,
    particle_intensity: float = 600.0,  # brighter than before for clear tracks
    read_noise_sigma: float = 1.0,      # lower noise -> higher SNR
    bit_depth: int = 12,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Synthetic 3D Brownian colloidal spheres: stack shape (T, Z, Y, X).

    - Each particle is rendered as a solid 3D sphere with soft edges.
    - True 3D Brownian motion with reflecting boundaries.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = box_size_xyz
    T = n_frames

    # 3D Brownian step: <Δr^2> = 6 D dt
    step_std = np.sqrt(2 * diffusion_coeff * dt)

    # Initial positions (centered away from boundaries by ~1 radius)
    margin = sphere_radius_px + 1
    x = rng.uniform(margin, nx - 1 - margin, size=n_particles)
    y = rng.uniform(margin, ny - 1 - margin, size=n_particles)
    z = rng.uniform(margin, nz - 1 - margin, size=n_particles)

    stack = np.zeros((T, nz, ny, nx), dtype=np.float32)

    # Precompute spherical kernel
    rad = int(np.ceil(sphere_radius_px))
    x_grid = np.arange(-rad, rad + 1)
    y_grid = np.arange(-rad, rad + 1)
    z_grid = np.arange(-rad, rad + 1)
    Xg, Yg, Zg = np.meshgrid(x_grid, y_grid, z_grid, indexing="xy")
    r = np.sqrt(Xg**2 + Yg**2 + Zg**2)

    # Soft-edged sphere: intensity ~ 1 inside radius, decays smoothly to 0
    sphere_kernel = np.exp(-((r / sphere_radius_px) ** 4))
    sphere_kernel[r > 1.5 * sphere_radius_px] = 0.0
    sphere_kernel /= sphere_kernel.sum()

    for t in range(T):
        frame = np.full((nz, ny, nx), background_level, dtype=np.float32)

        # Brownian step
        x += rng.normal(0.0, step_std, size=n_particles)
        y += rng.normal(0.0, step_std, size=n_particles)
        z += rng.normal(0.0, step_std, size=n_particles)

        # Reflective boundaries
        for arr, bound in ((x, nx - 1), (y, ny - 1), (z, nz - 1)):
            arr[arr < 0] = -arr[arr < 0]
            arr[arr > bound] = 2 * bound - arr[arr > bound]

        # Render each 3D sphere
        for px, py, pz in zip(x, y, z):
            cx, cy, cz = int(round(px)), int(round(py)), int(round(pz))

            x_min = max(cx - rad, 0)
            x_max = min(cx + rad, nx - 1)
            y_min = max(cy - rad, 0)
            y_max = min(cy + rad, ny - 1)
            z_min = max(cz - rad, 0)
            z_max = min(cz + rad, nz - 1)

            kx_min = x_min - (cx - rad)
            kx_max = kx_min + (x_max - x_min)
            ky_min = y_min - (cy - rad)
            ky_max = ky_min + (y_max - y_min)
            kz_min = z_min - (cz - rad)
            kz_max = kz_min + (z_max - z_min)

            frame[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] += (
                particle_intensity
                * sphere_kernel[
                    ky_min : ky_max + 1,
                    kx_min : kx_max + 1,
                    kz_min : kz_max + 1,
                ].transpose(2, 0, 1)
            )

        # Add read noise
        frame += rng.normal(0.0, read_noise_sigma, size=frame.shape)
        stack[t] = frame

    max_val = float(2**bit_depth - 1)
    stack = np.clip(stack, 0, max_val).astype(np.uint16)

    meta = DEFAULT_META.copy()
    meta.update(
        {
            "n_frames": int(T),
            "pixel_size_um": 0.1,
            "z_step_um": 0.2,
            "dt_s": float(dt),
            "description": "3D Brownian colloidal spheres (solid 3D particles with tunable radius).",
            "diffusion_coeff_um2_s": float(diffusion_coeff),
            "n_particles": int(n_particles),
            "box_size_xyz": [nx, ny, nz],
            "sphere_radius_px": float(sphere_radius_px),
        }
    )
    return stack, meta




def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    folder = data_dir / "brownian_3d colloids"
    stack_path = folder / "stack.tif"
    meta_path = folder / "metadata.json"

    folder.mkdir(parents=True, exist_ok=True)

    if stack_path.exists() and meta_path.exists():
        print(f"[INFO] Brownian data already exists at {folder}, nothing to do.")
        return

    stack, meta = simulate_colloidal_3d_spheres()
    tiff.imwrite(str(stack_path), stack, imagej=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[OK] Wrote Brownian data to {stack_path}")
    print(f"[OK] Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()

import os
import json
import numpy as np
from tifffile import imsave

# ---- Small helper utilities ----

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def gaussian_spot(shape, center, sigma_xy=1.2, sigma_z=1.2):
    """Create a 3D anisotropic Gaussian spot inside an array of given shape."""
    zdim, ydim, xdim = shape
    zc, yc, xc = center
    z = np.arange(zdim)[:, None, None]
    y = np.arange(ydim)[None, :, None]
    x = np.arange(xdim)[None, None, :]
    return np.exp(-(((z - zc) ** 2) / (2 * sigma_z**2)
                    + ((y - yc) ** 2) / (2 * sigma_xy**2)
                    + ((x - xc) ** 2) / (2 * sigma_xy**2)))

def add_poisson_gaussian_noise(stack, read_noise=3.0, gain=200.0):
    """Simple Poisson + Gaussian noise model."""
    # Poisson noise (photon counting)
    noisy = np.random.poisson(np.clip(stack * gain, 0, None)).astype(np.float32)
    # Add camera read noise
    noisy += np.random.normal(0, read_noise, size=noisy.shape).astype(np.float32)
    # Normalize back to 0–1
    noisy = noisy / noisy.max()
    return noisy

def depth_attenuation(stack, length_scale_vox=6.0):
    """Apply exponential attenuation along z for confocal-like stacks."""
    T, Z, Y, X = stack.shape
    z = np.arange(Z)[None, :, None, None]
    attenuation = np.exp(-z / length_scale_vox)
    return stack * attenuation

# ---- Dataset generators ----

def generate_soft_matter_gel(T=8, Z=8, Y=64, X=64, n_particles=15):
    stack = np.zeros((T, Z, Y, X), dtype=np.float32)
    # Initial positions
    positions = np.stack([
        np.random.uniform(1, Z-2, size=n_particles),
        np.random.uniform(8, Y-8, size=n_particles),
        np.random.uniform(8, X-8, size=n_particles),
    ], axis=-1)  # (n_particles, 3) -> (z,y,x)
    # Subdiffusive-like: small random steps
    step_sigma = np.array([0.3, 0.7, 0.7])

    for t in range(T):
        frame = np.zeros((Z, Y, X), dtype=np.float32)
        for i in range(n_particles):
            zc, yc, xc = positions[i]
            frame += gaussian_spot((Z, Y, X), (zc, yc, xc),
                                   sigma_xy=1.4, sigma_z=1.2)
        # Crowded background (blurred noise)
        background = np.random.normal(0.1, 0.02, size=(Z, Y, X))
        background = np.clip(background, 0, None)
        frame = frame * 0.9 + background.astype(np.float32)
        stack[t] = frame / frame.max()

        # Update positions
        steps = np.random.normal(0, step_sigma, size=(n_particles, 3))
        positions += steps
        # Reflecting boundaries
        positions[:, 0] = np.clip(positions[:, 0], 1, Z-2)
        positions[:, 1] = np.clip(positions[:, 1], 8, Y-8)
        positions[:, 2] = np.clip(positions[:, 2], 8, X-8)

    stack = depth_attenuation(stack, length_scale_vox=6.0)
    stack = add_poisson_gaussian_noise(stack)
    return stack

def generate_cell_nucleus(T=5, Z=6, Y=64, X=64, n_foci=6):
    stack = np.zeros((T, Z, Y, X), dtype=np.float32)
    # Random static centers (slight drift later)
    centers = np.stack([
        np.random.uniform(1.5, Z-1.5, size=n_foci),
        np.random.uniform(12, Y-12, size=n_foci),
        np.random.uniform(12, X-12, size=n_foci),
    ], axis=-1)

    drift_per_frame = np.array([0.05, 0.1, -0.05])

    for t in range(T):
        frame = np.zeros((Z, Y, X), dtype=np.float32)
        for i in range(n_foci):
            zc, yc, xc = centers[i] + t * drift_per_frame
            frame += gaussian_spot((Z, Y, X), (zc, yc, xc),
                                   sigma_xy=1.6, sigma_z=1.0)
        # Smooth nuclear background
        bg = np.random.normal(0.12, 0.015, size=(Z, Y, X))
        bg = np.clip(bg, 0, None)
        frame = frame * 1.4 + bg.astype(np.float32)
        stack[t] = frame / frame.max()

    stack = depth_attenuation(stack, length_scale_vox=4.0)
    stack = add_poisson_gaussian_noise(stack)
    return stack

def generate_colloidal_monolayer(T=10, Y=64, X=64, n_particles=20):
    # Single-plane stack, but stored with Z=1
    Z = 1
    stack = np.zeros((T, Z, Y, X), dtype=np.float32)
    # 2D positions
    positions = np.stack([
        np.random.uniform(8, Y-8, size=n_particles),
        np.random.uniform(8, X-8, size=n_particles),
    ], axis=-1)
    step_sigma = 0.8

    for t in range(T):
        frame = np.zeros((Y, X), dtype=np.float32)
        for i in range(n_particles):
            yc, xc = positions[i]
            # Use Gaussian spot in 2D
            y = np.arange(Y)[:, None]
            x = np.arange(X)[None, :]
            frame += np.exp(-(((y - yc) ** 2) + ((x - xc) ** 2)) / (2 * 1.5**2))
        bg = np.random.normal(0.05, 0.01, size=(Y, X))
        bg = np.clip(bg, 0, None)
        frame = frame * 1.2 + bg.astype(np.float32)
        frame = frame / frame.max()
        stack[t, 0] = frame

        # Update positions
        steps = np.random.normal(0, step_sigma, size=(n_particles, 2))
        positions += steps
        positions[:, 0] = np.clip(positions[:, 0], 8, Y-8)
        positions[:, 1] = np.clip(positions[:, 1], 8, X-8)

    stack = add_poisson_gaussian_noise(stack)
    return stack

def generate_membrane_proteins(T=12, Y=64, X=64, n_particles=30, D_fast=True):
    Z = 1
    stack = np.zeros((T, Z, Y, X), dtype=np.float32)
    positions = np.stack([
        np.random.uniform(6, Y-6, size=n_particles),
        np.random.uniform(6, X-6, size=n_particles),
    ], axis=-1)
    step_sigma = 1.6 if D_fast else 0.7

    for t in range(T):
        frame = np.zeros((Y, X), dtype=np.float32)
        for i in range(n_particles):
            yc, xc = positions[i]
            # Blinking probability
            if np.random.rand() < 0.25:
                continue
            y = np.arange(Y)[:, None]
            x = np.arange(X)[None, :]
            frame += np.exp(-(((y - yc) ** 2) + ((x - xc) ** 2)) / (2 * 1.2**2))
        bg = np.random.normal(0.06, 0.01, size=(Y, X))
        bg = np.clip(bg, 0, None)
        frame = frame * 1.3 + bg.astype(np.float32)
        frame = frame / frame.max()
        stack[t, 0] = frame

        steps = np.random.normal(0, step_sigma, size=(n_particles, 2))
        positions += steps
        positions[:, 0] = np.clip(positions[:, 0], 6, Y-6)
        positions[:, 1] = np.clip(positions[:, 1], 6, X-6)

    stack = add_poisson_gaussian_noise(stack)
    return stack

def generate_material_microstructure(Z=16, Y=64, X=64, n_grains=6):
    # Single time point, 3D volume
    T = 1
    stack = np.zeros((T, Z, Y, X), dtype=np.float32)
    volume = np.zeros((Z, Y, X), dtype=np.float32)

    centers = np.stack([
        np.random.uniform(2, Z-2, size=n_grains),
        np.random.uniform(8, Y-8, size=n_grains),
        np.random.uniform(8, X-8, size=n_grains),
    ], axis=-1)

    for i in range(n_grains):
        zc, yc, xc = centers[i]
        grain = gaussian_spot((Z, Y, X), (zc, yc, xc),
                              sigma_xy=np.random.uniform(4, 7),
                              sigma_z=np.random.uniform(2, 4))
        scale = np.random.uniform(0.6, 1.2)
        volume += scale * grain

    volume = volume / volume.max()
    # Add some “phase contrast” noise
    noise = np.random.normal(0.05, 0.02, size=(Z, Y, X))
    volume = np.clip(volume + noise, 0, 1)
    stack[0] = volume
    stack = add_poisson_gaussian_noise(stack)
    return stack

# ---- Save everything under data/ ----

ROOT = "data"

def save_stack_with_metadata(subfolder, stack, description, voxel_size_um=(0.3, 0.1, 0.1), dt_s=0.5):
    folder = os.path.join(ROOT, subfolder)
    ensure_dir(folder)
    tiff_path = os.path.join(folder, "stack.tif")
    meta_path = os.path.join(folder, "metadata.json")

    # Save TIFF (float32 to uint16 scaled)
    arr = stack.astype(np.float32)
    arr = arr / arr.max()
    arr_u16 = (arr * 65535).astype(np.uint16)
    imsave(tiff_path, arr_u16)

    T, Z, Y, X = arr.shape
    metadata = {
        "description": description,
        "shape_TZYX": [int(T), int(Z), int(Y), int(X)],
        "voxel_size_um": {
            "z": float(voxel_size_um[0]),
            "y": float(voxel_size_um[1]),
            "x": float(voxel_size_um[2]),
        },
        "frame_interval_s": float(dt_s),
        "dtype": "uint16",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {subfolder}: {tiff_path}")

def main():
    ensure_dir(ROOT)

    soft = generate_soft_matter_gel()
    save_stack_with_metadata(
        "soft_matter_gel",
        soft,
        "Soft-matter gel: Brownian tracer particles in a crowded viscoelastic matrix (short 4D stack).",
    )

    cell = generate_cell_nucleus()
    save_stack_with_metadata(
        "cell_nucleus_spots",
        cell,
        "Cell nucleus: Confocal z-stack with a few quasi-static fluorescent foci and realistic shot noise.",
    )

    colloids = generate_colloidal_monolayer()
    save_stack_with_metadata(
        "colloidal_monolayer",
        colloids,
        "Colloidal monolayer: 2D Brownian motion of ~20 particles in a single focal plane.",
    )

    membrane = generate_membrane_proteins()
    save_stack_with_metadata(
        "membrane_proteins",
        membrane,
        "Membrane proteins: Fast diffusing and blinking fluorescent spots in a single confocal slice.",
    )

    micro = generate_material_microstructure()
    save_stack_with_metadata(
        "material_microstructure",
        micro,
        "Material microstructure: Static 3D grain-like intensity pattern (single time point, 3D only).",
    )

if __name__ == "__main__":
    main()

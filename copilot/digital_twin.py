import numpy as np
from scipy.ndimage import gaussian_filter

def simulate_brownian_trajectories(n_particles, n_frames, dt, D, box_size_um):
    positions = np.zeros((n_frames, n_particles, 3), dtype=float)
    positions[0] = np.random.rand(n_particles, 3) * np.array(box_size_um)
    sigma_step = np.sqrt(2 * D * dt)
    for t in range(1, n_frames):
        steps = np.random.normal(scale=sigma_step, size=(n_particles, 3))
        positions[t] = positions[t - 1] + steps
        positions[t] = np.clip(positions[t], 0, box_size_um)
    return positions

def render_stack_from_positions(positions_um, voxel_size_um, img_shape_xyz,
                                psf_sigma_xyz_vox, noise_std=10.0,
                                z_att_um=50.0, bleach_tau_s=80.0, dt=0.1):
    n_frames, n_particles, _ = positions_um.shape
    nz, ny, nx = img_shape_xyz
    vx, vy, vz = voxel_size_um

    stack = np.zeros((n_frames, nz, ny, nx), dtype=np.float32)
    z_coords_um = np.arange(nz) * vz
    depth_factor = np.exp(-z_coords_um / z_att_um).reshape(-1, 1, 1)

    for t in range(n_frames):
        frame = np.zeros((nz, ny, nx), dtype=np.float32)
        for p in range(n_particles):
            x_um, y_um, z_um = positions_um[t, p]
            ix = int(x_um / vx)
            iy = int(y_um / vy)
            iz = int(z_um / vz)
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                frame[iz, iy, ix] += 1.0

        frame = gaussian_filter(frame, sigma=psf_sigma_xyz_vox)
        frame *= depth_factor
        frame *= np.exp(-t * dt / bleach_tau_s)
        frame += np.random.normal(scale=noise_std, size=frame.shape)

        stack[t] = frame

    return stack

def simulate_confocal_stack(meta, twin_settings):
    voxel_size_um = tuple(meta.get("voxel_size_um", [0.1, 0.1, 0.2]))
    img_shape_xyz = tuple(meta.get("img_shape_xyz", [32, 64, 64]))
    psf_sigma_xyz_vox = tuple(meta.get("psf_sigma_xyz_vox", [2.0, 1.0, 1.0]))
    noise_std = float(meta.get("noise_std", 5.0))
    z_att_um = float(meta.get("z_att_um", 50.0))
    bleach_tau_s = float(meta.get("bleach_tau_s", 80.0))

    n_particles = int(twin_settings.get("n_particles", 200))
    n_frames = int(twin_settings.get("n_frames", 100))
    dt = float(twin_settings.get("dt", 0.1))
    D = float(twin_settings.get("D", 0.2))
    box_size_um = meta.get("box_size_um", [30.0, 30.0, 30.0])

    positions = simulate_brownian_trajectories(
        n_particles=n_particles,
        n_frames=n_frames,
        dt=dt,
        D=D,
        box_size_um=box_size_um,
    )

    stack = render_stack_from_positions(
        positions_um=positions,
        voxel_size_um=voxel_size_um,
        img_shape_xyz=img_shape_xyz,
        psf_sigma_xyz_vox=psf_sigma_xyz_vox,
        noise_std=noise_std,
        z_att_um=z_att_um,
        bleach_tau_s=bleach_tau_s,
        dt=dt,
    )

    return stack, positions


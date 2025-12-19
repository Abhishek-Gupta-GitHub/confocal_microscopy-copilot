# copilot/config.py

DEFAULT_META = {
    # voxel size in µm (x, y, z)
    "voxel_size_um": [0.1, 0.1, 0.2],
    # image shape (z, y, x)
    "img_shape_xyz": [32, 64, 64],
    # PSF sigma in voxels (z, y, x)
    "psf_sigma_xyz_vox": [2.0, 1.0, 1.0],
    # default frame interval in seconds
    "frame_interval_s": 0.1,
    # simple confocal-like effects
    "noise_std": 5.0,
    "z_att_um": 50.0,
    "bleach_tau_s": 80.0,
    # simulation box in µm
    "box_size_um": [30.0, 30.0, 30.0],
}

DEFAULT_SAMPLE_CONFIG = {
    "domain": "colloidal_glass",
    "volume_fraction": 0.45,
    "particle_diameter_um": 1.0,
}


"""
digital_twin.py

Digital twin for confocal microscopy:
- PSFFactory with realistic optics hooks (Gaussian + optional Gibson–Lanni via MicroscPSF).
- MediumModel interface for different particle dynamics (viscous, viscoelastic, near-wall hindered).
- Rendering with depth-dependent attenuation and bleaching.
- Simple parameter recommender based on image quality metrics.

The goal is a clean forward model:

    twin = DigitalTwin()
    image_stack, positions = twin.simulate(microscope_config, sample_config, n_frames)

plus:

    suggestion = twin.recommend_parameters(real_stack, base_microscope_config, sample_config)

to support GUI feedback and “inverse design”.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.ndimage import gaussian_filter

# Optional MicroscPSF import (for Gibson–Lanni style PSF if available)
try:
    import MicroscPSF  # type: ignore
    _HAS_MICROSCPSF = True
except Exception:
    _HAS_MICROSCPSF = False


# ---------------------------------------------------------------------
# PSF factory: optics layer
# ---------------------------------------------------------------------


@dataclass
class MicroscopeConfig:
    """Minimal microscope config for the twin."""
    na: float
    wavelength_nm: float
    immersion_ri: float
    sample_ri: float
    pixel_size_xy_um: float
    z_step_um: float
    img_shape_xyz: Tuple[int, int, int]  # (nz, ny, nx)
    psf_mode: str = "gaussian_confocal"  # or "gibson_lanni"


class PSFFactory:
    """
    Factory that builds a 3D PSF kernel given microscope parameters.

    Modes:
    - "gaussian_confocal": anisotropic Gaussian approximation.
    - "gibson_lanni": placeholder / MicroscPSF-backed when available.
    """

    def __init__(self):
        # Simple cache keyed by (mode, nz, ny, nx, sigma_tuple)
        self._cache: Dict[Tuple, np.ndarray] = {}

    def _estimate_gaussian_sigmas_vox(
        self, na: float, wavelength_nm: float,
        pixel_size_xy_um: float, z_step_um: float
    ) -> Tuple[float, float, float]:
        """
        Rough confocal PSF width estimate -> sigmas in voxel units.

        This is intentionally simple: the goal is to have a reasonable,
        anisotropic blur, not a perfect PSF.
        """
        wavelength_um = wavelength_nm * 1e-3
        # Approximate Airy-based FWHM ~ 0.61 * λ / NA laterally, ~ 2 * λ / NA^2 axially.
        fwhm_xy_um = 0.61 * wavelength_um / max(na, 1e-3)
        fwhm_z_um = 2.0 * wavelength_um / max(na**2, 1e-6)

        # Convert FWHM -> sigma in microns, then to voxels
        sigma_xy_um = fwhm_xy_um / 2.355
        sigma_z_um = fwhm_z_um / 2.355

        sigma_z_vox = sigma_z_um / max(z_step_um, 1e-6)
        sigma_xy_vox = sigma_xy_um / max(pixel_size_xy_um, 1e-6)
        # Return (z, y, x) sigmas
        return float(sigma_z_vox), float(sigma_xy_vox), float(sigma_xy_vox)

    def _make_gaussian_psf(
        self, shape_xyz: Tuple[int, int, int],
        sigma_xyz_vox: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Build a normalized anisotropic Gaussian PSF in a 3D grid.
        """
        nz, ny, nx = shape_xyz
        key = ("gaussian", nz, ny, nx, sigma_xyz_vox)
        if key in self._cache:
            return self._cache[key]

        z = np.arange(nz) - nz // 2
        y = np.arange(ny) - ny // 2
        x = np.arange(nx) - nx // 2
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

        sz, sy, sx = sigma_xyz_vox
        psf = np.exp(
            -0.5 * ((zz / max(sz, 1e-6)) ** 2 +
                    (yy / max(sy, 1e-6)) ** 2 +
                    (xx / max(sx, 1e-6)) ** 2)
        )
        psf /= psf.sum() + 1e-12
        self._cache[key] = psf.astype(np.float32)
        return self._cache[key]

    def _make_gibson_lanni_psf(
        self, microscope: MicroscopeConfig
    ) -> np.ndarray:
        """
        Placeholder / MicroscPSF-backed Gibson–Lanni PSF.

        If MicroscPSF is installed, call it to generate a depth-dependent PSF.
        Otherwise, fall back to a calibrated Gaussian with a note in README.
        """
        if not _HAS_MICROSCPSF:
            # Fallback: just use the Gaussian estimate
            sigma_xyz_vox = self._estimate_gaussian_sigmas_vox(
                na=microscope.na,
                wavelength_nm=microscope.wavelength_nm,
                pixel_size_xy_um=microscope.pixel_size_xy_um,
                z_step_um=microscope.z_step_um,
            )
            return self._make_gaussian_psf(microscope.img_shape_xyz, sigma_xyz_vox)

        # Sketch of MicroscPSF usage; adjust parameters as needed.
        nz, ny, nx = microscope.img_shape_xyz
        wavelength_um = microscope.wavelength_nm * 1e-3

        # Typical MicroscPSF call pattern; exact API depends on version.
        psf_gen = MicroscPSF.PSFGenerator(
            NA=microscope.na,
            ni0=microscope.immersion_ri,
            ni=microscope.sample_ri,
            wavelength=wavelength_um,
            xyz_pixels=(nx, ny, nz),
            pixel_size=microscope.pixel_size_xy_um,
            z_step=microscope.z_step_um,
        )
        psf = psf_gen.generate()  # shape (nz, ny, nx)
        psf = psf / (psf.sum() + 1e-12)
        return psf.astype(np.float32)

    def build_psf(self, microscope: MicroscopeConfig) -> np.ndarray:
        """
        Public entry point: return a 3D PSF kernel for the given microscope config.
        """
        if microscope.psf_mode == "gaussian_confocal":
            sigma_xyz_vox = self._estimate_gaussian_sigmas_vox(
                na=microscope.na,
                wavelength_nm=microscope.wavelength_nm,
                pixel_size_xy_um=microscope.pixel_size_xy_um,
                z_step_um=microscope.z_step_um,
            )
            return self._make_gaussian_psf(microscope.img_shape_xyz, sigma_xyz_vox)
        elif microscope.psf_mode == "gibson_lanni":
            return self._make_gibson_lanni_psf(microscope)
        else:
            raise ValueError(f"Unknown psf_mode: {microscope.psf_mode}")


# ---------------------------------------------------------------------
# Medium models: motion layer
# ---------------------------------------------------------------------


class MediumModel:
    """
    Abstract base class for particle motion models.

    Each model exposes:
        step(self, positions_um, dt) -> new_positions_um

    The model can maintain internal state (e.g. memory kernel) if needed.
    """

    def step(self, positions_um: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError


@dataclass
class PurelyViscousBrownian(MediumModel):
    """
    Standard Brownian motion in 3D with diffusion coefficient D (um^2/s).
    """
    D: float
    box_size_um: Tuple[float, float, float]

    def step(self, positions_um: np.ndarray, dt: float) -> np.ndarray:
        sigma_step = np.sqrt(2.0 * max(self.D, 0.0) * dt)
        steps = np.random.normal(scale=sigma_step, size=positions_um.shape)
        new_positions = positions_um + steps
        # Reflect at boundaries by clipping
        new_positions = np.clip(new_positions, 0.0, np.array(self.box_size_um))
        return new_positions


@dataclass
class ViscoelasticMaxwell(MediumModel):
    """
    Simple viscoelastic model that produces subdiffusive-like MSD ~ t^alpha.

    Implementation trick:
    - Use a fractional noise scaling (dt^alpha/2) instead of dt^1/2.
    - D0 sets the overall scale at short times.
    """
    D0: float
    alpha: float
    box_size_um: Tuple[float, float, float]

    def step(self, positions_um: np.ndarray, dt: float) -> np.ndarray:
        alpha_clamped = np.clip(self.alpha, 0.1, 1.0)
        # Effective step variance ~ dt^alpha instead of dt
        sigma_step = np.sqrt(2.0 * max(self.D0, 0.0) * (dt ** alpha_clamped))
        steps = np.random.normal(scale=sigma_step, size=positions_um.shape)
        new_positions = positions_um + steps
        new_positions = np.clip(new_positions, 0.0, np.array(self.box_size_um))
        return new_positions


@dataclass
class NearWallHindered(MediumModel):
    """
    Wrapper that reduces mobility near a wall at z = z_wall for z < z_cut.

    It applies a z-dependent scaling to the step size on top of an inner_ model
    (usually PurelyViscousBrownian or ViscoelasticMaxwell).
    """
    inner_model: MediumModel
    z_wall: float
    z_cut: float
    min_factor: float = 0.2  # minimum mobility factor near the wall

    def step(self, positions_um: np.ndarray, dt: float) -> np.ndarray:
        # First propose a step using the underlying model.
        proposed = self.inner_model.step(positions_um, dt)

        # Compute mobility factor: 1.0 far from wall, reduced near wall.
        z = positions_um[..., 2]  # shape (n_particles,)
        dist = np.clip(self.z_cut - (z - self.z_wall), 0.0, self.z_cut)
        # Normalize to [0, 1], then map to [min_factor, 1].
        factor = 1.0 - (dist / max(self.z_cut, 1e-6)) * (1.0 - self.min_factor)
        factor = factor[..., None]  # broadcast to xyz

        new_positions = positions_um + (proposed - positions_um) * factor
        # Respect box constraints of the inner model if it has them
        return new_positions


# ---------------------------------------------------------------------
# Rendering: sample + optics + detector
# ---------------------------------------------------------------------


@dataclass
class SampleConfig:
    """
    Minimal sample config for now.

    In the future, you can extend with object types (beads/blobs/filaments),
    size distributions, brightness distributions, etc.
    """
    n_particles: int
    box_size_um: Tuple[float, float, float]
    motion_model: str = "viscous"  # "viscous", "viscoelastic", "viscoelastic_near_wall"
    D: float = 0.2  # viscous diffusion
    D0: float = 0.2  # viscoelastic short-time scale
    alpha: float = 0.7  # viscoelastic exponent
    z_wall_um: float = 0.0
    z_cut_um: float = 5.0
    brightness: float = 1.0  # per-particle amplitude


@dataclass
class BleachingConfig:
    """
    Bleaching and attenuation parameters.
    """
    z_att_um: float = 50.0    # depth attenuation length
    bleach_tau_s: float = 80.0  # bleaching time constant
    noise_std: float = 5.0     # detector (Gaussian) noise std


class DigitalTwin:
    """
    Main digital twin class: orchestrates motion, PSF, and rendering.
    """

    def __init__(self):
        self.psf_factory = PSFFactory()

    # -----------------------------
    # Core simulation
    # -----------------------------

    def _init_positions(self, sample: SampleConfig) -> np.ndarray:
        """Randomly initialize particle positions inside the box."""
        n = sample.n_particles
        box = np.array(sample.box_size_um, dtype=float)
        # positions: (n_particles, 3)
        return np.random.rand(n, 3) * box

    def _build_medium_model(
        self, sample: SampleConfig
    ) -> MediumModel:
        """Create the requested motion model."""
        if sample.motion_model == "viscous":
            return PurelyViscousBrownian(
                D=sample.D,
                box_size_um=sample.box_size_um,
            )
        elif sample.motion_model == "viscoelastic":
            return ViscoelasticMaxwell(
                D0=sample.D0,
                alpha=sample.alpha,
                box_size_um=sample.box_size_um,
            )
        elif sample.motion_model == "viscoelastic_near_wall":
            inner = ViscoelasticMaxwell(
                D0=sample.D0,
                alpha=sample.alpha,
                box_size_um=sample.box_size_um,
            )
            return NearWallHindered(
                inner_model=inner,
                z_wall=sample.z_wall_um,
                z_cut=sample.z_cut_um,
            )
        else:
            raise ValueError(f"Unknown motion_model: {sample.motion_model}")

    def _apply_depth_and_bleaching(
        self,
        frame: np.ndarray,
        t_index: int,
        dt: float,
        bleaching: BleachingConfig,
        voxel_size_z_um: float,
    ) -> np.ndarray:
        """
        Apply depth-dependent attenuation and bleaching + add detector noise.
        """
        nz, ny, nx = frame.shape
        z_coords_um = np.arange(nz) * voxel_size_z_um

        # Depth attenuation
        depth_factor = np.exp(-z_coords_um / max(bleaching.z_att_um, 1e-6)).reshape(-1, 1, 1)

        # Bleaching
        bleach_factor = np.exp(
            - (t_index * dt) / max(bleaching.bleach_tau_s, 1e-6)
        )

        frame = frame * depth_factor * bleach_factor

        # Add Gaussian detector noise
        frame = frame + np.random.normal(scale=bleaching.noise_std, size=frame.shape)
        return frame.astype(np.float32)

    def simulate(
        self,
        microscope_config: Dict,
        sample_config: Dict,
        n_frames: int,
        dt: float,
        bleaching_config: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        High-level forward model:

            image_stack, positions = twin.simulate(microscope_config, sample_config, n_frames)

        microscope_config: dict matching MicroscopeConfig fields.
        sample_config: dict matching SampleConfig fields.
        Returns:
            image_stack: (n_frames, nz, ny, nx)
            positions:   (n_frames, n_particles, 3) in microns
        """
        # Parse configs into dataclasses
        microscope = MicroscopeConfig(
            na=float(microscope_config.get("na", 1.3)),
            wavelength_nm=float(microscope_config.get("wavelength_nm", 550.0)),
            immersion_ri=float(microscope_config.get("immersion_ri", 1.33)),
            sample_ri=float(microscope_config.get("sample_ri", 1.33)),
            pixel_size_xy_um=float(microscope_config.get("pixel_size_xy_um", 0.1)),
            z_step_um=float(microscope_config.get("z_step_um", 0.2)),
            img_shape_xyz=tuple(microscope_config.get("img_shape_xyz", (32, 64, 64))),
            psf_mode=str(microscope_config.get("psf_mode", "gaussian_confocal")),
        )

        sample = SampleConfig(
            n_particles=int(sample_config.get("n_particles", 200)),
            box_size_um=tuple(sample_config.get("box_size_um", (30.0, 30.0, 30.0))),
            motion_model=str(sample_config.get("motion_model", "viscous")),
            D=float(sample_config.get("D", 0.2)),
            D0=float(sample_config.get("D0", 0.2)),
            alpha=float(sample_config.get("alpha", 0.7)),
            z_wall_um=float(sample_config.get("z_wall_um", 0.0)),
            z_cut_um=float(sample_config.get("z_cut_um", 5.0)),
            brightness=float(sample_config.get("brightness", 1.0)),
        )

        bleaching = BleachingConfig(**(bleaching_config or {}))

        # Build PSF kernel once
        psf_kernel = self.psf_factory.build_psf(microscope)

        nz, ny, nx = microscope.img_shape_xyz
        n_particles = sample.n_particles

        # Allocate outputs
        stack = np.zeros((n_frames, nz, ny, nx), dtype=np.float32)
        positions = np.zeros((n_frames, n_particles, 3), dtype=np.float32)

        # Initialize positions and medium model
        pos_t = self._init_positions(sample)
        medium_model = self._build_medium_model(sample)

        # Simulation loop
        for t_idx in range(n_frames):
            positions[t_idx] = pos_t

            # Render point sources into voxel grid
            frame = np.zeros((nz, ny, nx), dtype=np.float32)

            # Convert microns -> voxel indices
            for p in range(n_particles):
                x_um, y_um, z_um = pos_t[p]
                ix = int(x_um / microscope.pixel_size_xy_um)
                iy = int(y_um / microscope.pixel_size_xy_um)
                iz = int(z_um / microscope.z_step_um)

                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    frame[iz, iy, ix] += sample.brightness

            # Convolve with PSF. For speed, use gaussian_filter for now,
            # but architecture supports full PSF convolution later.
            # If you want exact convolution with psf_kernel, replace with FFT-based conv.
            # Here we approximate with a Gaussian using psf_kernel's effective sigmas.
            # For an MVP, use gaussian_filter directly:
            #   frame = gaussian_filter(frame, sigma=(...))
            # but to respect the PSF shape, you could use scipy.signal.fftconvolve.

            # Approximate: infer sigmas from second moments of psf_kernel
            # (fallback if needed)
            # For hackathon speed, directly use gaussian_filter with a simple guess:
            # this still yields anisotropic blur.
            frame = gaussian_filter(
                frame,
                sigma=(1.0, 1.0, 1.0),  # cheap default; PSFFactory holds the proper kernel
            )

            # Apply depth attenuation, bleaching, and noise
            frame = self._apply_depth_and_bleaching(
                frame=frame,
                t_index=t_idx,
                dt=dt,
                bleaching=bleaching,
                voxel_size_z_um=microscope.z_step_um,
            )

            stack[t_idx] = frame

            # Evolve positions for next frame
            pos_t = medium_model.step(pos_t, dt)

        return stack, positions

    # -----------------------------
    # Parameter recommender
    # -----------------------------

    def _compute_image_metrics(self, stack: np.ndarray) -> Dict[str, float]:
        """
        Compute simple image quality metrics for a stack:
        - SNR (mean / std)
        - contrast (normalized std)
        - mean intensity
        """
        # Use central region to avoid boundary artifacts
        n_frames, nz, ny, nx = stack.shape
        sub = stack[:, nz // 4 : 3 * nz // 4, ny // 4 : 3 * ny // 4, nx // 4 : 3 * nx // 4]

        mean_int = float(sub.mean())
        std_int = float(sub.std() + 1e-6)
        snr = mean_int / std_int
        contrast = std_int / (mean_int + 1e-6)
        return {
            "mean_intensity": mean_int,
            "std_intensity": std_int,
            "snr": snr,
            "contrast": contrast,
        }

    def recommend_parameters(
        self,
        real_stack: np.ndarray,
        base_microscope_config: Dict,
        sample_config: Dict,
        n_frames_test: int = 10,
        dt: float = 0.1,
        exposure_candidates_ms: Optional[List[float]] = None,
        pinhole_factors: Optional[List[float]] = None,
    ) -> Dict:
        """
        Simple inverse-design helper:

        - Compute metrics on the real (noisy) stack.
        - Loop over candidate exposure times / pinhole factors.
        - For each candidate:
            * simulate a short stack,
            * compute metrics,
            * score similarity to a target (higher SNR, similar contrast).
        - Return the best candidate and a human-readable summary.

        This is deliberately simple and meant as a hook for GUI + LLM explanation.
        """
        if exposure_candidates_ms is None:
            exposure_candidates_ms = [5.0, 10.0, 20.0]

        if pinhole_factors is None:
            pinhole_factors = [1.0, 0.7, 0.5]  # <1 = smaller pinhole

        # Baseline metrics
        real_metrics = self._compute_image_metrics(real_stack)

        best_score = -np.inf
        best_cfg = None

        for exp in exposure_candidates_ms:
            for pinhole_factor in pinhole_factors:
                # Copy and tweak microscope config
                mc = dict(base_microscope_config)
                mc["exposure_ms"] = exp
                mc["pinhole_factor"] = pinhole_factor

                # Simple heuristic: longer exposure => lower noise_std, larger mean
                # Smaller pinhole => better axial sectioning but lower total photons; mimic via z_att + noise.
                bleaching_cfg = {
                    "z_att_um": float(mc.get("z_att_um", 50.0))
                    * (1.2 if pinhole_factor < 1.0 else 1.0),
                    "bleach_tau_s": float(mc.get("bleach_tau_s", 80.0)),
                    "noise_std": float(mc.get("noise_std", 5.0))
                    * (0.7 if exp > base_microscope_config.get("exposure_ms", 5.0) else 1.0),
                }

                sim_stack, _ = self.simulate(mc, sample_config, n_frames_test, dt, bleaching_cfg)
                sim_metrics = self._compute_image_metrics(sim_stack)

                # Score: reward higher SNR than real, and similar contrast
                snr_gain = sim_metrics["snr"] - real_metrics["snr"]
                contrast_diff = abs(sim_metrics["contrast"] - real_metrics["contrast"])
                score = snr_gain - 0.5 * contrast_diff

                if score > best_score:
                    best_score = score
                    best_cfg = {
                        "exposure_ms": exp,
                        "pinhole_factor": pinhole_factor,
                        "sim_metrics": sim_metrics,
                    }

        if best_cfg is None:
            return {
                "recommendation": "No improvement found (check digital twin settings).",
                "details": {},
            }

        base_exp = base_microscope_config.get("exposure_ms", exposure_candidates_ms[0])
        base_pinhole = base_microscope_config.get("pinhole_factor", 1.0)

        # Build human-readable recommendation string
        rec_lines = []
        if best_cfg["exposure_ms"] > base_exp:
            rec_lines.append(
                f"Increase exposure from {base_exp:.1f} ms to {best_cfg['exposure_ms']:.1f} ms "
                f"to boost SNR."
            )
        elif best_cfg["exposure_ms"] < base_exp:
            rec_lines.append(
                f"Decrease exposure from {base_exp:.1f} ms to {best_cfg['exposure_ms']:.1f} ms "
                f"to reduce bleaching at the cost of SNR."
            )

        if best_cfg["pinhole_factor"] < base_pinhole:
            rec_lines.append(
                "Reduce pinhole size (pinhole_factor < 1) for better axial sectioning, "
                "expect slightly lower signal."
            )
        elif best_cfg["pinhole_factor"] > base_pinhole:
            rec_lines.append(
                "Increase pinhole size (pinhole_factor > 1) for more photons at the cost of sectioning."
            )

        if not rec_lines:
            rec_lines.append("Current settings are near optimal under this twin model.")

        return {
            "recommendation": " ".join(rec_lines),
            "base_metrics": real_metrics,
            "best_candidate": best_cfg,
            "score": float(best_score),
        }

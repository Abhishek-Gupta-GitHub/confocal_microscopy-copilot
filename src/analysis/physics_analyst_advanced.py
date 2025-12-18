# src/analysis/physics_analyst_advanced.py

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np
from scipy import spatial

# ---------------------- data structures ---------------------- #


@dataclass
class Trajectory:
    """Universal trajectory representation (2D/3D + annotations)."""
    particle_id: int
    t: np.ndarray
    x: np.ndarray
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    radius: float = 0.5
    label: Optional[str] = None
    dim: int = 3
    tags: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)

    def length(self) -> int:
        return len(self.t)

    def duration(self):
        return self.t[-1] - self.t[0]

    def positions(self) -> np.ndarray:
        if self.dim == 2:
            if self.y is None:
                raise ValueError("dim=2 but y is None")
            return np.column_stack([self.x, self.y])
        if self.y is None or self.z is None:
            raise ValueError("dim=3 but y or z is None")
        return np.column_stack([self.x, self.y, self.z])


@dataclass
class HeterogeneityMetrics:
    """Non-Gaussian and heterogeneity descriptors."""
    alpha2_max: Optional[float] = None
    alpha2_tau_peak: Optional[float] = None
    msd_distribution_std: Optional[float] = None
    msd_distribution_mean: Optional[float] = None
    subdiffusion_detected: bool = False
    superdiffusion_detected: bool = False
    dynamic_susceptibility: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class StructureAnalysis:
    """Static structure and correlation functions."""
    q_values: List[float]
    structure_factor: List[float]
    pair_correlation: Optional[List[float]] = None
    distances: Optional[List[float]] = None
    density: Optional[float] = None
    clustering_detected: bool = False

    def to_dict(self):
        return asdict(self)


# ------------------------ main analyst ------------------------ #


class PhysicsAnalystAdvanced:
    """
    Multi-domain physics analyzer for confocal microscopy trajectories.
    Computes MSD variants, heterogeneity metrics, and structure functions.
    Interface-compatible with PhysicsAnalystBasic.summarize_physics.
    """

    def __init__(
        self,
        dt: float = 1.0,
        px_to_um: float = 1.0,
        wall_z_threshold: float = 2.0,
        dim: int = 3,
    ):
        self.dt = dt
        self.px_to_um = px_to_um
        self.wall_z_threshold = wall_z_threshold
        self.dim = dim

    # -------------------------- MSD -------------------------- #

    def compute_msd(
        self,
        trajectory: Trajectory,
        max_lag_frames: Optional[int] = None,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos = trajectory.positions() * self.px_to_um
        N = len(pos)

        if max_lag_frames is None:
            max_lag_frames = max(N // 4, 1)
        max_lag_frames = min(max_lag_frames, N - 1)

        lag_frames = np.arange(0, max_lag_frames + 1, stride)
        msd = np.zeros(len(lag_frames))

        for i, tau in enumerate(lag_frames):
            if tau == 0:
                msd[i] = 0.0
            else:
                disp = pos[tau:] - pos[:-tau]
                msd[i] = np.mean(np.sum(disp ** 2, axis=1))

        lag_times = lag_frames * self.dt
        return lag_times, msd

    # ---------------------- near-wall MSD --------------------- #

    def compute_wall_proximity_mask(self, trajectory: Trajectory) -> np.ndarray:
        if self.dim == 2 or trajectory.z is None:
            # no meaningful z → treat everything as bulk
            return np.zeros_like(trajectory.x, dtype=bool)
        z_scaled = trajectory.z * self.px_to_um
        return z_scaled < self.wall_z_threshold

    def compute_near_wall_msd(
        self,
        trajectory: Trajectory,
        max_lag_frames: Optional[int] = None,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = trajectory.positions() * self.px_to_um
        wall_mask = self.compute_wall_proximity_mask(trajectory)

        N = len(pos)
        if max_lag_frames is None:
            max_lag_frames = max(N // 4, 1)
        max_lag_frames = min(max_lag_frames, N - 1)

        lag_frames = np.arange(0, max_lag_frames + 1, stride)

        msd_wall = np.zeros(len(lag_frames))
        msd_bulk = np.zeros(len(lag_frames))

        for i, tau in enumerate(lag_frames):
            if tau == 0:
                msd_wall[i] = 0.0
                msd_bulk[i] = 0.0
                continue

            disp = pos[tau:] - pos[:-tau]
            disp_sq = np.sum(disp ** 2, axis=1)

            wall_starts = wall_mask[:-tau]
            bulk_starts = ~wall_mask[:-tau]

            if wall_starts.sum() > 0:
                msd_wall[i] = np.mean(disp_sq[wall_starts])

            if bulk_starts.sum() > 0:
                msd_bulk[i] = np.mean(disp_sq[bulk_starts])

        lag_times = lag_frames * self.dt
        return lag_times, msd_wall, msd_bulk

    # -------------------- cage-relative MSD ------------------- #

    def compute_cage_relative_msd(
        self,
        trajectories: List[Trajectory],
        reference_traj_idx: int,
        cage_radius: float = 5.0,
        max_lag_frames: Optional[int] = None,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref_traj = trajectories[reference_traj_idx]
        ref_pos = ref_traj.positions() * self.px_to_um
        N = len(ref_pos)

        if max_lag_frames is None:
            max_lag_frames = max(N // 4, 1)

        # neighbors at t0
        t0_ref = ref_pos[0]
        cage_neighbors: List[int] = []
        for j, traj in enumerate(trajectories):
            if j == reference_traj_idx:
                continue
            other_pos = traj.positions() * self.px_to_um
            dist = np.linalg.norm(other_pos[0] - t0_ref)
            if dist < cage_radius:
                cage_neighbors.append(j)

        if not cage_neighbors:
            lag_times, msd_bulk = self.compute_msd(ref_traj, max_lag_frames, stride)
            return lag_times, msd_bulk.copy(), msd_bulk.copy()

        neighbor_positions = np.stack(
            [trajectories[j].positions() * self.px_to_um for j in cage_neighbors],
            axis=1,
        )  # (N_frames, N_neighbors, d)

        cage_com = np.mean(neighbor_positions, axis=1)  # (N_frames, d)
        rel_pos = ref_pos - cage_com

        max_lag_frames = min(max_lag_frames, N - 1)
        lag_frames = np.arange(0, max_lag_frames + 1, stride)

        msd_cage = np.zeros(len(lag_frames))
        msd_bulk = np.zeros(len(lag_frames))

        for i, tau in enumerate(lag_frames):
            if tau == 0:
                msd_cage[i] = 0.0
                msd_bulk[i] = 0.0
                continue

            disp_bulk = ref_pos[tau:] - ref_pos[:-tau]
            msd_bulk[i] = np.mean(np.sum(disp_bulk ** 2, axis=1))

            disp_rel = rel_pos[tau:] - rel_pos[:-tau]
            msd_cage[i] = np.mean(np.sum(disp_rel ** 2, axis=1))

        lag_times = lag_frames * self.dt
        return lag_times, msd_cage, msd_bulk

    # ----------------- heterogeneity metrics ------------------ #

    def fit_msd_power_law(
        self,
        lag_times: np.ndarray,
        msd: np.ndarray,
        window_fraction: float = 0.3,
    ) -> Tuple[float, float]:
        n_points = max(int(len(lag_times) * window_fraction), 2)
        idx = slice(1, n_points + 1)

        log_tau = np.log(lag_times[idx])
        log_msd = np.log(np.maximum(msd[idx], 1e-10))

        coeffs = np.polyfit(log_tau, log_msd, 1)
        alpha = float(coeffs[0])
        D_alpha = float(np.exp(coeffs[1]) / (2 * self.dim))
        return alpha, D_alpha

    def compute_non_gaussian_parameter(
        self,
        trajectories: List[Trajectory],
        max_lag_frames: Optional[int] = None,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not trajectories:
            return np.array([0.0]), np.array([0.0])

        N_frames = max(len(t.t) for t in trajectories)
        if max_lag_frames is None:
            max_lag_frames = max(N_frames // 4, 1)
        max_lag_frames = min(max_lag_frames, N_frames - 1)

        lag_frames = np.arange(0, max_lag_frames + 1, stride)
        alpha2 = np.zeros(len(lag_frames))

        for i, tau in enumerate(lag_frames):
            if tau == 0:
                alpha2[i] = 0.0
                continue

            r2_list: List[float] = []
            r4_list: List[float] = []

            for traj in trajectories:
                if len(traj.t) <= tau:
                    continue
                pos = traj.positions() * self.px_to_um
                disp = pos[tau:] - pos[:-tau]
                r2 = np.sum(disp ** 2, axis=1)
                r2_list.extend(r2)
                r4_list.extend(r2 ** 2)

            if len(r2_list) > 1:
                r2_mean = float(np.mean(r2_list))
                r4_mean = float(np.mean(r4_list))
                alpha2[i] = (r4_mean / (3 * r2_mean ** 2)) - 1.0 if r2_mean > 0 else 0.0
            else:
                alpha2[i] = 0.0

        lag_times = lag_frames * self.dt
        return lag_times, alpha2

    def compute_heterogeneity_metrics(
        self,
        trajectories: List[Trajectory],
        max_lag_frames: Optional[int] = None,
    ) -> HeterogeneityMetrics:
        if not trajectories:
            return HeterogeneityMetrics()

        alphas: List[float] = []
        msd_at_long_tau: List[float] = []

        for traj in trajectories:
            lag_times, msd = self.compute_msd(traj, max_lag_frames)
            if len(lag_times) > 1:
                alpha, _ = self.fit_msd_power_law(lag_times, msd, window_fraction=0.3)
                alphas.append(alpha)
                msd_at_long_tau.append(float(msd[-1]))

        lag_times, alpha2_array = self.compute_non_gaussian_parameter(
            trajectories, max_lag_frames
        )
        alpha2_max = float(np.max(alpha2_array)) if len(alpha2_array) > 0 else None
        alpha2_tau_peak = (
            float(lag_times[np.argmax(alpha2_array)]) if len(alpha2_array) > 0 else None
        )

        msd_mean = float(np.mean(msd_at_long_tau)) if msd_at_long_tau else None
        msd_std = float(np.std(msd_at_long_tau)) if msd_at_long_tau else None

        mean_alpha = float(np.mean(alphas)) if alphas else 1.0
        subdiffusion = bool(mean_alpha < 0.9)
        superdiffusion = bool(mean_alpha > 1.1)

        dynamic_susceptibility = (
            msd_std / msd_mean if (msd_mean is not None and msd_std is not None and msd_mean != 0) else None
        )

        return HeterogeneityMetrics(
            alpha2_max=alpha2_max,
            alpha2_tau_peak=alpha2_tau_peak,
            msd_distribution_std=msd_std,
            msd_distribution_mean=msd_mean,
            subdiffusion_detected=subdiffusion,
            superdiffusion_detected=superdiffusion,
            dynamic_susceptibility=dynamic_susceptibility,
        )

    # -------------------- structure functions ------------------ #

    def compute_static_structure_factor(
        self,
        trajectories: List[Trajectory],
        time_window_fraction: float = 0.5,
        n_q_bins: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not trajectories:
            return np.array([0.0]), np.array([0.0])

        all_positions: List[np.ndarray] = []
        for traj in trajectories:
            pos = traj.positions() * self.px_to_um
            t_start = int(len(pos) * (1 - time_window_fraction))
            all_positions.append(np.mean(pos[t_start:], axis=0))

        positions = np.array(all_positions)
        N = len(positions)
        if N < 2:
            return np.array([0.0]), np.array([1.0])

        system_size = float(np.max(np.ptp(positions, axis=0)))
        max_q = 2 * np.pi / (system_size / 10.0)
        q_values = np.linspace(0.1, max_q, n_q_bins)

        S_q = np.ones_like(q_values)
        for i, q_mag in enumerate(q_values):
            theta = np.random.rand() * 2 * np.pi
            phi = np.random.rand() * np.pi
            q_vec = q_mag * np.array(
                [
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ]
            )
            phase = np.dot(positions, q_vec)
            sum_exp = np.sum(np.exp(1j * phase))
            S_q[i] = 1.0 + (np.abs(sum_exp) ** 2 / N)

        return q_values, S_q

    def compute_pair_correlation(
        self,
        trajectories: List[Trajectory],
        time_window_fraction: float = 0.5,
        n_r_bins: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not trajectories:
            return np.array([0.0]), np.array([0.0])

        all_positions: List[np.ndarray] = []
        for traj in trajectories:
            pos = traj.positions() * self.px_to_um
            t_start = int(len(pos) * (1 - time_window_fraction))
            all_positions.append(np.mean(pos[t_start:], axis=0))

        positions = np.array(all_positions)
        N = len(positions)
        if N < 2:
            return np.array([0.0]), np.array([0.0])

        distances = spatial.distance.pdist(positions, metric="euclidean")

        r_max = float(np.max(distances))
        r_bins = np.linspace(0.0, r_max, n_r_bins)
        g_r, _ = np.histogram(distances, bins=r_bins, density=False)

        dr = r_bins[1] - r_bins[0] if len(r_bins) > 1 else 1.0
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
        bin_volume = 4.0 * np.pi * r_centers ** 2 * dr

        extents = np.ptp(positions, axis=0)
        density = N / (extents.prod() if np.all(extents > 0) else 1.0)
        expected = bin_volume * density * (N - 1) / 2.0

        g_r_norm = np.where(expected > 0, g_r / expected, 0.0)
        return r_centers, g_r_norm

    def compute_structure_analysis(
        self,
        trajectories: List[Trajectory],
    ) -> StructureAnalysis:
        q_vals, S_q = self.compute_static_structure_factor(trajectories)
        r_vals, g_r = self.compute_pair_correlation(trajectories)

        volumes = [np.ptp(t.positions(), axis=0).prod() for t in trajectories]
        density = len(trajectories) / (np.mean(volumes) * (self.px_to_um ** self.dim) + 1e-10)

        clustering = bool(S_q[0] > 1.5) if len(S_q) > 0 else False

        return StructureAnalysis(
            q_values=q_vals.tolist(),
            structure_factor=S_q.tolist(),
            pair_correlation=g_r.tolist(),
            distances=r_vals.tolist(),
            density=float(density),
            clustering_detected=clustering,
        )

    # --------------------- JSON-style summary ------------------ #

    def summarize_physics(
        self,
        trajectories: List[Trajectory],
        domain: str = "general",
    ) -> Dict[str, Any]:
        if not trajectories:
            return {"error": "No trajectories provided"}

        # 1. bulk MSD
        ref_traj = trajectories[0]
        lag_times, msd_bulk = self.compute_msd(ref_traj)
        alpha_bulk, D_bulk = self.fit_msd_power_law(lag_times, msd_bulk)

        # 2. near-wall
        lag_times_wall, msd_wall, msd_bulk_full = self.compute_near_wall_msd(ref_traj)
        alpha_wall, _ = self.fit_msd_power_law(
            lag_times_wall, np.maximum(msd_wall, 1e-10)
        )

        # 3. cage-relative
        if len(trajectories) > 1:
            lag_times_cage, msd_cage, msd_bulk_cage = self.compute_cage_relative_msd(
                trajectories, reference_traj_idx=0
            )
            alpha_cage, _ = self.fit_msd_power_law(lag_times_cage, msd_cage)
        else:
            lag_times_cage, msd_cage, msd_bulk_cage = (
                lag_times,
                msd_bulk.copy(),
                msd_bulk.copy(),
            )
            alpha_cage = alpha_bulk

        # 4. heterogeneity
        hetero = self.compute_heterogeneity_metrics(trajectories)

        # 5. structure
        struct = self.compute_structure_analysis(trajectories)

        summary: Dict[str, Any] = {
            "metadata": {
                "domain": domain,
                "n_trajectories": len(trajectories),
                "mean_trajectory_length": float(
                    np.mean([len(t.t) for t in trajectories])
                ),
                "dt_physical_units": self.dt,
                "px_to_um_conversion": self.px_to_um,
                "analyzer_type": "advanced",
            },
            "msd_analysis": {
                "bulk": {
                    "lag_times": lag_times.tolist(),
                    "msd_values": msd_bulk.tolist(),
                    "anomaly_exponent_alpha": float(alpha_bulk),
                    "diffusion_coefficient": float(D_bulk),
                    "interpretation": self._interpret_msd_exponent(alpha_bulk),
                },
                "near_wall": {
                    "lag_times": lag_times_wall.tolist(),
                    "msd_wall": msd_wall.tolist(),
                    "msd_bulk": msd_bulk_full.tolist(),
                    "wall_hindrance_ratio": float(
                        np.mean(
                            msd_bulk_full[1:] / (msd_wall[1:] + 1e-10)
                        )
                    )
                    if len(msd_wall) > 1
                    else 1.0,
                    "anomaly_exponent_wall": float(alpha_wall),
                },
                "cage_relative": {
                    "lag_times": lag_times_cage.tolist(),
                    "msd_cage": msd_cage.tolist(),
                    "msd_reference": msd_bulk_cage.tolist(),
                    "anomaly_exponent_cage": float(alpha_cage),
                },
            },
            "heterogeneity": asdict(hetero),
            "structure": asdict(struct),
            "flags_and_anomalies": self._generate_flags(
                alpha_bulk, hetero, struct, domain
            ),
        }
        return summary

    # ----------------------- helpers -------------------------- #

    def _interpret_msd_exponent(self, alpha: float) -> str:
        if alpha < 0.8:
            return "Strong subdiffusion (possibly caged/glassy)"
        elif alpha < 1.0:
            return "Subdiffusion (viscoelastic or confined)"
        elif 0.95 < alpha < 1.05:
            return "Brownian diffusion (normal)"
        elif alpha <= 1.2:
            return "Slight superdiffusion (transient dynamics)"
        else:
            return "Strong superdiffusion (active or ballistic)"

    def _generate_flags(
        self,
        alpha: float,
        hetero: HeterogeneityMetrics,
        struct: StructureAnalysis,
        domain: str,
    ) -> List[str]:
        flags: List[str] = []

        if hetero.subdiffusion_detected:
            flags.append(
                "subdiffusion_detected: alpha < 0.9 (viscoelasticity or confinement)"
            )
        if hetero.superdiffusion_detected:
            flags.append(
                "superdiffusion_detected: alpha > 1.1 (active transport or drift)"
            )

        if hetero.alpha2_max is not None and hetero.alpha2_max > 0.1:
            flags.append(
                f"non_gaussian_dynamics: alpha2_max = {hetero.alpha2_max:.3f} "
                f"at tau = {hetero.alpha2_tau_peak}"
            )

        if (
            hetero.dynamic_susceptibility is not None
            and hetero.dynamic_susceptibility > 0.3
        ):
            flags.append(
                f"strong_heterogeneity: MSD variance/mean ratio = {hetero.dynamic_susceptibility:.2f}"
            )

        if struct.clustering_detected:
            flags.append(
                f"spatial_clustering: S(0) = {struct.structure_factor[0]:.2f} "
                "(non-random distribution)"
            )

        if domain == "colloidal" and hetero.subdiffusion_detected:
            flags.append(
                "colloidal_context: subdiffusion may indicate gel formation or jamming"
            )
        elif domain == "cellular" and struct.clustering_detected:
            flags.append(
                "cellular_context: clustering may indicate organelle co-localization"
            )
        elif (
            domain == "polymer"
            and hetero.alpha2_max is not None
            and hetero.alpha2_max > 0.2
        ):
            flags.append(
                "polymer_context: non-Gaussian dynamics consistent with polymer chain dynamics"
            )

        return flags or ["system_behaves_normally_no_anomalies_detected"]


# ------------------- helper: numpy→native -------------------- #

def numpy_to_native(obj):
    import numpy as _np
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    return obj

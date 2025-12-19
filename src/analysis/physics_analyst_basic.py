# src/analysis/physics_analyst_basic.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class Trajectory:
    """Minimal trajectory representation for basic MSD analysis."""
    particle_id: int
    t: np.ndarray
    x: np.ndarray
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None

    def positions(self) -> np.ndarray:
        """Return positions as (N, d) with d inferred from provided arrays."""
        if self.z is not None and self.y is not None:
            return np.column_stack([self.x, self.y, self.z])
        if self.y is not None:
            return np.column_stack([self.x, self.y])
        return self.x[:, None]


class PhysicsAnalystBasic:
    """
    Basic physics analyzer: bulk MSD + power-law fit.
    Interface-compatible with PhysicsAnalystAdvanced.summarize_physics.
    """

    def __init__(self, dt: float = 1.0, px_to_um: float = 1.0, dim: int = 3):
        self.dt = dt
        self.px_to_um = px_to_um
        self.dim = dim

    # ------------------------- core MSD ------------------------- #

    def compute_msd(
        self,
        trajectory: Trajectory,
        max_lag_frames: Optional[int] = None,
        stride: int = 1,
    ) -> (np.ndarray, np.ndarray):
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

    def fit_msd_power_law(
        self,
        lag_times: np.ndarray,
        msd: np.ndarray,
        window_fraction: float = 0.3,
    ) -> (float, float):
        n_points = max(int(len(lag_times) * window_fraction), 2)
        idx = slice(1, n_points + 1)

        log_tau = np.log(lag_times[idx])
        log_msd = np.log(np.maximum(msd[idx], 1e-10))

        coeffs = np.polyfit(log_tau, log_msd, 1)
        alpha = float(coeffs[0])
        D_alpha = float(np.exp(coeffs[1]) / (2 * self.dim))
        return alpha, D_alpha

    # ------------------------- public API ------------------------- #

    def summarize_physics(
        self,
        trajectories: List[Trajectory],
        domain: str = "general",
    ) -> Dict[str, Any]:
        """
        Minimal JSON-compatible summary:
        - metadata
        - msd_analysis.bulk (lag_times, msd_values, alpha, D, interpretation)
        - heterogeneity: empty
        - structure: empty
        - flags_and_anomalies: simple interpretation string
        """
        if not trajectories:
            return {"error": "No trajectories provided"}

        ref_traj = trajectories[0]
        lag_times, msd_bulk = self.compute_msd(ref_traj)
        alpha, D = self.fit_msd_power_law(lag_times, msd_bulk)

        summary = {
            "metadata": {
                "domain": domain,
                "n_trajectories": len(trajectories),
                "mean_trajectory_length": float(
                    np.mean([len(t.t) for t in trajectories])
                ),
                "dt_physical_units": self.dt,
                "px_to_um_conversion": self.px_to_um,
                "analyzer_type": "basic",
            },
            "msd_analysis": {
                "bulk": {
                    "lag_times": lag_times.tolist(),
                    "msd_values": msd_bulk.tolist(),
                    "anomaly_exponent_alpha": float(alpha),
                    "diffusion_coefficient": float(D),
                    "interpretation": self._interpret_msd_exponent(alpha),
                },
                "near_wall": None,
                "cage_relative": None,
            },
            "heterogeneity": {},
            "structure": {},
            "flags_and_anomalies": [self._interpret_msd_exponent(alpha)],
        }

        return summary

    # ------------------------- helpers ------------------------- #

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

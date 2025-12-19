import numpy as np
import trackpy as tp
import pandas as pd

class PhysicsAnalyst:
    def __init__(self):
        pass

    def compute_msd_alpha_D(self, trajectories, frame_interval_s):
        if trajectories is None or trajectories.empty:
            return None, np.nan, np.nan

        # Ensure unique (particle, frame) pairs to avoid msd_gaps error
        traj = trajectories.sort_values(["particle", "frame"]).copy()
        traj = traj.drop_duplicates(subset=["particle", "frame"])

        try:
            msd_series = tp.motion.msd(
                traj, mpp=1.0, fps=1.0 / frame_interval_s
            )
        except Exception as e:
            print("MSD computation failed:", e)
            return None, np.nan, np.nan

        taus = msd_series.index.values
        msd_vals = msd_series["msd"].values
        mask = (taus > 0) & np.isfinite(msd_vals)
        if mask.sum() < 3:
            return msd_series, np.nan, np.nan

        log_t = np.log10(taus[mask])
        log_msd = np.log10(msd_vals[mask])
        coeffs = np.polyfit(log_t, log_msd, 1)
        alpha = float(coeffs[0])
        D = (10 ** coeffs[1]) / (4 * 1.0 ** alpha)
        return msd_series, alpha, float(D)

    def depth_profile(self, stack_3d):
        return stack_3d.mean(axis=(0, 2, 3)).tolist()

    def bleaching_curve(self, stack_3d):
        return stack_3d.mean(axis=(1, 2, 3)).tolist()

    def crowding_metric(self, trajectories):
        nndists = []
        for _, group in trajectories.groupby("frame"):
            coords = group[["x", "y"]].values
            if len(coords) < 2:
                continue
            dists = np.linalg.norm(
                coords[:, None, :] - coords[None, :, :], axis=-1
            )
            dists += np.eye(dists.shape[0]) * 1e9
            nn = dists.min(axis=1)
            nndists.extend(nn.tolist())
        if not nndists:
            return {"mean_nn": None, "n_points": 0}
        return {"mean_nn": float(np.mean(nndists)), "n_points": len(nndists)}
    
    import trackpy as tp

    def _prepare_for_msd(traj: pd.DataFrame) -> pd.DataFrame:
        # only keep necessary columns
        cols = [c for c in ["particle", "frame", "x", "y", "z"] if c in traj.columns]
        df = traj[cols].copy()

        # sort and drop any exact duplicates
        df = df.sort_values(["particle", "frame"]).drop_duplicates(
            subset=["particle", "frame"]
        )

        # OPTIONAL: reindex particles densely (0..N-1)
        new_ids = {old: i for i, old in enumerate(df["particle"].unique())}
        df["particle"] = df["particle"].map(new_ids)

        return df

    def compute_emsd(traj: pd.DataFrame, mpp: float, fps: float, max_lag=100):
        df = _prepare_for_msd(traj)

        # final sanity check: if any (frame, particle) duplicates remain, abort gracefully
        dup = df.duplicated(subset=["particle", "frame"])
        if dup.any():
            raise RuntimeError(
                f"Still have duplicate (particle, frame) rows: {dup.sum()} duplicates."
            )

        # standard trackpy call
        em = tp.emsd(df, mpp=mpp, fps=fps, max_lagtime=max_lag, detail=False)
        return em


    def summarize(self, trajectories, stack_3d, metadata):
        frame_interval_s = float(metadata.get("frame_interval_s", 0.1))
        msd_series, alpha, D = self.compute_msd_alpha_D(
            trajectories, frame_interval_s
        )

        summary = {
            "msd": None if msd_series is None else {
                "taus_s": msd_series.index.values.tolist(),
                "values": msd_series["msd"].values.tolist(),
            },
            "alpha": alpha,
            "D": D,
            "diagnostics": {
                "depth_profile_mean_intensity": self.depth_profile(stack_3d),
                "bleaching_mean_intensity": self.bleaching_curve(stack_3d),
                "crowding": self.crowding_metric(trajectories),
            },
        }
        return summary


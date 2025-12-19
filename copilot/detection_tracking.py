# copilot/detection_tracking.py

import pandas as pd
import numpy as np
import trackpy as tp

from src.analysis.physics_analyst_basic import Trajectory
from copilot.detection_deeptrack import DeepTrackDetector, DeepTrackConfig


class DetectionTrackingWorker:
    def __init__(self):
        pass

    def _project_to_2d(self, stack_3d):
        # stack_3d: (T, Z, Y, X)
        return stack_3d.max(axis=1)  # (T, Y, X)

    def run(self, stack_3d, plan):
        frames_2d = self._project_to_2d(stack_3d)

        # Loosen parameters a bit for synthetic data
        diameter = int(2 * plan.detection_params_initial["max_sigma"] + 1)
        minmass = plan.detection_params_initial["minmass"]

        f_list = []
        for t, frame in enumerate(frames_2d):
            f = tp.locate(
                frame,
                diameter=diameter,
                minmass=minmass,
            )
            if len(f) == 0:
                continue
            f["frame"] = t
            f_list.append(f)

        print("Frames with detections:", len(f_list))

        # NEW: guard against no detections
        if not f_list:
            print("No features detected in any frame; returning empty trajectories.")
            empty = pd.DataFrame(columns=["x", "y", "frame", "particle"])
            return {
                "trajectories": empty,
                "quality_metrics": {
                    "n_tracks": 0,
                    "track_length_hist": {},
                    "detections_per_frame": {},
                },
                "used_params": {
                    "detection": plan.detection_params_initial,
                    "tracking": plan.tracking_params_initial,
                },
            }

        features = pd.concat(f_list, ignore_index=True)

        # NEW EXTRA GUARD
        if features.empty:
            print("Features DataFrame is empty; skipping linking.")
            empty = pd.DataFrame(columns=["x", "y", "frame", "particle"])
            return {
                "trajectories": empty,
                "quality_metrics": {
                    "n_tracks": 0,
                    "track_length_hist": {},
                    "detections_per_frame": {},
                },
                "used_params": {
                    "detection": plan.detection_params_initial,
                    "tracking": plan.tracking_params_initial,
                },
            }


        trajectories = tp.link_df(
            features,
            search_range=plan.tracking_params_initial["search_range"],
            memory=plan.tracking_params_initial["memory"],
        )

        n_tracks = trajectories["particle"].nunique()
        track_lengths = trajectories.groupby("particle")["frame"].count()
        track_length_hist = track_lengths.value_counts().to_dict()
        detections_per_frame = features.groupby("frame").size().to_dict()

        return {
            "trajectories": trajectories,
            "quality_metrics": {
                "n_tracks": int(n_tracks),
                "track_length_hist": track_length_hist,
                "detections_per_frame": detections_per_frame,
            },
            "used_params": {
                "detection": plan.detection_params_initial,
                "tracking": plan.tracking_params_initial,
            },
        }
    def run_with_deeptrack(
        self,
        stack_3d: np.ndarray,
        plan,
    ) -> dict:
        """
        Use DeepTrack for feature finding (2D+t projection),
        then Trackpy for linking, then convert to Trajectory list.
        """
        # project to 2D+t (same as you already do)
        stack_2d_t = self._project_to_2d(stack_3d)  # shape (T, H, W)

        dt_detector = DeepTrackDetector(DeepTrackConfig(image_shape=stack_2d_t.shape[1:]))
        det_result = dt_detector.detect_2d_t(
            stack_2d_t,
            threshold=plan.detection_params_initial.get("threshold_rel", 0.5),
        )
        positions = det_result["positions"]

        if positions.size == 0:
            return {
                "trajectories": [],
                "quality_metrics": {
                    "n_tracks": 0,
                    "detections_per_frame": {},
                    "backend": det_result["meta"]["backend"],
                },
            }

        # Convert to DataFrame expected by trackpy: columns ['frame','x','y']
        df = pd.DataFrame(
            {
                "frame": positions[:, 0].astype(int),
                "y": positions[:, 1],
                "x": positions[:, 2],
            }
        )

        linked = tp.link_df(
            df,
            search_range=plan.tracking_params_initial.get("search_range", 5),
            memory=plan.tracking_params_initial.get("memory", 2),
        )

        trajectories = self._df_to_trajectories(linked, stack_3d.shape)

        detections_per_frame = df["frame"].value_counts().sort_index().to_dict()
        quality_metrics = {
            "n_tracks": int(linked["particle"].nunique()),
            "detections_per_frame": detections_per_frame,
            "backend": det_result["meta"]["backend"],
        }

        return {
            "trajectories": trajectories,
            "quality_metrics": quality_metrics,
        }

    def _df_to_trajectories(self, linked: pd.DataFrame, stack_shape) -> list[Trajectory]:
        T, Z, H, W = stack_shape
        trajectories: list[Trajectory] = []
        for pid, grp in linked.groupby("particle"):
            grp = grp.sort_values("frame")
            times = grp["frame"].to_numpy()
            xs = grp["x"].to_numpy()
            ys = grp["y"].to_numpy()
            for t, x, y in zip(times, xs, ys):
                trajectories.append(
                    Trajectory(
                        id=int(pid),
                        t=float(t),
                        x=float(x),
                        y=float(y),
                        z=None,
                        frame=int(t),
                        meta={},
                    )
                )
        return trajectories
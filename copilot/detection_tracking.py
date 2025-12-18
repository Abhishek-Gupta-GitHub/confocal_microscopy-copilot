import pandas as pd
import numpy as np
import trackpy as tp

class DetectionTrackingWorker:
    def __init__(self):
        pass

    def _project_to_2d(self, stack_3d):
        # stack_3d: (T, Z, Y, X)
        return stack_3d.max(axis=1)  # (T, Y, X)

    def run(self, stack_3d, plan):
        frames_2d = self._project_to_2d(stack_3d)

        diameter = int(2 * plan.detection_params_initial["max_sigma"] + 1)
        minmass = plan.detection_params_initial["minmass"]

        f_list = []
        for t, frame in enumerate(frames_2d):
            f = tp.locate(frame, diameter=diameter, minmass=minmass)
            f["frame"] = t
            f_list.append(f)

        if not f_list:
            return {
                "trajectories": pd.DataFrame(),
                "quality_metrics": {},
                "used_params": {},
            }

        features = pd.concat(f_list, ignore_index=True)

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


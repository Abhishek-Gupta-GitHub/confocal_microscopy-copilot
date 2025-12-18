from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class Plan:
    pipeline_type: str
    twin_needed: bool
    twin_settings: Dict[str, Any]
    detection_params_initial: Dict[str, Any]
    tracking_params_initial: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class Orchestrator:
    def __init__(self):
        pass

    def make_plan(self, user_query: str, metadata: dict, quick_stats: dict) -> Plan:
        text = user_query.lower()
        if "structure" in text or "g(" in text:
            pipeline_type = "structure"
        else:
            pipeline_type = "diffusion"

        twin_needed = True

        twin_settings = {
            "n_particles": int(quick_stats.get("density_est", 200)),
            "D": float(quick_stats.get("D_est", 0.2)),
            "n_frames": int(metadata.get("n_frames", 100)),
            "dt": float(metadata.get("frame_interval_s", 0.1)),
        }

        detection_params_initial = {
            "min_sigma": 1,
            "max_sigma": 4,
            "threshold": float(quick_stats.get("snr_est", 5.0) * 0.5),
            "minmass": 200.0,
        }
        tracking_params_initial = {
            "search_range": float(quick_stats.get("search_range_um", 1.0)),
            "memory": 2,
        }

        return Plan(
            pipeline_type=pipeline_type,
            twin_needed=twin_needed,
            twin_settings=twin_settings,
            detection_params_initial=detection_params_initial,
            tracking_params_initial=tracking_params_initial,
        )


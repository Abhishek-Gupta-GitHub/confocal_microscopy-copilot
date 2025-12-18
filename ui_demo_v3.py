# ui_demo.py

"""
Confocal Microscopy Copilot – unified demo UI.

Features:
- Digital-twin example and pre-registered datasets.
- Own dataset upload (TIFF / OME-TIFF).
- Planner → detection/tracking → physics analysis → explainer.
- MSD and depth plots + trajectories CSV download.
- Clean Gradio layout with tabs.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

from copilot.config import DEFAULT_META
from copilot.io_utils import (
    load_stack,
    load_metadata,
    save_stack,
    save_metadata,
    save_json,
    save_trajectories_csv,
)
from copilot.digital_twin import DigitalTwin
from copilot.orchestrator import Orchestrator
from copilot.detection_tracking import DetectionTrackingWorker
from copilot.physics_analysis import PhysicsAnalyst
from copilot.chat_explainer import ChatExplainer

# ---------------- Paths and singletons ---------------- #

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

EXAMPLE_STACK_PATH = DATA_DIR / "example_stack.tif"
EXAMPLE_META_PATH = DATA_DIR / "example_stack_meta.json"
ANALYSIS_SUMMARY_PATH = DATA_DIR / "analysis_summary.json"
TRAJ_CSV_PATH = RESULTS_DIR / "trajectories_latest.csv"

_twin = DigitalTwin()
_orchestrator = Orchestrator()
_detector = DetectionTrackingWorker()
_physics = PhysicsAnalyst()
_explainer = ChatExplainer(llm_client=None)  # or your LLM client if configured


# ---------------- Dataset helpers ---------------- #

_twin = DigitalTwin()  # keep this singleton near top if not already

def ensure_example_dataset(dataset_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Return (stack_3d, metadata) for a named example.
    If not present, create via digital twin as a fallback.
    """
    if dataset_name == "imagej_confocal":
        stack_path = DATA_DIR / "imagej_confocal" / "stack.tif"
        meta_path = DATA_DIR / "imagej_confocal" / "metadata.json"
    elif dataset_name == "synthetic_gaussian":
        stack_path = DATA_DIR / "synthetic_gaussian" / "stack.tif"
        meta_path = DATA_DIR / "synthetic_gaussian" / "metadata.json"
    elif dataset_name == "deeptrack_example":
        stack_path = DATA_DIR / "deeptrack_example" / "stack.tif"
        meta_path = DATA_DIR / "deeptrack_example" / "metadata.json"
    else:
        stack_path = DATA_DIR / "example_stack.tif"
        meta_path = DATA_DIR / "example_stack_meta.json"

    stack_path.parent.mkdir(parents=True, exist_ok=True)

    # If files already exist, just load them
    if stack_path.exists() and meta_path.exists():
        stack_3d = load_stack(str(stack_path))
        metadata = load_metadata(str(meta_path))
        return stack_3d, metadata

    # Otherwise simulate via DigitalTwin
    settings = getattr(config, "DEFAULT_DIGITAL_TWIN_SETTINGS", None)
    if settings is None:
        # Fallback: build minimal settings from DEFAULT_META
        metadata = DEFAULT_META.copy()
        settings = {
            "n_particles": 200,
            "D": 0.2,
            "n_frames": 100,
            "dt": metadata["frame_interval_s"],
        }
    else:
        metadata = settings.get("metadata", DEFAULT_META.copy())

    sim_result = _twin.simulate(settings)
    stack_3d = sim_result["stack"]
    meta = sim_result.get("metadata", metadata)

    save_stack(str(stack_path), stack_3d)
    save_metadata(str(meta_path), meta)

    return stack_3d, meta



def load_user_stack(uploaded_file) -> Tuple[np.ndarray, Dict[str, Any]]:
    """uploaded_file is a Gradio File object."""
    stack_path = Path(uploaded_file.name)
    stack_3d = load_stack(str(stack_path))
    metadata = DEFAULT_META.copy()
    metadata["n_frames"] = int(stack_3d.shape[0])
    return stack_3d, metadata


def crop_stack(stack_3d: np.ndarray, max_frames: int) -> np.ndarray:
    n_total = stack_3d.shape[0]
    n_use = int(min(max_frames, n_total))
    return stack_3d[:n_use]


# ---------------- Plot helpers ---------------- #

def make_msd_plot(summary: Dict[str, Any]):
    msd = summary.get("msd")
    if not isinstance(msd, dict):
        return None

    taus = np.array(msd.get("taus_s", msd.get("taus", [])))
    vals = np.array(msd.get("values", []))
    if len(taus) == 0 or len(vals) == 0:
        return None

    fig, ax = plt.subplots()
    ax.loglog(taus, vals, "o-")
    ax.set_xlabel("lag time τ [s]")
    ax.set_ylabel("MSD")
    ax.set_title("Mean-squared displacement")
    fig.tight_layout()
    return fig


def make_depth_plot(summary: Dict[str, Any]):
    depth = summary.get("diagnostics", {}).get("depth_profile_mean_intensity")
    if depth is None:
        return None
    depth = np.array(depth)
    if depth.size == 0:
        return None

    z = np.arange(len(depth))
    fig, ax = plt.subplots()
    ax.plot(z, depth, "-o")
    ax.set_xlabel("z index")
    ax.set_ylabel("mean intensity")
    ax.set_title("Depth-dependent intensity")
    fig.tight_layout()
    return fig


# ---------------- Core pipeline ---------------- #

def run_pipeline(dataset_choice, uploaded_file, max_frames, user_prompt):
    reasoning_log = []

    # 1) Load stack + metadata
    if dataset_choice != "own_dataset":
        stack_3d, metadata = ensure_example_dataset(dataset_choice)
        dataset_used = dataset_choice
        reasoning_log.append(f"Orchestrator: using built-in dataset '{dataset_choice}'.")
    else:
        if uploaded_file is None:
            return (
                "⚠️ Please upload a TIFF stack for 'own_dataset'.",
                None,
                None,
            )
        stack_3d, metadata = load_user_stack(uploaded_file)
        dataset_used = f"user file: {Path(uploaded_file.name).name}"
        reasoning_log.append("Orchestrator: using user-uploaded dataset.")

    # 2) Crop frames
    n_total = stack_3d.shape[0]
    n_use = int(min(max_frames, n_total))
    stack_3d = stack_3d[:n_use]
    metadata["n_frames"] = n_use
    reasoning_log.append(f"Orchestrator: cropped to {n_use} frames.")

    # 3) Quick stats
    mean_signal = float(stack_3d.mean())
    std_signal = float(stack_3d.std() + 1e-6)
    snr_est = mean_signal / std_signal
    quick_stats = {
        "snr_est": snr_est,
        "density_est": 200,
        "D_est": 0.2,
        "search_range_um": 1.5,
    }

    # 4) Planning
    orch = Orchestrator()
    plan = orch.make_plan(user_prompt, metadata, quick_stats)
    # Relax detection/tracking slightly
    if hasattr(plan, "detection_params_initial"):
        plan.detection_params_initial["minmass"] = 20.0
        plan.detection_params_initial["max_sigma"] = 3
    if hasattr(plan, "tracking_params_initial"):
        plan.tracking_params_initial["search_range"] = 3.0
    reasoning_log.append(
        f"Agent 1 (Planner): pipeline='{getattr(plan, 'pipeline_type', 'NA')}'."
    )

    # 5) Detection & tracking
    dt_result = _detector.run(stack_3d, plan)
    # Normalize result to trajectories DataFrame + quality dict
    if isinstance(dt_result, dict):
        traj = dt_result.get("trajectories")
        quality_metrics = dt_result.get("quality_metrics", {})
    else:
        traj, quality_metrics = dt_result, {}

    if isinstance(traj, dict):
        trajectories = pd.DataFrame(traj)
    else:
        trajectories = traj

    if trajectories is None or len(trajectories) == 0:
        warning = (
            f"⚠️ No trajectories detected for {dataset_used} "
            f"(used {n_use} frames, SNR≈{snr_est:.2f}).\n"
            f"Try increasing SNR or using a different dataset."
        )
        return warning, {"reasoning": "\n".join(reasoning_log)}, None

    n_tracks = quality_metrics.get("n_tracks", len(trajectories))
    reasoning_log.append(
        f"Agent 2 (Detection/Tracking): found {n_tracks} tracks across {n_use} frames."
    )

    # 6) Physics analysis
    analyst = _physics
    summary = analyst.summarize(trajectories, stack_3d, metadata)
    save_json(DATA_DIR / "analysis_summary.json", summary)
    reasoning_log.append(
        "Agent 3 (Physics): computed MSD, alpha, D, depth profile, bleaching, crowding."
    )

    # Persist trajectories CSV for download
    save_trajectories_csv(TRAJ_CSV_PATH, trajectories)

    # 7) Explanation
    explanation = _explainer.explain(user_prompt, summary)
    reasoning_log.append("Agent 4 (Explainer): generated textual summary for the prompt.")

    text_info = (
        f"**Dataset:** {dataset_used}\n"
        f"**Frames used:** {n_use} / {n_total}\n"
        f"**Tracks found:** {n_tracks}\n"
        f"**SNR estimate:** {snr_est:.2f}\n\n"
        f"{explanation}"
    )

    # 8) MSD & depth plots (optional)
    msd_fig = make_msd_plot(summary)
    depth_fig = make_depth_plot(summary)

    # For this layout, we will render MSD as the “Particle Preview” image.
    # If no figure, return None.
    preview_fig = msd_fig if msd_fig is not None else None

    # Build JSON-like output for the Analysis Summary tab
    analysis_dict = {
        "metadata": summary.get("metadata", metadata),
        "quick_stats": quick_stats,
        "summary": summary,
        "reasoning": "\n".join(reasoning_log),
        "trajectories_csv": str(TRAJ_CSV_PATH) if TRAJ_CSV_PATH.exists() else None,
    }

    return text_info, analysis_dict, preview_fig


# ---------------- Gradio UI ---------------- #

custom_css = """
* { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.gradio-container { max-width: 100% !important; width: 100% !important; margin: 0 auto; padding: 1rem; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
<h1 style="text-align:center; margin-bottom: 0.5rem;">
    Confocal Microscopy Copilot
</h1>
<p style="text-align:center; font-size:0.9rem; color:#555;">
    Digital-twin–assisted analysis of confocal particle-tracking data.
</p>
        """
    )

    # ---------------- Top input 2x2 ---------------- #
    with gr.Row():
        with gr.Column(scale=1):
            dataset_choice = gr.Dropdown(
                choices=[
                    "example",
                    "imagej_confocal",
                    "synthetic_gaussian",
                    "deeptrack_example",
                    "own_dataset",
                ],
                value="example",
                label="Dataset",
            )
        with gr.Column(scale=1):
            max_frames_slider = gr.Slider(
                10,
                200,
                value=60,
                step=10,
                label="Max frames to use",
            )

    with gr.Row():
        with gr.Column(scale=1):
            uploaded_file = gr.File(
                label="Upload TIFF stack (for 'own_dataset')",
                file_types=[".tif", ".tiff"],
            )
        with gr.Column(scale=1):
            user_prompt = gr.Textbox(
                label="Analysis prompt",
                lines=5,
                placeholder="e.g. Analyze diffusion and comment on depth and bleaching.",
                value="Analyze diffusion and comment on depth and bleaching.",
            )
            run_button = gr.Button("▶️ Run analysis", variant="primary")

    # ---------------- Outputs below ---------------- #
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("Explanation"):
                output_text = gr.Markdown(label="Copilot explanation")
            with gr.Tab("Analysis Summary"):
                output_json = gr.JSON(label="Analysis summary")
            with gr.Tab("Particle Preview"):
                particle_preview_image = gr.Plot(label="MSD / Preview")

    # ---------------- Callbacks ---------------- #
    run_button.click(
        fn=run_pipeline,
        inputs=[dataset_choice, uploaded_file, max_frames_slider, user_prompt],
        outputs=[output_text, output_json, particle_preview_image],
    )

if __name__ == "__main__":
    demo.launch()

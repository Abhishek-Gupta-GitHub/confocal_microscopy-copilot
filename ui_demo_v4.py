# ui_demo_v4.py

from pathlib import Path
from typing import Dict, Any, Tuple
from typing import Any
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

from copilot import config
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

# Optional advanced physics module
try:
    from src.analysis.physics_analyst_advanced import PhysicsAnalystAdvanced
    HAS_ADVANCED = True
except ImportError:
    PhysicsAnalystAdvanced = None
    HAS_ADVANCED = False

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

ANALYSIS_SUMMARY_PATH = DATA_DIR / "analysis_summary.json"
TRAJ_CSV_PATH = RESULTS_DIR / "trajectories_latest.csv"

_twin = DigitalTwin()
_orch = Orchestrator()
_detector = DetectionTrackingWorker()
_physics_basic = PhysicsAnalyst()
_physics_advanced = PhysicsAnalystAdvanced() if HAS_ADVANCED else None
_explainer = ChatExplainer(llm_client=None)


# ---------- Digital twin presets by domain ---------- #

def _domain_twin_settings(domain: str) -> Dict[str, Any]:
    """
    Start from config.DEFAULT_DIGITAL_TWIN_SETTINGS and tweak per domain.
    """
    base = getattr(config, "DEFAULT_DIGITAL_TWIN_SETTINGS", {}).copy()
    if not base:
        # Fallback if setting is missing
        base = {
            "n_particles": 200,
            "D": 0.2,
            "n_frames": 100,
            "dt": DEFAULT_META["frame_interval_s"],
        }

    if domain == "Soft matter":
        base.update({"n_particles": 400, "D": 0.15})
    elif domain == "Biology":
        base.update({"n_particles": 250, "D": 0.05})
    elif domain == "Materials science":
        base.update({"n_particles": 120, "D": 0.01})
    return base



def _json_safe(obj: Any):
    # Minimal converter: handles numpy types and arrays
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj


# ---------- Dataset helpers ---------- #

def ensure_example_dataset(dataset_name: str, domain: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Return (stack_3d, metadata) for a named example.
    If not present, create via digital twin with a domain-specific preset.
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

    if stack_path.exists() and meta_path.exists():
        stack_3d = load_stack(str(stack_path))
        metadata = load_metadata(str(meta_path))
        return stack_3d, metadata

    # Simulate via digital twin with domain-specific tuning
    metadata = DEFAULT_META.copy()
    settings = _domain_twin_settings(domain)
    sim_result = _twin.simulate(settings)
    stack_3d = sim_result["stack"]
    meta = sim_result.get("metadata", metadata)

    save_stack(str(stack_path), stack_3d)
    save_metadata(str(meta_path), meta)
    return stack_3d, meta


def load_user_stack(uploaded_file):
    stack_path = Path(uploaded_file.name)
    stack_3d = load_stack(str(stack_path))
    metadata = DEFAULT_META.copy()
    metadata["n_frames"] = int(stack_3d.shape[0])
    return stack_3d, metadata


# ---------- Plot helpers ---------- #

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


def _choose_physics_engine(use_advanced: bool):
    """
    Return (engine, is_advanced_used).
    If advanced engine has no summarize(), fall back to basic.
    """
    if use_advanced and _physics_advanced is not None and hasattr(_physics_advanced, "summarize"):
        return _physics_advanced, True
    return _physics_basic, False


import warnings
warnings.filterwarnings(
    "ignore",
    message="No maxima survived mass- and size-based filtering",
)


# ---------- Main pipeline ---------- #

def run_pipeline(dataset_choice, domain_choice, uploaded_file, max_frames, use_advanced, user_prompt):
    reasoning_log = []

    # 1) Load stack + metadata
    if dataset_choice != "own_dataset":
        stack_3d, metadata = ensure_example_dataset(dataset_choice, domain_choice)
        dataset_used = f"{dataset_choice} ({domain_choice})"
        reasoning_log.append(f"Using built-in dataset '{dataset_choice}' with domain '{domain_choice}'.")
    else:
        if uploaded_file is None:
            return (
                "⚠️ Please upload a TIFF stack for 'own_dataset'.",
                None,
                None,
                None,
            )
        stack_3d, metadata = load_user_stack(uploaded_file)
        dataset_used = f"user file: {Path(uploaded_file.name).name}"
        reasoning_log.append("Using user-uploaded dataset.")

    # 2) Crop frames
    n_total = stack_3d.shape[0]
    n_use = int(min(max_frames, n_total))
    stack_3d = stack_3d[:n_use]
    metadata["n_frames"] = n_use
    reasoning_log.append(f"Cropped to {n_use} frames (of {n_total}).")

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
    plan = _orch.make_plan(user_prompt, metadata, quick_stats)
    if hasattr(plan, "detection_params_initial"):
        plan.detection_params_initial["minmass"] = 20.0
        plan.detection_params_initial["max_sigma"] = 3
    if hasattr(plan, "tracking_params_initial"):
        plan.tracking_params_initial["search_range"] = 3.0
    reasoning_log.append(f"Planner pipeline='{getattr(plan, 'pipeline_type', 'NA')}'.")

    # 5) Detection & tracking
    dt_result = _detector.run(stack_3d, plan)
    if isinstance(dt_result, dict):
        traj = dt_result.get("trajectories")
        quality = dt_result.get("quality_metrics", {})
    else:
        traj, quality = dt_result, {}

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
        summary = {
            "metadata": metadata,
            "quick_stats": quick_stats,
            "quality": quality,
            "flags_and_anomalies": ["no_trajectories"],
        }
        save_json(ANALYSIS_SUMMARY_PATH, summary)
        return warning, summary, None, None

    n_tracks = quality.get("n_tracks", len(trajectories))
    reasoning_log.append(f"Detection/Tracking found {n_tracks} tracks.")

    # 6) Physics analysis
    physics_engine, advanced_used = _choose_physics_engine(use_advanced)
    summary = physics_engine.summarize(trajectories, stack_3d, metadata)
    summary["metadata"] = metadata
    summary["quick_stats"] = quick_stats
    summary["quality"] = quality
    summary["domain"] = domain_choice
    save_json(ANALYSIS_SUMMARY_PATH, summary)
    reasoning_log.append(
        f"Physics analysis complete (MSD, depth, etc.). Advanced={advanced_used}."
    )

    # 7) Explanation
    explanation = _explainer.explain(user_prompt, summary)
    reasoning_log.append("Generated explanation via ChatExplainer.")

    save_trajectories_csv(TRAJ_CSV_PATH, trajectories)

    text_info = (
        f"**Dataset:** {dataset_used}\n"
        f"**Frames used:** {n_use} / {n_total}\n"
        f"**Tracks found:** {n_tracks}\n"
        f"**SNR estimate:** {snr_est:.2f}\n"
        f"**Domain preset:** {domain_choice}\n\n"
        f"{explanation}"
    )

    msd_fig = make_msd_plot(summary)
    depth_fig = make_depth_plot(summary)

    safe_summary = _json_safe(summary)
    safe_metadata = _json_safe(metadata)
    safe_quick_stats = _json_safe(quick_stats)

    analysis_dict = {
        "metadata": safe_metadata,
        "quick_stats": safe_quick_stats,
        "summary": safe_summary,
        "reasoning": "\n".join(reasoning_log),
        "advanced_physics_used": bool(use_advanced and HAS_ADVANCED),
        "trajectories_csv": str(TRAJ_CSV_PATH) if TRAJ_CSV_PATH.exists() else None,
    }

    return text_info, analysis_dict, msd_fig, depth_fig


def followup_question(user_followup):
    if not ANALYSIS_SUMMARY_PATH.exists():
        return "No previous analysis found. Run analysis first."
    import json
    summary = json.loads(ANALYSIS_SUMMARY_PATH.read_text())
    prompt = user_followup if user_followup.strip() else "Summarise the main findings."
    return _explainer.explain(prompt, summary)


# ---------- Gradio UI ---------- #

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
            domain_choice = gr.Dropdown(
                choices=["Generic", "Soft matter", "Biology", "Materials science"],
                value="Generic",
                label="Synthetic domain preset",
                info="Controls digital twin parameters for example datasets.",
            )

    with gr.Row():
        with gr.Column(scale=1):
            upload_info = gr.Markdown(
                "Select **'own_dataset'** above to upload a TIFF stack.\n\n"
                "Built-in options:\n"
                "- `synthetic_gaussian`: simulated soft-matter style data.\n"
                "- `imagej_confocal` / `deeptrack_example`: example stacks if present in `data/`."
            )
            uploaded_file = gr.File(
                label="Upload TIFF stack (only for 'own_dataset')",
                file_types=[".tif", ".tiff"],
                visible=False,
            )

        with gr.Column(scale=1):
            user_prompt = gr.Textbox(
                label="Analysis prompt",
                lines=5,
                placeholder="e.g. Analyze diffusion and comment on depth and bleaching.",
                value="Analyze diffusion and comment on depth and bleaching.",
            )
            use_advanced = gr.Checkbox(
                value=HAS_ADVANCED,
                label="Use advanced physics analysis (if available)",
            )
            max_frames_slider = gr.Slider(
                minimum=10,
                maximum=200,
                value=60,
                step=10,
                label="Max frames to use",
            )
            run_button = gr.Button("▶️ Run analysis", variant="primary")

    def _toggle_upload(choice):
        if choice == "own_dataset":
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)

    dataset_choice.change(
        _toggle_upload,
        inputs=[dataset_choice],
        outputs=[upload_info, uploaded_file],
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("Explanation"):
                output_text = gr.Markdown(label="Copilot explanation")
            with gr.Tab("Analysis Summary"):
                output_json = gr.JSON(label="Analysis summary")
            with gr.Tab("Follow-up"):
                followup_box = gr.Textbox(
                    label="Ask a follow-up question",
                    lines=3,
                    value="How reliable is the MSD at long times?",
                )
                followup_button = gr.Button("Ask follow-up")
                followup_answer = gr.Markdown(label="Follow-up answer")

        with gr.Column(scale=1):
            with gr.Tab("Plots"):
                msd_plot = gr.Plot(label="MSD")
                depth_plot = gr.Plot(label="Depth profile")

    run_button.click(
        fn=run_pipeline,
        inputs=[dataset_choice, domain_choice, uploaded_file, max_frames_slider, use_advanced, user_prompt],
        outputs=[output_text, output_json, msd_plot, depth_plot],
    )

    followup_button.click(
        fn=followup_question,
        inputs=[followup_box],
        outputs=[followup_answer],
    )

if __name__ == "__main__":
    demo.launch()

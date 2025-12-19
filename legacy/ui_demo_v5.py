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
    save_json,
    save_trajectories_csv,
)
from copilot.digital_twin import DigitalTwin
from copilot.orchestrator import Orchestrator
from copilot.detection_tracking import DetectionTrackingWorker
from copilot.physics_analysis import PhysicsAnalyst
from copilot.chat_explainer import ChatExplainer

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

# --------------------------------------------------------------------
# Sample datasets
# --------------------------------------------------------------------

SAMPLE_DATASETS = {
    "soft_matter_gel": {
        "folder": "soft_matter_gel",
        "label": "Soft-matter gel: Brownian tracer particles in a crowded viscoelastic matrix (short 4D stack).",
    },
    "cell_nucleus_spots": {
        "folder": "cell_nucleus_spots",
        "label": "Cell nucleus: Confocal z-stack with a few quasi-static fluorescent foci and realistic noise.",
    },
    "colloidal_monolayer": {
        "folder": "colloidal_monolayer",
        "label": "Colloidal monolayer: 2D Brownian motion of ~20 particles in a single focal plane.",
    },
    "membrane_proteins": {
        "folder": "membrane_proteins",
        "label": "Membrane proteins: Fast diffusing and blinking fluorescent spots in a single confocal slice.",
    },
    "material_microstructure": {
        "folder": "material_microstructure",
        "label": "Material microstructure: Static 3D grain-like intensity pattern (single time point, 3D only).",
    },
}


def ensure_sample_dataset(dataset_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    info = SAMPLE_DATASETS[dataset_name]
    folder = DATA_DIR / info["folder"]
    stack_path = folder / "stack.tif"
    meta_path = folder / "metadata.json"

    if not stack_path.exists():
        raise FileNotFoundError(
            f"Expected sample dataset TIFF at {stack_path}. "
            "Please generate the small stacks into data/ first."
        )

    stack_3d = load_stack(str(stack_path))
    if meta_path.exists():
        metadata = load_metadata(str(meta_path))
    else:
        metadata = DEFAULT_META.copy()
        metadata["n_frames"] = int(stack_3d.shape[0])
    return stack_3d, metadata


def load_user_stack(uploaded_file):
    stack_path = Path(uploaded_file.name)
    stack_3d = load_stack(str(stack_path))
    metadata = DEFAULT_META.copy()
    metadata["n_frames"] = int(stack_3d.shape[0])
    return stack_3d, metadata


def _json_safe(obj: Any):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj


def _choose_physics_engine(use_advanced: bool):
    if use_advanced and _physics_advanced is not None and hasattr(
        _physics_advanced, "summarize"
    ):
        return _physics_advanced, True
    return _physics_basic, False


# --------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------

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


def make_rdf_plot(summary: Dict[str, Any]):
    rdf = summary.get("structure", {}).get("rdf")
    if rdf is None or not isinstance(rdf, dict):
        return None
    r = np.array(rdf.get("r", []))
    g = np.array(rdf.get("g_r", []))
    if r.size == 0 or g.size == 0:
        return None
    fig, ax = plt.subplots()
    ax.plot(r, g, "-")
    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial distribution function")
    fig.tight_layout()
    return fig


def make_raw_frame_montage(stack_3d, n_frames=3):
    T = stack_3d.shape[0]
    idx = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(3 * len(idx), 3))
    if len(idx) == 1:
        axes = [axes]
    for ax, t in zip(axes, idx):
        # Show max-projection over z if 4D, else 2D
        if stack_3d.ndim == 4:
            img = stack_3d[t].max(axis=0)
        else:
            img = stack_3d[t]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Frame {t}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def make_annotated_montage(stack_3d, trajectories, n_frames=3):
    if trajectories is None or len(trajectories) == 0:
        return None
    T = stack_3d.shape[0]
    idx = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(3 * len(idx), 3))
    if len(idx) == 1:
        axes = [axes]
    for ax, t in zip(axes, idx):
        if stack_3d.ndim == 4:
            img = stack_3d[t].max(axis=0)
        else:
            img = stack_3d[t]
        ax.imshow(img, cmap="gray")
        # overlay particles at frame t (assumes columns 'frame','y','x')
        df_t = trajectories[trajectories["frame"] == t]
        if not df_t.empty and {"y", "x"}.issubset(df_t.columns):
            ax.scatter(df_t["x"], df_t["y"], s=10, edgecolors="r", facecolors="none")
        ax.set_title(f"Frame {t}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def make_trajectory_plot(trajectories):
    if trajectories is None or len(trajectories) == 0:
        return None
    if not {"y", "x", "particle"}.issubset(trajectories.columns):
        return None
    fig, ax = plt.subplots()
    for pid, df_p in trajectories.groupby("particle"):
        ax.plot(df_p["x"], df_p["y"], "-", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectories (2D projection)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


import warnings

warnings.filterwarnings(
    "ignore", message="No maxima survived mass- and size-based filtering"
)

# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------

def run_pipeline(
    dataset_choice,
    analysis_preset,
    custom_prompt,
    uploaded_file,
    max_frames,
    use_advanced,
    do_msd,
    do_depth,
    do_structure,
    do_traj_plot,
):
    reasoning_log = []

    # Load data
    if dataset_choice != "own_dataset":
        stack_3d, metadata = ensure_sample_dataset(dataset_choice)
        dataset_used = dataset_choice
        reasoning_log.append(f"Using built-in sample dataset '{dataset_choice}'.")
    else:
        if uploaded_file is None:
            return (
                "⚠️ Please upload a TIFF stack for 'own_dataset'.",
                None,
                None,
                None,
                None,
                None,
            )
        stack_3d, metadata = load_user_stack(uploaded_file)
        dataset_used = f"user file: {Path(uploaded_file.name).name}"
        reasoning_log.append("Using user-uploaded dataset.")

    # Crop
    n_total = stack_3d.shape[0]
    n_use = int(min(max_frames, n_total))
    stack_3d = stack_3d[:n_use]
    metadata["n_frames"] = n_use
    reasoning_log.append(f"Cropped to {n_use} frames (of {n_total}).")

    # Quick stats
    mean_signal = float(stack_3d.mean())
    std_signal = float(stack_3d.std() + 1e-6)
    snr_est = mean_signal / std_signal
    quick_stats = {
        "snr_est": snr_est,
        "density_est": 200,
        "D_est": 0.2,
        "search_range_um": 1.5,
    }

    # Build prompt from preset
    if analysis_preset == "Custom":
        prompt = custom_prompt or "Describe the main physics in this dataset."
    else:
        preset_prompts = {
            "Diffusion (MSD focus)": "Quantify diffusion via MSD and comment on anomalous vs Brownian behavior.",
            "Depth / bleaching": "Analyze depth dependence of intensity and possible bleaching.",
            "Structure / RDF": "Analyze spatial structure using RDF and comment on short-range order.",
            "Trajectories overview": "Summarise trajectories, confinement and qualitative motion types.",
            "Full analysis": "Perform a full analysis: diffusion, depth, structural metrics and trajectories.",
        }
        prompt = preset_prompts.get(
            analysis_preset,
            "Describe the main physics in this dataset.",
        )
    reasoning_log.append(f"Analysis preset: {analysis_preset}")

    # Planner
    plan = _orch.make_plan(prompt, metadata, quick_stats)
    if hasattr(plan, "detection_params_initial"):
        plan.detection_params_initial["minmass"] = 20.0
        plan.detection_params_initial["max_sigma"] = 3
    if hasattr(plan, "tracking_params_initial"):
        plan.tracking_params_initial["search_range"] = 3.0
    reasoning_log.append(f"Planner pipeline='{getattr(plan, 'pipeline_type', 'NA')}'.")

    # Detection & tracking
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
        # Only raw frames make sense here
        raw_fig = make_raw_frame_montage(stack_3d)
        return warning, summary, raw_fig, None, None, None

    n_tracks = quality.get("n_tracks", len(trajectories))
    reasoning_log.append(f"Detection/Tracking found {n_tracks} tracks.")

    # Physics analysis
    physics_engine, advanced_used = _choose_physics_engine(use_advanced)
    summary = physics_engine.summarize(trajectories, stack_3d, metadata)

    # Optionally prune heavy parts:
    if not do_msd:
        summary.pop("msd", None)
    if not do_depth:
        summary.get("diagnostics", {}).pop("depth_profile_mean_intensity", None)
    if not do_structure:
        summary.pop("structure", None)

    summary["metadata"] = metadata
    summary["quick_stats"] = quick_stats
    summary["quality"] = quality

    save_json(ANALYSIS_SUMMARY_PATH, summary)
    reasoning_log.append(
        f"Physics analysis complete. Advanced={advanced_used}."
    )

    # Explanation
    explanation = _explainer.explain(prompt, summary)
    reasoning_log.append("Generated explanation via ChatExplainer.")

    save_trajectories_csv(TRAJ_CSV_PATH, trajectories)

    text_info = (
        f"**Dataset:** {dataset_used}\n"
        f"**Frames used:** {n_use} / {n_total}\n"
        f"**Tracks found:** {n_tracks}\n"
        f"**SNR estimate:** {snr_est:.2f}\n"
        f"**Preset:** {analysis_preset}\n\n"
        f"{explanation}"
    )

    # Decide which plots to show
    msd_fig = make_msd_plot(summary) if do_msd else None
    depth_fig = make_depth_plot(summary) if do_depth else None
    rdf_fig = make_rdf_plot(summary) if do_structure else None

    raw_fig = make_raw_frame_montage(stack_3d)
    annotated_fig = make_annotated_montage(stack_3d, trajectories)
    traj_fig = make_trajectory_plot(trajectories) if do_traj_plot else None

    safe_summary = _json_safe(summary)
    analysis_dict = {
        "summary": safe_summary,
        "reasoning": "\n".join(reasoning_log),
        "advanced_physics_used": bool(use_advanced and HAS_ADVANCED),
        "trajectories_csv": str(TRAJ_CSV_PATH) if TRAJ_CSV_PATH.exists() else None,
    }

    return (
        text_info,      # markdown
        analysis_dict,  # json
        raw_fig,        # plot
        annotated_fig,  # plot
        traj_fig,       # plot
        msd_fig,        # plot
        depth_fig,      # plot
        rdf_fig,        # plot
    )


def followup_question(user_followup):
    if not ANALYSIS_SUMMARY_PATH.exists():
        return "No previous analysis found. Run analysis first."
    summary = json.loads(ANALYSIS_SUMMARY_PATH.read_text())
    prompt = user_followup.strip() or "Summarise the main findings."
    return _explainer.explain(prompt, summary)


# --------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------

custom_css = """
* {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 auto;
  padding: 1rem;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
**Confocal Microscopy Copilot**

Small synthetic datasets from soft matter, biology and materials science, with digital-twin–assisted analysis.
"""
    )

    with gr.Row():
        # LEFT: inputs
        with gr.Column(scale=1):
            gr.Markdown("### Input settings")

            dataset_choice = gr.Dropdown(
                choices=[
                    "soft_matter_gel",
                    "cell_nucleus_spots",
                    "colloidal_monolayer",
                    "membrane_proteins",
                    "material_microstructure",
                    "own_dataset",
                ],
                value="soft_matter_gel",
                label="Dataset",
            )

            dataset_info = gr.Markdown(
                value=SAMPLE_DATASETS["soft_matter_gel"]["label"],
                label="Dataset information",
            )

            uploaded_file = gr.File(
                label="Upload TIFF stack (only for 'own_dataset')",
                file_types=[".tif", ".tiff"],
                visible=False,
            )

            gr.Markdown("### Analysis prompt")

            analysis_preset = gr.Dropdown(
                choices=[
                    "Diffusion (MSD focus)",
                    "Depth / bleaching",
                    "Structure / RDF",
                    "Trajectories overview",
                    "Full analysis",
                    "Custom",
                ],
                value="Diffusion (MSD focus)",
                label="Analysis preset",
            )

            custom_prompt = gr.Textbox(
                label="Custom prompt (only used when preset = 'Custom')",
                lines=4,
                value="Describe the main physics in this dataset.",
                interactive=True,
            )

            gr.Markdown("### Speed / analysis options")

            do_msd = gr.Checkbox(
                value=True,
                label="Compute MSD / diffusion metrics",
            )
            do_depth = gr.Checkbox(
                value=True,
                label="Compute depth / bleaching diagnostics",
            )
            do_structure = gr.Checkbox(
                value=False,
                label="Compute structural metrics (RDF etc.)",
            )
            do_traj_plot = gr.Checkbox(
                value=True,
                label="Plot trajectories",
            )

            max_frames_slider = gr.Slider(
                minimum=5,
                maximum=80,
                value=30,
                step=5,
                label="Max frames to use",
            )

            use_advanced = gr.Checkbox(
                value=HAS_ADVANCED,
                label="Use advanced physics analysis (if available)",
            )

            run_button = gr.Button("▶️ Run analysis", variant="primary")

        # RIGHT: outputs
        with gr.Column(scale=1):
            with gr.Tab("Explanation & JSON"):
                output_text = gr.Markdown(label="Explanation")
                output_json = gr.JSON(label="Analysis summary")

            with gr.Tab("Images"):
                raw_plot = gr.Plot(label="Raw frames (few examples)")
                annotated_plot = gr.Plot(label="Annotated frames")
                traj_plot = gr.Plot(label="Trajectories")

            with gr.Tab("Physics plots"):
                msd_plot = gr.Plot(label="MSD")
                depth_plot = gr.Plot(label="Depth profile")
                rdf_plot = gr.Plot(label="RDF / structure")

            with gr.Tab("Follow-up"):
                followup_box = gr.Textbox(
                    label="Ask a follow-up question",
                    lines=3,
                    value="How reliable is the MSD at long times?",
                )
                followup_button = gr.Button("Ask follow-up")
                followup_answer = gr.Markdown(label="Follow-up answer")

    # Interactions

    def _on_dataset_change(choice):
        if choice == "own_dataset":
            info = "Upload your own TIFF stack for analysis."
            return gr.update(value=info), gr.update(visible=True)
        else:
            label = SAMPLE_DATASETS[choice]["label"]
            return gr.update(value=label), gr.update(visible=False)

    dataset_choice.change(
        _on_dataset_change,
        inputs=[dataset_choice],
        outputs=[dataset_info, uploaded_file],
    )

    def _on_preset_change(preset):
        if preset == "Custom":
            return gr.update(interactive=True)
        else:
            return gr.update(interactive=False)

    analysis_preset.change(
        _on_preset_change,
        inputs=[analysis_preset],
        outputs=[custom_prompt],
    )

    run_button.click(
    fn=run_pipeline,
    inputs=[
        dataset_choice,
        analysis_preset,
        custom_prompt,
        uploaded_file,
        max_frames_slider,
        use_advanced,
        do_msd,
        do_depth,
        do_structure,
        do_traj_plot,
    ],
    outputs=[
        output_text,
        output_json,
        raw_plot,
        annotated_plot,
        traj_plot,
        msd_plot,
        depth_plot,
        rdf_plot,
    ],
)


    followup_button.click(
        fn=followup_question,
        inputs=[followup_box],
        outputs=[followup_answer],
    )

if __name__ == "__main__":
    demo.launch()

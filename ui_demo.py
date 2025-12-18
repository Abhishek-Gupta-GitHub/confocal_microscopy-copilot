"""
ui_demo.py

Hackathon-facing demo UI for the Confocal Microscopy Copilot.

This wraps:
- digital_twin (simulate stacks or load example_stack),
- orchestrator (Planner / Plan),
- detection_tracking (Trackpy-based detection + linking),
- physics_analysis (+ optional advanced module),
- chat_explainer (LLM / template-based explainer),

into a single Gradio app with three main flows:
1) Golden demo (example stack from twin),
2) Real-data analysis (upload or choose dataset),
3) Follow-up questions (reuse last JSON summary, no recompute).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

# Core copilot modules
from copilot import config
from copilot.io_utils import (
    load_stack,
    load_metadata,
    save_metadata,
    save_json,
    save_trajectories_csv,
)
from copilot.digital_twin import DigitalTwin
from copilot.orchestrator import Orchestrator, Plan
from copilot.detection_tracking import DetectionTrackingWorker
from copilot.physics_analysis import PhysicsAnalyst
# If you have the advanced module, import it as well
try:
    from src.analysis.physics_analyst_advanced import PhysicsAnalystAdvanced
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False

from copilot.chat_explainer import ChatExplainer

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXAMPLE_STACK_PATH = DATA_DIR / "example_stack.tif"
EXAMPLE_META_PATH = DATA_DIR / "example_stack_meta.json"
ANALYSIS_SUMMARY_PATH = DATA_DIR / "analysis_summary.json"
TRAJ_CSV_PATH = RESULTS_DIR / "trajectories_latest.csv"

# Global singletons (simple “agents”)
_twin = DigitalTwin()
_orchestrator = Orchestrator()
_detector_tracker = DetectionTrackingWorker()
_physics_basic = PhysicsAnalyst()
_explainer = ChatExplainer()

if HAS_ADVANCED:
    _physics_advanced = PhysicsAnalystAdvanced()
else:
    _physics_advanced = None


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------

def _ensure_example_stack() -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ensure that example_stack.tif and its metadata exist.

    If not, use the DigitalTwin to simulate a small 3D+t stack and save it.
    Returns (stack, metadata).
    """
    if EXAMPLE_STACK_PATH.exists() and EXAMPLE_META_PATH.exists():
        stack = load_stack(EXAMPLE_STACK_PATH)
        meta = load_metadata(EXAMPLE_META_PATH)
        return stack, meta

    # Fallback: simulate a default twin dataset
    settings = config.DEFAULT_DIGITAL_TWIN_SETTINGS
    sim_result = _twin.simulate(settings)
    stack = sim_result["stack"]
    meta = sim_result["metadata"]

    # Persist so the demo is reproducible
    _ = DATA_DIR.mkdir(exist_ok=True, parents=True)
    from copilot.io_utils import save_stack  # local import to avoid circulars
    save_stack(EXAMPLE_STACK_PATH, stack)
    save_metadata(EXAMPLE_META_PATH, meta)
    return stack, meta


def _crop_stack(stack: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Crop a 4D stack (T, Z, Y, X) or 3D stack (Z, Y, X) along time if needed.
    """
    if stack.ndim == 4:
        t = min(stack.shape[0], max_frames)
        return stack[:t]
    return stack


def _choose_dataset(
    dataset_choice: str,
    uploaded_file,
    max_frames: int,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    Resolve the dataset according to the dropdown and uploaded file.

    Returns
    -------
    stack : np.ndarray
    metadata : dict
    dataset_label : str
    """
    # 1) Golden example from twin
    if dataset_choice == "example":
        stack, meta = _ensure_example_stack()
        stack = _crop_stack(stack, max_frames)
        return stack, meta, "example (digital twin)"

    # 2) Pre-registered folders (if present)
    if dataset_choice in {"imagej_confocal", "synthetic_gaussian", "deeptrack_example"}:
        folder = DATA_DIR / dataset_choice
        stack_path = folder / "stack.tif"
        meta_path = folder / "metadata.json"
        if stack_path.exists() and meta_path.exists():
            stack = load_stack(stack_path)
            meta = load_metadata(meta_path)
            stack = _crop_stack(stack, max_frames)
            return stack, meta, dataset_choice

    # 3) User upload wins if provided
    if uploaded_file is not None:
        from copilot.io_utils import load_stack as load_any_tiff
        stack = load_any_tiff(uploaded_file.name)
        # Minimal metadata when none is provided
        meta = {
            "pixel_size_um": config.DEFAULT_PIXEL_SIZE_UM,
            "z_step_um": config.DEFAULT_Z_STEP_UM,
            "frame_interval_s": config.DEFAULT_FRAME_INTERVAL_S,
            "source": "uploaded",
        }
        stack = _crop_stack(stack, max_frames)
        return stack, meta, f"uploaded: {Path(uploaded_file.name).name}"

    # 4) Fallback: example
    stack, meta = _ensure_example_stack()
    stack = _crop_stack(stack, max_frames)
    return stack, meta, "example (fallback)"


def _choose_physics_engine(use_advanced: bool):
    """
    Select basic vs advanced physics analyst.
    """
    if use_advanced and _physics_advanced is not None:
        return _physics_advanced
    return _physics_basic


def _run_full_analysis(
    dataset_choice: str,
    uploaded_file,
    max_frames: int,
    analysis_prompt: str,
    use_advanced: bool,
    use_twin_recommend: bool,
) -> Tuple[str, str, str, str, any, any]:
    """
    Core pipeline: data -> plan -> detection/tracking -> physics -> explanation.

    Returns
    -------
    explanation : str
    agent_log : str
    dataset_info : str
    summary_json_str : str  (serialized JSON)
    msd_plot : matplotlib.Figure or None
    depth_plot : matplotlib.Figure or None
    """
    # 0) Load / simulate stack
    stack, meta, label = _choose_dataset(dataset_choice, uploaded_file, max_frames)
    dataset_info = f"Dataset: {label}\nShape: {tuple(stack.shape)}"

        # 1) Planning (planner agent)
    # Use only the arguments that Orchestrator.make_plan actually supports.
    # If you later extend it, you can expand this call.
    try:
        plan: Plan = _orchestrator.make_plan(analysis_prompt, meta)
    except TypeError:
        # Fallback: if Plan is not used or Orchestrator returns a dict
        plan = _orchestrator.make_plan(analysis_prompt, meta)


    # 2) Detection + tracking
    result = _detector_tracker.run(stack, plan)
    if isinstance(result, tuple):
        traj_df, quality = result
    else:
        traj_df, quality = result, {}


    if traj_df is None or len(traj_df) == 0:
        explanation = (
            "No reliable trajectories were found. "
            "Try fewer frames, lower density, or adjust detection parameters."
        )
        agent_log = "DetectionTrackingWorker: no features or tracks detected."
        summary = {
            "metadata": meta,
            "quick_stats": quick_stats,
            "plan": plan.to_dict(),
            "quality": quality,
            "flags_and_anomalies": ["no_trajectories"],
        }
        # Persist JSON
        save_json(ANALYSIS_SUMMARY_PATH, summary)
        return explanation, agent_log, dataset_info, str(summary), None, None

    # Persist trajectories for download
    save_trajectories_csv(TRAJ_CSV_PATH, traj_df)

    # 3) Physics analysis
    physics_engine = _choose_physics_engine(use_advanced)
    summary = physics_engine.summarize(traj_df, stack, meta, plan)

    # Add metadata to summary for the explainer
    summary["metadata"] = meta
    summary["quick_stats"] = quick_stats
        # Add plan in a JSON-friendly way
    if hasattr(plan, "to_dict"):
        summary["plan"] = plan.to_dict()
    else:
        summary["plan"] = dict(plan) if isinstance(plan, dict) else {"plan_repr": str(plan)}

    summary["quality"] = quality

    # Persist JSON summary
    save_json(ANALYSIS_SUMMARY_PATH, summary)

    # 4) Explanation (LLM or template)
    explanation = _explainer.explain(analysis_prompt, summary)

    # 5) Agent log: concise reasoning trail
    log_lines = []
    log_lines.append(f"Planner → pipeline_type: {getattr(plan, 'pipeline_type', 'NA')}")
    log_lines.append(f"Planner → twin_used: {getattr(plan, 'twin_needed', False)}")
    log_lines.append(f"Planner → detection params: {getattr(plan, 'detection_params', {})}")
    log_lines.append(f"Planner → tracking params: {getattr(plan, 'tracking_params', {})}")

    log_lines.append(f"DetectionTracking → n_tracks: {quality.get('n_tracks', 'NA')}")
    log_lines.append(
        f"Physics → alpha: {summary.get('alpha', 'NA')}, "
        f"D: {summary.get('D', 'NA')}"
    )
    flags = summary.get("flags_and_anomalies", [])
    if flags:
        log_lines.append(f"Physics → flags: {flags}")
    agent_log = "\n".join(log_lines)

    # 6) Plots: MSD + depth profile (if present)
    msd_plot = None
    depth_plot = None

    msd = summary.get("msd", None)
    if msd is not None and isinstance(msd, dict):
        taus = np.array(msd.get("taus", []))
        vals = np.array(msd.get("values", []))
        if len(taus) > 0 and len(vals) > 0:
            fig1, ax1 = plt.subplots()
            ax1.loglog(taus, vals, "o-")
            ax1.set_xlabel("Lag time")
            ax1.set_ylabel("MSD")
            ax1.set_title("MSD (log-log)")
            fig1.tight_layout()
            msd_plot = fig1

    depth = summary.get("depth_profile", None)
    if depth is not None and isinstance(depth, dict):
        z_idx = np.array(depth.get("z_index", []))
        mean_int = np.array(depth.get("mean_intensity", []))
        if len(z_idx) > 0 and len(mean_int) > 0:
            fig2, ax2 = plt.subplots()
            ax2.plot(z_idx, mean_int, "-o")
            ax2.set_xlabel("z index")
            ax2.set_ylabel("Mean intensity")
            ax2.set_title("Depth profile")
            fig2.tight_layout()
            depth_plot = fig2

    # 7) Summary JSON as pretty string (for advanced users)
    import json
    summary_json_str = json.dumps(summary, indent=2)

    return (
        explanation,
        agent_log,
        dataset_info,
        summary_json_str,
        msd_plot,
        depth_plot,
    )


def _follow_up(question: str) -> str:
    """
    Follow-up questions reuse the last saved JSON summary; no recomputation.
    """
    if not ANALYSIS_SUMMARY_PATH.exists():
        return "No previous analysis summary found. Run an analysis first."

    import json
    summary = json.loads(ANALYSIS_SUMMARY_PATH.read_text())
    answer = _explainer.explain(question, summary)
    return answer


# --------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Confocal Microscopy Copilot – Hackathon Demo") as demo:
        gr.Markdown(
            "## Confocal Physics Copilot\n"
            "Upload a stack **or** use the digital twin example, then run:\n"
            "from images → trajectories → MSD → physics explanation.\n"
            "The backend uses a digital twin, multi-agent pipeline, and an LLM-style explainer."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Dataset & Twin")

                dataset_choice = gr.Dropdown(
                    choices=[
                        "example",
                        "imagej_confocal",
                        "synthetic_gaussian",
                        "deeptrack_example",
                        "uploaded",
                    ],
                    value="example",
                    label="Dataset",
                    info="Use 'example' for the golden demo (digital twin). "
                         "Other options require data/ subfolders.",
                )

                uploaded_file = gr.File(
                    label="Upload TIFF / OME-TIFF (optional)",
                    file_types=[".tif", ".tiff"],
                )

                max_frames = gr.Slider(
                    1, 200, value=60, step=1,
                    label="Max frames to use (time cropping)",
                    info="Crop long movies for faster tracking.",
                )

                # use_twin_recommend = gr.Checkbox(
                #     value=True,
                #     label="Use digital twin to suggest parameters",
                # )

                use_advanced = gr.Checkbox(
                    value=HAS_ADVANCED,
                    label="Use advanced physics analysis (if available)",
                )

                gr.Markdown("### 2. Analysis prompt")

                prompt_templates = gr.Dropdown(
                    choices=[
                        "Analyze diffusion and comment on depth and bleaching.",
                        "Quantify subdiffusion and identify a reliable MSD fit window.",
                        "Focus on crowding and near-wall effects.",
                        "Custom",
                    ],
                    value="Analyze diffusion and comment on depth and bleaching.",
                    label="Suggested prompts",
                )

                analysis_prompt = gr.Textbox(
                    label="Analysis question",
                    lines=3,
                    value=(
                        "Compute MSD, estimate alpha and D, and explain whether the "
                        "motion is diffusive, subdiffusive, or confined. Comment on "
                        "depth-dependent intensity and bleaching."
                    ),
                )

                def _update_prompt(template: str):
                    if template == "Custom":
                        return gr.update()
                    return gr.update(value=template)

                prompt_templates.change(
                    _update_prompt,
                    inputs=prompt_templates,
                    outputs=analysis_prompt,
                )

                run_btn = gr.Button("▶ Run full analysis", variant="primary")

                gr.Markdown("### 3. Follow-up questions (no recompute)")
                followup_q = gr.Textbox(
                    label="Ask the copilot about the last run",
                    lines=2,
                    value="How reliable is the MSD at long times?",
                )
                followup_btn = gr.Button("Ask follow-up")

            with gr.Column(scale=2):
                gr.Markdown("### Outputs")

                explanation_out = gr.Textbox(
                    label="Copilot explanation",
                    lines=8,
                )

                agent_log_out = gr.Textbox(
                    label="Agent reasoning (debug view)",
                    lines=8,
                )

                dataset_info_out = gr.Textbox(
                    label="Dataset info",
                    lines=3,
                )

                summary_json_out = gr.Textbox(
                    label="Analysis summary (JSON)",
                    lines=10,
                )

                msd_plot_out = gr.Plot(label="MSD")
                depth_plot_out = gr.Plot(label="Depth profile")

                traj_download = gr.File(
                    label="Download trajectories (CSV)",
                    value=None,
                )

                # Wire main button
                def _on_run(
                    ds_choice,
                    up_file,
                    max_fr,
                    prompt,
                    use_adv,
                    use_twin,
                ):
                    (
                        explanation,
                        agent_log,
                        ds_info,
                        summary_json_str,
                        msd_fig,
                        depth_fig,
                    ) = _run_full_analysis(
                        dataset_choice=ds_choice,
                        uploaded_file=up_file,
                        max_frames=int(max_fr),
                        analysis_prompt=prompt,
                        use_advanced=use_adv,
                        #use_twin_recommend=use_twin,
                    )

                    traj_file = TRAJ_CSV_PATH if TRAJ_CSV_PATH.exists() else None
                    return (
                        explanation,
                        agent_log,
                        ds_info,
                        summary_json_str,
                        msd_fig,
                        depth_fig,
                        traj_file,
                    )

                run_btn.click(
                    _on_run,
                    inputs=[
                        dataset_choice,
                        uploaded_file,
                        max_frames,
                        analysis_prompt,
                        use_advanced,
                        #use_twin_recommend,
                    ],
                    outputs=[
                        explanation_out,
                        agent_log_out,
                        dataset_info_out,
                        summary_json_out,
                        msd_plot_out,
                        depth_plot_out,
                        traj_download,
                    ],
                )

                # Wire follow-up
                followup_btn.click(
                    _follow_up,
                    inputs=[followup_q],
                    outputs=[explanation_out],
                )

        return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()



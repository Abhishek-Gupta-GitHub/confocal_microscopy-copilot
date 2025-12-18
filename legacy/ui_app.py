# ui_app.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from copilot.digital_twin import simulate_confocal_stack
from copilot.orchestrator import Orchestrator
from copilot.detection_tracking import DetectionTrackingWorker
from copilot.physics_analysis import PhysicsAnalyst
from copilot.chat_explainer import ChatExplainer


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------- Dataset helpers ---------- #

def ensure_example_dataset(dataset_name: str):
    """Return (stack_3d, metadata) for a named example.
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

    if stack_path.exists() and meta_path.exists():
        stack_3d = load_stack(str(stack_path))
        metadata = load_metadata(str(meta_path))
    else:
        metadata = DEFAULT_META.copy()
        stack_3d, _ = simulate_confocal_stack(
            metadata,
            twin_settings={
                "n_particles": 200,
                "D": 0.2,
                "n_frames": 100,
                "dt": metadata["frame_interval_s"],
            },
        )
        save_stack(str(stack_path), stack_3d)
        save_metadata(str(meta_path), metadata)

    return stack_3d, metadata


def load_user_stack(uploaded_file):
    """uploaded_file is a gradio File object."""
    stack_path = Path(uploaded_file.name)
    stack_3d = load_stack(str(stack_path))
    metadata = DEFAULT_META.copy()
    metadata["n_frames"] = int(stack_3d.shape[0])
    return stack_3d, metadata


# ---------- Plot helpers ---------- #

def make_msd_plot(summary):
    msd = summary.get("msd")
    if not msd:
        return None
    taus = np.array(msd["taus_s"])
    vals = np.array(msd["values"])
    if len(taus) == 0:
        return None
    fig, ax = plt.subplots()
    ax.loglog(taus, vals, "o-")
    ax.set_xlabel("lag time τ [s]")
    ax.set_ylabel("MSD [arb. units]")
    ax.set_title("Mean-squared displacement")
    fig.tight_layout()
    return fig


def make_depth_plot(summary):
    depth = summary.get("diagnostics", {}).get("depth_profile_mean_intensity")
    if depth is None:
        return None
    z = np.arange(len(depth))
    fig, ax = plt.subplots()
    ax.plot(z, depth, "-o")
    ax.set_xlabel("z index")
    ax.set_ylabel("mean intensity")
    ax.set_title("Depth-dependent intensity")
    fig.tight_layout()
    return fig


# ---------- Main analysis pipeline ---------- #

def run_pipeline(dataset_choice, uploaded_file, max_frames, prompt_choice, user_prompt):
    reasoning_log = []

    # Decide effective prompt
    if prompt_choice != "Custom":
        effective_prompt = prompt_choice
    else:
        effective_prompt = user_prompt if user_prompt.strip() else "Analyze diffusion."

    # 1) Load stack + metadata
    if dataset_choice != "own_dataset":
        stack_3d, metadata = ensure_example_dataset(dataset_choice)
        dataset_used = dataset_choice
        reasoning_log.append(f"Orchestrator: using built-in dataset '{dataset_choice}'.")
    else:
        if uploaded_file is None:
            return (
                "⚠️ Please upload a TIFF stack for 'own_dataset'.",
                None, None, None, None, None,
            )
        stack_3d, metadata = load_user_stack(uploaded_file)
        dataset_used = f"user file: {Path(uploaded_file.name).name}"
        reasoning_log.append("Orchestrator: using user-uploaded dataset.")

    # 2) Crop frames for speed
    n_total = stack_3d.shape[0]
    n_use = int(min(max_frames, n_total))
    stack_3d = stack_3d[:n_use]
    metadata["n_frames"] = n_use
    reasoning_log.append(f"Orchestrator: cropped to {n_use} frames.")

    # 3) Quick stats
    mean_signal = float(stack_3d.mean())
    std_signal = float(stack_3d.std())
    snr_est = mean_signal / (std_signal + 1e-6)
    quick_stats = {
        "snr_est": snr_est,
        "density_est": 200,
        "D_est": 0.2,
        "search_range_um": 1.5,
    }

    # 4) Plan (Agent 1)
    orch = Orchestrator()
    plan = orch.make_plan(effective_prompt, metadata, quick_stats)
    plan.detection_params_initial["minmass"] = 20.0
    plan.detection_params_initial["max_sigma"] = 3
    plan.tracking_params_initial["search_range"] = 3.0
    reasoning_log.append(
        f"Agent 1 (Planner): pipeline='{plan.pipeline_type}', "
        f"search_range={plan.tracking_params_initial['search_range']} px, "
        f"minmass={plan.detection_params_initial['minmass']}."
    )

    # 5) Detection & tracking (Agent 2)
    det_worker = DetectionTrackingWorker()
    dt_result = det_worker.run(stack_3d, plan)
    trajectories = dt_result["trajectories"]
    quality_metrics = dt_result["quality_metrics"]

    if trajectories is None or len(trajectories) == 0:
        return (
            f"⚠️ No trajectories detected for {dataset_used} "
            f"(used {n_use} frames, SNR≈{snr_est:.2f}).\n"
            f"Try increasing SNR or using a different dataset.",
            None, None, None, None, "\n".join(reasoning_log),
        )

    n_tracks = quality_metrics["n_tracks"]
    reasoning_log.append(
        f"Agent 2 (Detection/Tracking): found {n_tracks} tracks "
        f"across {n_use} frames."
    )

    # 6) Physics analysis (Agent 3)
    analyst = PhysicsAnalyst()
    summary = analyst.summarize(trajectories, stack_3d, metadata)
    save_json(DATA_DIR / "analysis_summary.json", summary)
    reasoning_log.append("Agent 3 (Physics): computed MSD, alpha, D, depth profile, bleaching, crowding.")

    # 7) Explanation (Agent 4)
    explainer = ChatExplainer(llm_client=None)
    explanation = explainer.explain(effective_prompt, summary)
    reasoning_log.append("Agent 4 (Explainer): generated textual summary for the prompt.")

    text_info = (
        f"**Dataset:** {dataset_used}\n"
        f"**Frames used:** {n_use} / {n_total}\n"
        f"**Tracks found:** {n_tracks}\n"
        f"**SNR estimate:** {snr_est:.2f}\n\n"
        f"{explanation}"
    )

    # 8) Plots
    msd_fig = make_msd_plot(summary)
    depth_fig = make_depth_plot(summary)

    # 9) Save trajectories CSV for download
    csv_path = RESULTS_DIR / "trajectories_latest.csv"
    save_trajectories_csv(csv_path, trajectories)

    return (
        text_info,              # explanation markdown
        summary,                # JSON summary (also cached in state)
        msd_fig,                # MSD plot
        depth_fig,              # depth plot
        str(csv_path),          # CSV file path
        "\n".join(reasoning_log),
    )


def followup_question(user_followup, cached_summary):
    """Answer further questions using cached summary only (no recompute)."""
    if cached_summary is None:
        return "No previous analysis found in this session. Run analysis first."
    prompt = user_followup if user_followup.strip() else "Summarise the main findings."
    explainer = ChatExplainer(llm_client=None)
    return explainer.explain(prompt, cached_summary)


# ---------- Gradio UI ---------- #

custom_css = """
* { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.gradio-container { max-width: 1000px !important; }
"""

suggested_prompts = [
    "Analyze diffusion and comment on depth and bleaching.",
    "Quantify subdiffusion and identify a reliable MSD fit window.",
    "Evaluate g(r), coordination number, and crowding-related tracking failures.",
    "Assess depth-dependent bias and suggest a trustworthy z-range.",
]

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style="text-align:center; margin-bottom: 0.3rem;">
          Confocal Microscopy Copilot
        </h1>
        <p style="text-align:center; font-size: 0.9rem; color: #555;">
          Digital twin–assisted confocal tracking with MSD, depth diagnostics, and an explainer agent.
        </p>
        """,
    )

    summary_state = gr.State(value=None)

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
                info="Choose a built-in example or 'own_dataset' to upload.",
            )
            uploaded_file = gr.File(
                label="Upload TIFF stack (for 'own_dataset')",
                file_types=[".tif", ".tiff"],
            )
            max_frames_slider = gr.Slider(
                minimum=10,
                maximum=200,
                value=60,
                step=10,
                label="Max frames to use",
                info="Cropping speeds up tracking and analysis.",
            )

        with gr.Column(scale=1):
            prompt_choice = gr.Dropdown(
                choices=["Custom"] + suggested_prompts,
                value=suggested_prompts[0],
                label="Suggested prompts",
                info="Pick a template or choose 'Custom' and type your own.",
            )
            user_prompt = gr.Textbox(
                label="Custom prompt (used if 'Custom' is selected)",
                lines=4,
                placeholder="Describe your question to the copilot...",
                value="Analyze diffusion and comment on depth and bleaching.",
            )
            run_button = gr.Button("▶ Run analysis", variant="primary")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=2):
            output_text = gr.Markdown(label="Copilot explanation")
            reasoning_box = gr.Markdown(label="Agent reasoning (debug view)")
        with gr.Column(scale=1):
            msd_plot = gr.Plot(label="MSD")
            depth_plot = gr.Plot(label="Depth profile")
            download_button = gr.DownloadButton(
                label="Download trajectories (CSV)",
                value=None,
            )

    gr.Markdown("### Follow-up questions (no recompute)")
    followup_box = gr.Textbox(
        label="Ask a follow-up question about this analysis",
        lines=3,
        placeholder="e.g. How reliable is MSD at long times?",
    )
    followup_button = gr.Button("Ask follow-up")
    followup_answer = gr.Markdown(label="Follow-up answer")

    # Main analysis
    run_button.click(
        fn=run_pipeline,
        inputs=[dataset_choice, uploaded_file, max_frames_slider, prompt_choice, user_prompt],
        outputs=[
            output_text,
            summary_state,   # cache summary in state
            msd_plot,
            depth_plot,
            download_button,
            reasoning_box,
        ],
    )

    # Follow-up questions
    followup_button.click(
        fn=followup_question,
        inputs=[followup_box, summary_state],
        outputs=[followup_answer],
    )

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(RESULTS_DIR)],
        theme=gr.themes.Soft(),
        css=custom_css,
    )


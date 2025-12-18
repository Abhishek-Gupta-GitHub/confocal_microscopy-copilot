# ui_app.py
from pathlib import Path

import gradio as gr
import numpy as np

from copilot.config import DEFAULT_META
from copilot.io_utils import (
    load_stack,
    load_metadata,
    save_stack,
    save_metadata,
    save_json,
)
from copilot.digital_twin import simulate_confocal_stack
from copilot.orchestrator import Orchestrator
from copilot.detection_tracking import DetectionTrackingWorker
from copilot.physics_analysis import PhysicsAnalyst
from copilot.chat_explainer import ChatExplainer


DATA_DIR = Path("data")


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


def run_pipeline(dataset_choice, uploaded_file, max_frames, user_prompt):
    # 1) Load stack + metadata
    if dataset_choice != "own_dataset":
        stack_3d, metadata = ensure_example_dataset(dataset_choice)
        dataset_used = dataset_choice
    else:
        if uploaded_file is None:
            return "⚠️ Please upload a TIFF stack for 'own_dataset'.", None
        stack_3d, metadata = load_user_stack(uploaded_file)
        dataset_used = f"user file: {Path(uploaded_file.name).name}"

    # 2) Crop frames for speed & robustness
    n_total = stack_3d.shape[0]
    n_use = int(min(max_frames, n_total))
    stack_3d = stack_3d[:n_use]
    metadata["n_frames"] = n_use

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
    plan = orch.make_plan(user_prompt, metadata, quick_stats)

    # Relax detection/tracking a bit
    plan.detection_params_initial["minmass"] = 20.0
    plan.detection_params_initial["max_sigma"] = 3
    plan.tracking_params_initial["search_range"] = 3.0

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
            None,
        )

    # 6) Physics analysis (Agent 3)
    analyst = PhysicsAnalyst()
    summary = analyst.summarize(trajectories, stack_3d, metadata)
    save_json(DATA_DIR / "analysis_summary.json", summary)

    # 7) Explanation (Agent 4)
    explainer = ChatExplainer(llm_client=None)
    explanation = explainer.explain(user_prompt, summary)

    n_tracks = quality_metrics["n_tracks"]
    text_info = (
        f"**Dataset:** {dataset_used}\n"
        f"**Frames used:** {n_use} / {n_total}\n"
        f"**Tracks found:** {n_tracks}\n"
        f"**SNR estimate:** {snr_est:.2f}\n\n"
        f"{explanation}"
    )
    return text_info, summary


# ----------------- Gradio UI ----------------- #

custom_css = """
* {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.gradio-container {
  max-width: 900px !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <h1 style="text-align:center; margin-bottom: 0.5rem;">
          Confocal Microscopy Copilot
        </h1>
        <p style="text-align:center; font-size: 0.9rem; color: #555;">
          Digital-twin–assisted analysis of confocal particle-tracking data,
          with MSD, depth/bleaching diagnostics, and LLM-style explanations.
        </p>
        """,
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
                info="Choose a built-in example or 'own_dataset' to upload.",
            )

            uploaded_file = gr.File(
                label="Upload TIFF stack (only for 'own_dataset')",
                file_types=[".tif", ".tiff"],
            )

            max_frames_slider = gr.Slider(
                minimum=10,
                maximum=200,
                value=60,
                step=10,
                label="Max frames to use in analysis",
                info="Cropping to fewer frames speeds up tracking.",
            )

        with gr.Column(scale=1):
            user_prompt = gr.Textbox(
                label="Analysis / question prompt",
                lines=5,
                placeholder="e.g. Analyze diffusion and comment on depth-dependent bias and bleaching.",
                value="Analyze diffusion and comment on depth and bleaching.",
            )
            run_button = gr.Button("▶ Run analysis", variant="primary")

    gr.Markdown("---")

    with gr.Row():
        output_text = gr.Markdown(label="Copilot explanation")
        output_json = gr.JSON(label="Analysis summary (JSON)")

    run_button.click(
        fn=run_pipeline,
        inputs=[dataset_choice, uploaded_file, max_frames_slider, user_prompt],
        outputs=[output_text, output_json],
    )

if __name__ == "__main__":
    demo.launch()

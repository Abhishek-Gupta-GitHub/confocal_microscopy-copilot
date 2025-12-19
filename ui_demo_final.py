"""
Confocal Microscope Copilot – end-to-end assistant for noisy confocal stacks.

- Denoise and track particles (Trackpy backend for now).
- Run physics analysis (basic or advanced if available).
- Use a microscopy-tuned LLM to generate a scientific summary.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple

import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr
from scipy.ndimage import median_filter, gaussian_filter

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
from openai import OpenAI

try:
    from src.analysis.physics_analyst_advanced import PhysicsAnalystAdvanced

    HAS_ADVANCED = True
except ImportError:
    PhysicsAnalystAdvanced = None
    HAS_ADVANCED = False

# ----------------------------------------------------------------------
# Paths and globals
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

ANALYSIS_SUMMARY_PATH = DATA_DIR / "analysis_summary.json"
TRAJ_CSV_PATH = RESULTS_DIR / "trajectories_latest.csv"

LAST_RUN_PARAMS: dict = {}

_twin = DigitalTwin()
_orch = Orchestrator()
_detector = DetectionTrackingWorker()
_physics_basic = PhysicsAnalyst()
_physics_advanced = PhysicsAnalystAdvanced() if HAS_ADVANCED else None

# LLM client is injected or configured inside ChatExplainer as you prefer
# _explainer = ChatExplainer()



warnings.filterwarnings(
    "ignore", message="No maxima survived mass- and size-based filtering"
)

# ----------------------------------------------------------------------
# Sample datasets
# ----------------------------------------------------------------------

SAMPLE_DATASETS = {
    "brownian_3d colloids": {
        "folder": "brownian_3d colloids",
        "label": (
            "Brownian data: 3D particles in confocal-style image frames "
            "(specified frame size and time step)."
        ),
    },
    "soft_matter_gel": {
        "folder": "soft_matter_gel",
        "label": (
            "Soft-matter gel: Brownian tracer particles in a crowded "
            "viscoelastic matrix (short 4D stack)."
        ),
    },
    "cell_nucleus_spots": {
        "folder": "cell_nucleus_spots",
        "label": (
            "Cell nucleus: Confocal z-stack with a few quasi-static "
            "fluorescent foci and realistic noise."
        ),
    },
    "colloidal_monolayer": {
        "folder": "colloidal_monolayer",
        "label": (
            "Colloidal monolayer: 2D Brownian motion of ~20 particles in "
            "a single focal plane."
        ),
    },
    "membrane_proteins": {
        "folder": "membrane_proteins",
        "label": (
            "Membrane proteins: Fast diffusing and blinking fluorescent "
            "spots in a single confocal slice."
        ),
    },
    "material_microstructure": {
        "folder": "material_microstructure",
        "label": (
            "Material microstructure: Static 3D grain-like intensity "
            "pattern (single time point, 3D only)."
        ),
    },
}


def ensure_sample_dataset(dataset_name: str):
    info = SAMPLE_DATASETS[dataset_name]
    folder = DATA_DIR / info["folder"]
    stack_path = folder / "stack.tif"
    meta_path = folder / "metadata.json"

    if not stack_path.exists():
        raise FileNotFoundError(
            f"Expected sample dataset TIFF at {stack_path}. "
            "Run generate_brownian_data.py once (for Brownian data) or "
            "place your example stacks in the data/ folder."
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


# ----------------------------------------------------------------------
# Utility: denoising and JSON helpers
# ----------------------------------------------------------------------

def denoise_stack(stack, method="median", strength=1):
    arr = stack.astype(np.float32)
    if method == "median":
        size = (1, 1, 3, 3) if arr.ndim == 4 else 3
        return median_filter(arr, size=size)
    elif method == "gaussian":
        sigma = 0.5 * strength
        return gaussian_filter(arr, sigma=sigma)
    return arr


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




def _make_explainer(backend: str) -> ChatExplainer:
    """
    Map the UI dropdown name to a concrete client + model.
    Assumes OPENAI_API_KEY etc. are set in the environment.
    """
    backend = backend or "openai_gpt4o"
    
   
    api_key = ""
    client = OpenAI(api_key=api_key)
    model = "gpt-4o"
    
    if backend == "openai_gpt4o":
        client = OpenAI()          # new OpenAI client; reads OPENAI_API_KEY
        model = "gpt-4o"
    elif backend == "anthropic_sonnet":
        # Placeholder – only if/when you wire Anthropic
        from anthropic import Anthropic
        client = Anthropic()
        model = "claude-3-5-sonnet-20240620"
    elif backend == "local_small_model":
        # TODO: implement your own local client exposing .chat.completions.create
        client = OpenAI()
        model = "gpt-4o-mini"
    else:
        client = OpenAI()
        model = "gpt-4o-mini"

    return ChatExplainer(client=client, model=model)

# ----------------------------------------------------------------------
# Manual MSD and RDF (agent options)
# ----------------------------------------------------------------------

def compute_msd_manual(trajectories: pd.DataFrame, max_lag: int = None):
    if trajectories is None or len(trajectories) == 0:
        return {"taus": np.array([]), "values": np.array([])}

    needed_cols = {"frame", "particle", "x", "y"}
    if not needed_cols.issubset(trajectories.columns):
        return {"taus": np.array([]), "values": np.array([])}

    df = trajectories[["frame", "particle", "x", "y"]].copy()
    df = df.sort_values(["particle", "frame"]).reset_index(drop=True)

    if max_lag is None:
        max_lag = int(df["frame"].max() - df["frame"].min())
    if max_lag <= 0:
        return {"taus": np.array([]), "values": np.array([])}

    taus = np.arange(1, max_lag + 1, dtype=int)
    msd_vals = np.full_like(taus, np.nan, dtype=float)

    pos = df.set_index(["frame", "particle"])[["x", "y"]]
    for i, lag in enumerate(taus):
        df_lag = pos.copy()
        df_lag.index = pd.MultiIndex.from_arrays(
            [
                df_lag.index.get_level_values("frame") - lag,
                df_lag.index.get_level_values("particle"),
            ],
            names=["frame", "particle"],
        )
        joined = pos.join(df_lag, lsuffix="_t", rsuffix="_tlag", how="inner")
        if joined.empty:
            msd_vals[i] = np.nan
            continue
        dx = joined["x_tlag"] - joined["x_t"]
        dy = joined["y_tlag"] - joined["y_t"]
        dr2 = dx * dx + dy * dy
        msd_vals[i] = dr2.mean()

    return {"taus": taus, "values": msd_vals}


def compute_rdf_manual(
    trajectories: pd.DataFrame,
    r_max: float = None,
    n_bins: int = 50,
) -> Dict[str, np.ndarray]:
    if trajectories is None or len(trajectories) == 0:
        return {"r": np.array([]), "g_r": np.array([])}

    needed_cols = {"frame", "x", "y"}
    if not needed_cols.issubset(trajectories.columns):
        return {"r": np.array([]), "g_r": np.array([])}

    df = trajectories[["frame", "x", "y"]].copy()
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    Lx = float(x_max - x_min)
    Ly = float(y_max - y_min)
    if Lx <= 0 or Ly <= 0:
        return {"r": np.array([]), "g_r": np.array([])}

    area = Lx * Ly
    if r_max is None:
        r_max = 0.5 * min(Lx, Ly)

    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    g_r_accum = np.zeros_like(r_centers, dtype=float)
    n_frames_used = 0

    for _, g in df.groupby("frame"):
        coords = g[["x", "y"]].to_numpy()
        n = coords.shape[0]
        if n < 2:
            continue
        dx = coords[:, 0][:, None] - coords[:, 0][None, :]
        dy = coords[:, 1][:, None] - coords[:, 1][None, :]
        r = np.sqrt(dx * dx + dy * dy)
        iu = np.triu_indices(n, k=1)
        r_ij = r[iu]
        hist, _ = np.histogram(r_ij, bins=r_edges)

        rho = n / area
        shell_areas = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
        expected = rho * shell_areas * n

        with np.errstate(divide="ignore", invalid="ignore"):
            g_frame = hist / expected
            g_frame[~np.isfinite(g_frame)] = 0.0
        g_r_accum += g_frame
        n_frames_used += 1

    if n_frames_used == 0:
        return {"r": np.array([]), "g_r": np.array([])}

    g_r = g_r_accum / n_frames_used

    tail_start = int(0.8 * g_r.size)
    if tail_start < g_r.size:
        tail_mean = np.mean(g_r[tail_start:])
        if tail_mean > 0:
            g_r = g_r / tail_mean

    return {"r": r_centers, "g_r": g_r}


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def make_msd_plot(summary: Dict[str, Any]):
    msd = summary.get("msd")
    if not isinstance(msd, dict):
        return None

    taus = np.array(msd.get("taus_s", msd.get("taus", [])))
    vals = np.array(msd.get("values", []))
    if len(taus) == 0 or len(vals) == 0:
        return None

    mask = (taus > 0) & (vals > 0)
    if not np.any(mask):
        return None
    taus = taus[mask]
    vals = vals[mask]

    fig, ax = plt.subplots()
    ax.loglog(taus, vals, "o-", label="MSD")

    tau_ref = np.median(taus)
    msd_ref = np.interp(tau_ref, taus, vals)
    tau_line = np.array([taus.min(), taus.max()])
    msd_line = msd_ref * (tau_line / tau_ref)
    ax.loglog(tau_line, msd_line, "--", color="gray", label="slope 1")

    ax.set_xlabel("lag time τ [s]")
    ax.set_ylabel("MSD")
    ax.set_title("Mean-squared displacement")
    ax.legend()
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


def make_raw_frame_montage(stack_3d, n_frames: int = 3):
    T = stack_3d.shape[0]
    idx = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(3 * len(idx), 3))
    if len(idx) == 1:
        axes = [axes]
    for ax, t in zip(axes, idx):
        img = stack_3d[t].max(axis=0) if stack_3d.ndim == 4 else stack_3d[t]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Frame {t}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def make_annotated_montage(stack_3d, trajectories, n_frames: int = 3):
    if trajectories is None or len(trajectories) == 0:
        return None

    T = stack_3d.shape[0]
    idx = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(3 * len(idx), 3))
    if len(idx) == 1:
        axes = [axes]
    for ax, t in zip(axes, idx):
        img = stack_3d[t].max(axis=0) if stack_3d.ndim == 4 else stack_3d[t]
        ax.imshow(img, cmap="gray")
        df_t = trajectories[trajectories["frame"] == t]
        if not df_t.empty and {"y", "x"}.issubset(df_t.columns):
            ax.scatter(
                df_t["x"],
                df_t["y"],
                s=10,
                edgecolors="r",
                facecolors="none",
            )
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
    for _, df_p in trajectories.groupby("particle"):
        ax.plot(df_p["x"], df_p["y"], "-", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectories (2D projection)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Agent-ish helpers: preset → focus question, compact summary
# ----------------------------------------------------------------------

def preset_to_focus_question(preset: str, user_prompt: str | None) -> str:
    if user_prompt and user_prompt.strip():
        return user_prompt

    mapping = {
        "Diffusion (MSD focus)": (
            "Focus on diffusion, MSD scaling, and the reliability of D and alpha."
        ),
        "Depth / bleaching": (
            "Focus on depth-dependent intensity, bleaching vs time, and SNR limits."
        ),
        "Structure / RDF": (
            "Focus on structure, RDF, and evidence for crowding or clustering."
        ),
        "Trajectories overview": (
            "Summarise trajectories, confinement and qualitative motion types."
        ),
        "Full analysis": (
            "Describe the main physics, including diffusion, depth trends, "
            "structure and trajectory quality."
        ),
    }
    return mapping.get(preset, "Describe the main physics in this dataset.")


def relabel_segments_unique(df: pd.DataFrame, max_frame_gap: int = 1) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df = df.sort_values(["particle", "frame"]).reset_index(drop=True)
    new_ids = np.empty(len(df), dtype=int)
    next_id = 0
    for _, g in df.groupby("particle", sort=False):
        frames = g["frame"].to_numpy()
        segment_starts = np.zeros(len(frames), dtype=bool)
        segment_starts[0] = True
        segment_starts[1:] = (frames[1:] - frames[:-1]) > max_frame_gap
        seg_labels = np.cumsum(segment_starts)
        for seg in np.unique(seg_labels):
            idx_seg = g.index[seg_labels == seg]
            new_ids[idx_seg] = next_id
            next_id += 1
    df = df.copy()
    df["particle"] = new_ids
    return df

def _make_explainer(backend: str) -> ChatExplainer:
    backend = backend or "openai_gpt4o"

    if backend == "openai_gpt4o":
        client = OpenAI()  # uses OPENAI_API_KEY from env
        model = "gpt-4o"
    elif backend == "anthropic_sonnet":
        # Example: Anthropic v2-style interface; adjust to your real client
        client = Anthropic()  # expects ANTHROPIC_API_KEY
        model = "claude-3-5-sonnet-20240620"
    elif backend == "local_small_model":
        # Plug in your local server client here; must expose .chat.completions.create
        client = LocalClient()  # your own class
        model = "local-small"
    else:
        client = OpenAI()
        model = "gpt-4o-mini"

    return ChatExplainer(client=client, model=model)


def make_json_safe_and_compact(full_summary: dict) -> dict:
    msd = full_summary.get("msd", {})
    rdf = full_summary.get("rdf", {})

    def sample_curve(curve_dict, max_points=20):
        taus = curve_dict.get("taus", [])
        vals = curve_dict.get("values", [])

        # Convert to numpy arrays if present, then check length
        taus = np.array(taus)
        vals = np.array(vals)

        if taus.size == 0 or vals.size == 0:
            return {}

        n = min(taus.size, vals.size, max_points)
        idx = np.arange(n)

        return {
            "taus": [float(t) for t in taus[idx]],
            "values": [float(v) for v in vals[idx]],
        }

    compact = {
        "alpha": full_summary.get("alpha"),
        "D": full_summary.get("D"),
        "n_tracks": full_summary.get("quality", {}).get("n_tracks"),
        "mean_track_length": full_summary.get("quality", {}).get("mean_track_length"),
        "snr_estimate": full_summary.get("quality", {}).get("snr_estimate"),
        "diagnostics": {
            "depth_profile": sample_curve(
                full_summary.get("diagnostics", {}).get("depth_profile", {})
            ),
            "bleaching_curve": sample_curve(
                full_summary.get("diagnostics", {}).get("bleaching_curve", {})
            ),
            "crowding_metric": full_summary.get("diagnostics", {}).get(
                "crowding_metric"
            ),
        },
        "msd": sample_curve(msd),
        "rdf": sample_curve(rdf),
        "flags_and_anomalies": full_summary.get("flags_and_anomalies", []),
    }
    return compact


# ----------------------------------------------------------------------
# Detection & tracking wrapper
# ----------------------------------------------------------------------

def run_detection_tracking(stack_3d, plan, backend: str):
    if backend == "DeepTrack (refined)":
        # Placeholder: currently falls back to default worker
        pass

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

    if trajectories is not None and len(trajectories) > 0:
        trajectories = relabel_segments_unique(trajectories, max_frame_gap=1)

    return trajectories, quality


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def run_pipeline(
    dataset_choice,
    analysis_preset,
    custom_prompt,
    tracking_backend,
    llm_backend,
    uploaded_file,
    max_frames,
    use_advanced,
    msd_mode,
    do_depth,
    do_structure,
    do_traj_plot,
    request_download_report,
):
    reasoning_log = []

    # MSD mode
    do_msd = msd_mode != "No MSD"
    use_manual_msd = msd_mode == "MSD (manual, robust)"

    # Remember last run (for future refinement hooks)
    LAST_RUN_PARAMS.update(
        dict(
            dataset_choice=dataset_choice,
            analysis_preset=analysis_preset,
            custom_prompt=custom_prompt,
            tracking_backend=tracking_backend,
            llm_backend=llm_backend,
            max_frames=max_frames,
            use_advanced=use_advanced,
            do_msd=do_msd,
            use_manual_msd=use_manual_msd,
            do_depth=do_depth,
            do_structure=do_structure,
            do_traj_plot=do_traj_plot,
        )
    )

    # Load data
    if dataset_choice != "own_dataset":
        stack_3d, metadata = ensure_sample_dataset(dataset_choice)
        dataset_used = dataset_choice
        reasoning_log.append(f"Using built-in sample dataset '{dataset_choice}'.")
    else:
        if uploaded_file is None:
            warning_text = "⚠️ Please upload a TIFF stack for 'own_dataset'."
            return (
                warning_text,
                None,
                gr.update(visible=True, value=None),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
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

    # Raw montage
    raw_fig = make_raw_frame_montage(stack_3d)

    # Denoise for detection
    stack_for_detection = denoise_stack(stack_3d, method="median", strength=1)
    reasoning_log.append("Applied median denoising before detection (always on).")

    # Quick stats (very simple SNR estimate)
    mean_signal = float(stack_for_detection.mean())
    std_signal = float(stack_for_detection.std() + 1e-6)
    snr_est = mean_signal / std_signal
    quick_stats = {
        "snr_est": snr_est,
        "density_est": 200,
        "D_est": 0.2,
        "search_range_um": 1.5,
    }

    # Build focus question from preset + custom prompt
    focus_question = preset_to_focus_question(analysis_preset, custom_prompt)
    reasoning_log.append(f"Analysis preset: {analysis_preset}")
    reasoning_log.append(f"Tracking backend: {tracking_backend}")
    reasoning_log.append(f"LLM backend: {llm_backend}")

    # Planner
    # Planner (use original Orchestrator API)
    plan = _orch.make_plan(
    custom_prompt if custom_prompt else "Describe the main physics in this dataset.",
    metadata,
    quick_stats,
)
    if hasattr(plan, "detection_params_initial"):
        plan.detection_params_initial["minmass"] = 20.0
        plan.detection_params_initial["max_sigma"] = 3
    if hasattr(plan, "tracking_params_initial"):
        plan.tracking_params_initial["search_range"] = 3.0
    reasoning_log.append(
        f"Planner pipeline='{getattr(plan, 'pipeline_type', 'NA')}'."
    )

    # Detection & tracking
    trajectories, quality = run_detection_tracking(
        stack_for_detection, plan, tracking_backend
    )
    if trajectories is None or len(trajectories) == 0:
        warning = (
            f"⚠️ No trajectories detected for {dataset_used} "
            f"(used {n_use} frames, SNR≈{snr_est:.2f})."
        )
        summary = {
            "metadata": metadata,
            "quick_stats": quick_stats,
            "quality": quality,
            "flags_and_anomalies": ["no_trajectories"],
        }
        safe_summary = _json_safe(summary)
        save_json(ANALYSIS_SUMMARY_PATH, safe_summary)

        annotated_fig = None
        downloadable = (
            TRAJ_CSV_PATH
            if (request_download_report and TRAJ_CSV_PATH.exists())
            else None
        )

        return (
            warning,
            {"summary": safe_summary},
            gr.update(visible=True, value=raw_fig),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            downloadable,
        )

    n_tracks = quality.get("n_tracks", len(trajectories))
    reasoning_log.append(f"Detection/Tracking found {n_tracks} tracks.")

    # Physics analysis
    physics_engine, advanced_used = _choose_physics_engine(use_advanced)
    summary = physics_engine.summarize(trajectories, stack_for_detection, metadata)

    # MSD
    if not do_msd:
        summary.pop("msd", None)
    elif use_manual_msd:
        summary["msd"] = compute_msd_manual(trajectories)

    # Depth
    if not do_depth:
        summary.get("diagnostics", {}).pop("depth_profile_mean_intensity", None)

    # Structure / RDF (manual RDF overwrite)
    if not do_structure:
        summary.pop("structure", None)
    else:
        summary.setdefault("structure", {})
        summary["structure"]["rdf"] = compute_rdf_manual(trajectories)

    summary["metadata"] = metadata
    summary["quick_stats"] = quick_stats
    summary["quality"] = quality

    reasoning_log.append(f"Physics analysis complete. Advanced={advanced_used}.")

    # LLM explanation (single, preset-aware call)
    compact_summary = make_json_safe_and_compact(summary)
    try:
        explainer = _make_explainer(llm_backend)
        explanation = explainer.explain(focus_question, compact_summary)
        reasoning_log.append(
            f"Generated explanation via ChatExplainer (backend='{llm_backend}')."
        )
    except Exception as e:
        # Fallback: simple, deterministic summary like before
        reasoning_log.append(f"LLM explanation failed: {e!r}")
        explanation = (
            "LLM explanation is unavailable; showing a basic numerical summary instead.\n\n"
            f"- Tracks found: {quality.get('n_tracks', 'NA')}\n"
            f"- Mean track length: {quality.get('mean_track_length', 'NA')}\n"
            f"- SNR estimate: {quick_stats.get('snr_est', 'NA'):.2f}\n"
            f"- MSD present: {'msd' in summary}\n"
            f"- Structure (RDF) present: {'structure' in summary}\n"
        )

    # Save JSON and trajectories
    safe_summary_full = _json_safe(summary)
    save_json(ANALYSIS_SUMMARY_PATH, safe_summary_full)
    save_trajectories_csv(TRAJ_CSV_PATH, trajectories)

    text_info = (
        f"## Confocal Microscope Copilot\n\n"
        f"**Dataset:** {dataset_used}\n"
        f"**Frames used:** {n_use} / {n_total}\n"
        f"**Tracks found:** {n_tracks}\n"
        f"**SNR estimate:** {snr_est:.2f}\n"
        f"**Preset:** {analysis_preset}\n"
        f"**Tracking backend:** {tracking_backend}\n\n"
        f"{explanation}\n\n"
        "_This summary is generated by a microscopy-tuned AI model based on the numerical analysis above._"
    )

    annotated_fig = make_annotated_montage(stack_for_detection, trajectories)
    traj_fig = make_trajectory_plot(trajectories) if do_traj_plot else None
    msd_fig = make_msd_plot(summary) if do_msd else None
    depth_fig = make_depth_plot(summary) if do_depth else None
    rdf_fig = make_rdf_plot(summary) if do_structure else None

    analysis_dict = {
        "summary": safe_summary_full,
        "reasoning": "\n".join(reasoning_log),
        "advanced_physics_used": bool(use_advanced and HAS_ADVANCED),
        "trajectories_csv": str(TRAJ_CSV_PATH) if TRAJ_CSV_PATH.exists() else None,
        "llm_backend": llm_backend,
        "suggested_followup": (
            "For example: 'Compare diffusion coefficients in different z-slices' or "
            "'Explain possible sources of subdiffusive behavior'."
        ),
    }

    def vis(fig):
        if fig is None:
            return gr.update(visible=False, value=None)
        return gr.update(visible=True, value=fig)

    downloadable = (
        TRAJ_CSV_PATH if (request_download_report and TRAJ_CSV_PATH.exists()) else None
    )

    return (
        text_info,
        analysis_dict,
        vis(raw_fig),
        vis(annotated_fig),
        vis(traj_fig),
        vis(msd_fig),
        vis(depth_fig),
        vis(rdf_fig),
        downloadable,
    )


# ----------------------------------------------------------------------
# UI helpers and layout
# ----------------------------------------------------------------------

def _toggle_custom_prompt(preset: str):
    return gr.update(visible=(preset == "Custom"))


def _toggle_select_all(
    all_on: bool,
    msd_mode_current: str,
    do_depth_current: bool,
    do_structure_current: bool,
    do_traj_current: bool,
):
    if all_on:
        return (
            "MSD (manual, robust)",
            True,
            True,
            True,
        )
    return (
        msd_mode_current,
        False,
        False,
        False,
    )


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

#title_block {
  margin-bottom: 1.5rem;
  text-align: center;       /* center the markdown text */
}
"""


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        "## Confocal Microscope Copilot\n\n"
        "The app denoises, tracks with Trackpy by default, runs physics analysis,\n"
        "and uses a modern LLM for natural-language explanations.",
        elem_id="title_block",
    )

    with gr.Row():
        with gr.Column(scale=1):
            dataset_choice = gr.Dropdown(
                choices=list(SAMPLE_DATASETS.keys()) + ["own_dataset"],
                value="brownian_3d colloids",
                label="Dataset",
            )
            dataset_info = gr.Markdown(
                value=SAMPLE_DATASETS["brownian_3d colloids"]["label"],
                label="Dataset information",
            )
            uploaded_file = gr.File(
                label="Upload TIFF stack (only for own_dataset)",
                file_types=[".tif", ".tiff"],
                visible=False,
            )

            analysis_preset = gr.Dropdown(
                label="Analysis preset",
                choices=[
                    "Full analysis",
                    "Diffusion (MSD focus)",
                    "Depth / bleaching",
                    "Structure / RDF",
                    "Trajectories overview",
                    "Custom",
                ],
                value="Full analysis",
            )

            custom_info = gr.Markdown(
                value=(
                    "Tip: choose **Custom** in the analysis preset to "
                    "enable the custom physics prompt below."
                )
            )

            custom_prompt = gr.Textbox(
                label="Custom physics prompt (for LLM)",
                placeholder=(
                    "Describe the motion, compare diffusion in different regions, etc."
                ),
                lines=4,
                visible=False,
            )

            tracking_backend = gr.Radio(
                label="Tracking backend",
                choices=[
                    "Trackpy (standard)",
                    "DeepTrack (refined)",
                ],
                value="Trackpy (standard)",
            )

            max_frames = gr.Slider(
                label="Max frames to analyze",
                minimum=10,
                maximum=200,
                step=10,
                value=60,
            )

            run_btn = gr.Button("Run analysis", variant="primary")

            # --- Expert options accordion starts here ---
            with gr.Accordion("Expert options", open=True):
                llm_backend = gr.Dropdown(
                    label="LLM backend",
                    choices=["openai_gpt4o", "anthropic_sonnet", "local_small_model"],
                    value="openai_gpt4o",
                    info="Advanced: choose which LLM to use for explanations.",
                )

                use_advanced = gr.Checkbox(
                    label="Use advanced physics module (if available)",
                    value=False,
                )

                select_all = gr.Checkbox(
                    label="Select all analyses (MSD, depth, RDF, trajectories)",
                    value=True,
                )

                msd_mode = gr.Radio(
                    label="MSD mode",
                    choices=[
                        "No MSD",
                        "MSD (standard, uses msd_gaps)",
                        "MSD (manual, robust)",
                    ],
                    value="MSD (manual, robust)",
                )

                do_depth = gr.Checkbox(
                    label="Compute depth/bleaching profile",
                    value=True,
                )

                do_structure = gr.Checkbox(
                    label="Compute RDF (structure)",
                    value=True,
                )

                do_traj_plot = gr.Checkbox(
                    label="Plot trajectories",
                    value=True,
                )

                

                request_download_report = gr.Checkbox(
                    label="Prepare CSV/report for download (optional)",
                    value=False,
                )
            # --- Expert options accordion ends here ---

            


        with gr.Column(scale=2):
            with gr.Tab("Montages"):
                with gr.Row():
                    raw_plot = gr.Plot(label="Raw", visible=True)
                    ann_plot = gr.Plot(label="Annotated", visible=True)
            with gr.Tab("Trajectories"):
                traj_plot = gr.Plot(visible=True)
            with gr.Tab("MSD"):
                msd_plot = gr.Plot(visible=True)
            with gr.Tab("Depth profile"):
                depth_plot = gr.Plot(visible=True)
            with gr.Tab("RDF"):
                rdf_plot = gr.Plot(visible=True)

            text_out = gr.Markdown(label="Analysis summary & explanation")
            analysis_json = gr.JSON(
                label="Analysis details (JSON)",
                visible=True,
            )

            feedback = gr.Radio(
                label="Was this analysis useful?",
                choices=["👍 Yes", "👎 Not really"],
                value=None,
            )
            feedback_comment = gr.Textbox(
                label="Optional feedback",
                placeholder="What worked well? What should be improved?",
                lines=2,
            )

            download_btn = gr.DownloadButton(
                label="Download latest trajectories CSV/report",
                value=None,
                visible=True,
            )

    # Interactions
    analysis_preset.change(
        fn=_toggle_custom_prompt,
        inputs=[analysis_preset],
        outputs=[custom_prompt],
    )

    select_all.change(
        fn=_toggle_select_all,
        inputs=[select_all, msd_mode, do_depth, do_structure, do_traj_plot],
        outputs=[msd_mode, do_depth, do_structure, do_traj_plot],
    )

    def _update_dataset_info(name):
        if name in SAMPLE_DATASETS:
            return SAMPLE_DATASETS[name]["label"], gr.update(visible=False)
        return (
            "User-supplied dataset. Make sure the stack is a 3D or 4D TIFF.",
            gr.update(visible=True),
        )

    dataset_choice.change(
        fn=_update_dataset_info,
        inputs=[dataset_choice],
        outputs=[dataset_info, uploaded_file],
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            dataset_choice,
            analysis_preset,
            custom_prompt,
            tracking_backend,
            llm_backend,
            uploaded_file,
            max_frames,
            use_advanced,
            msd_mode,
            do_depth,
            do_structure,
            do_traj_plot,
            request_download_report,
        ],
        outputs=[
            text_out,
            analysis_json,
            raw_plot,
            ann_plot,
            traj_plot,
            msd_plot,
            depth_plot,
            rdf_plot,
            download_btn,
        ],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css)

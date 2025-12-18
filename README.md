# Confocal Microscopy Copilot

A digital‑twin–assisted copilot for 3D confocal particle tracking. It combines:

- a simple **digital twin** for confocal stacks,
- a modular **multi‑agent pipeline** (planner → detection/tracking → physics analysis → explainer),
- a **Gradio UI** to load datasets, run analysis, visualise MSD/depth profiles, download trajectories, and ask questions.

The goal is to help design and diagnose confocal experiments (diffusion vs structure, depth/bleaching/crowding biases, and “what‑if” scenarios).

---

## Repository structure

### Root

- `README.md`  
  Project overview, folder descriptions, and run instructions.

- `requirements.txt`  
  Python dependencies (numpy, scipy, pandas, trackpy, tifffile, gradio, etc.).

- `ui_app.py`  
  Gradio web UI entry point. Provides:
  - dataset selection (example / ImageJ confocal / synthetic Gaussian / DeepTrack / own TIFF),
  - frame cropping slider (max frames to use),
  - suggested prompts + custom prompt,
  - run button to execute the multi‑agent pipeline,
  - visual outputs (MSD and depth plots),
  - JSON summary and trajectories CSV download,
  - follow‑up questions that reuse cached analysis (no recompute).

---

### `copilot/` – core library and agents

All reusable logic lives here.

- `config.py`  
  Default configuration:
  - voxel size (µm),
  - image shape (z, y, x),
  - PSF widths (voxels),
  - frame interval (s),
  - default noise, depth attenuation length, bleaching time constant,
  - simulation box size (µm).

- `io_utils.py`  
  IO utilities:
  - `load_stack(path)` / `save_stack(path, stack)` for TIFF stacks,
  - `load_metadata(path)` / `save_metadata(path, meta)` for JSON metadata,
  - `save_json(path, data)` for analysis summaries,
  - `save_trajectories_csv(path, df)` for Trackpy trajectories.

- `digital_twin.py`  
  Minimal confocal digital twin:
  - simulates 3D Brownian trajectories in a box,
  - renders “confocal‑like” 4D stacks (T, Z, Y, X) with:
    - anisotropic 3D Gaussian PSF,
    - depth‑dependent intensity attenuation,
    - global bleaching vs time,
    - additive noise,
  - used to generate `example_stack` and as a base for “what‑if” scenarios.

- `orchestrator.py` – **Agent 1: Planner**  
  Input:
  - user question (text),
  - dataset metadata,
  - quick statistics (SNR estimate, density estimate, rough diffusion coefficient).  
  Output:
  - a `Plan` object specifying:
    - pipeline type (diffusion vs structure),
    - whether/when to use the twin,
    - initial detection parameters (e.g. `minmass`, `min_sigma`, `max_sigma`),
    - initial tracking parameters (e.g. `search_range`, `memory`).

- `detection_tracking.py` – **Agent 2: Detection & Tracking**  
  Responsibilities:
  - optional max‑projection of the 3D stack to 2D per frame,
  - feature detection with Trackpy (`tp.locate`),
  - linking to trajectories with Trackpy (`tp.link_df`),
  - returns:
    - `trajectories` DataFrame (frame, x, y, particle, …),
    - `quality_metrics` (number of tracks, detections per frame, track‑length histogram).

- `physics_analysis.py` – **Agent 3: Physics Analyst**  
  Responsibilities:
  - compute MSD from trajectories (Trackpy `tp.motion.msd`),
  - fit anomalous exponent α and effective diffusion coefficient \(D\),
  - compute diagnostics:
    - depth‑dependent mean intensity profile,
    - bleaching curve (mean intensity vs frame),
    - simple crowding metric (nearest‑neighbor distances),
  - handle problematic trajectories robustly (drops duplicate frame indices, catches MSD errors),
  - returns a JSON‑friendly `summary` dict containing:
    - `msd` (taus, values),
    - `alpha`, `D`,
    - `diagnostics` (depth profile, bleaching, crowding metrics).

- `chat_explainer.py` – **Agent 4: Explainer / LLM**  
  Responsibilities:
  - read user question + JSON summary,
  - build a prompt at “soft‑matter postdoc” level,
  - produce a brief explanation + next‑experiment suggestions,
  - currently uses a dummy explainer (LLM client can be attached later).

---

### `data/`

- Example and real datasets:
  - `example_stack.tif` / `example_stack_meta.json` – synthetic example from the digital twin,
  - optional subfolders:
    - `imagej_confocal/stack.tif` + `metadata.json`,
    - `synthetic_gaussian/stack.tif` + `metadata.json`,
    - `deeptrack_example/stack.tif` + `metadata.json`.

- `analysis_summary.json` – latest analysis summary written by the physics analyst.

---

### `results/`

- Output artifacts for download from the UI:
  - `trajectories_latest.csv` – most recent trajectories used for analysis.

---

### `notebooks/`

- `notebooks/main_demo.ipynb`  
  Step‑by‑step demo of the pipeline (no UI):
  - load or simulate a stack,
  - compute quick stats and a `Plan`,
  - run detection + tracking,
  - compute MSD and diagnostics,
  - generate an explanation.

- `notebooks/ui_demo.ipynb` (optional)  
  Notebook that launches the Gradio UI from inside Jupyter/Colab.

---

### `legacy/` (optional)

- Older single‑file scripts (`analysis.py`, `detection.py`, `tracking.py`, etc.) kept for reference. New work should use `copilot/` modules.

---

## Installation

Clone the repo and create a virtual environment:

git clone https://github.com/Abhishek-Gupta-GitHub/confocal_microscopy-copilot.git

cd confocal_microscopy-copilot

python -m venv .venv

Windows:
.venv\Scripts\activate

Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

To ensure Gradio is available: pip install gradio

## Running the Gradio UI

From the project root (with the virtual environment activated):

python ui_app.py 

Gradio will print a line such as: Running on local URL: http://127.0.0.1:7860


Open that URL in your browser.

### UI workflow

1. **Choose dataset**
   - `example`: synthetic stack from the digital twin.
   - `imagej_confocal`, `synthetic_gaussian`, `deeptrack_example`: example stacks (if present under `data/`).
   - `own_dataset`: upload your own 3D/4D TIFF stack.

2. **Limit frames**
   - Use the “Max frames to use” slider to crop long movies (e.g. 60/100 frames) to speed up tracking and analysis.

3. **Choose / write an analysis prompt**
   - Pick a template in “Suggested prompts”, e.g.:
     - “Analyze diffusion and comment on depth and bleaching.”
     - “Quantify subdiffusion and identify a reliable MSD fit window.”
   - Or select `Custom` and type your own question in the text box.

4. **Run analysis**
   - Click **“▶ Run analysis”**.
   - The agents will:
     - **Agent 1** – Planner: choose pipeline and initial parameters from metadata and quick SNR.
     - **Agent 2** – Detection & Tracking: run Trackpy detection and linking on the cropped stack.
     - **Agent 3** – Physics: compute MSD, α, D, and diagnostics (depth profile, bleaching, crowding).
     - **Agent 4** – Explainer: produce a short explanation and suggestions.

5. **Inspect results**
   - **Copilot explanation** – human‑readable interpretation + next‑experiment suggestions.
   - **Agent reasoning (debug view)** – concise log of what each agent decided (dataset choice, cropping, parameters, analysis steps).
   - **MSD plot** – log–log MSD vs lag time.
   - **Depth profile plot** – mean intensity vs z index.
   - **Analysis summary (JSON)** – structured output usable in notebooks/code (available via file or state).

6. **Download trajectories**
   - Click **“Download trajectories (CSV)”** to get `results/trajectories_latest.csv`.
   - Contains Trackpy‑style columns (`frame`, `x`, `y`, `particle`, …) for further custom analysis.

7. **Ask follow‑up questions (no re‑analysis)**
   - In “Follow‑up questions (no recompute)”, type a question about the last run (e.g. “How reliable is MSD at long times?”).
   - Click **“Ask follow‑up”**.
   - The explainer reuses the cached analysis summary (no detection/tracking/MSD recomputation) and returns an answer based on the previous run.

---

## Running the demo notebook

To explore the pipeline step by step:

1. Open `notebooks/main_demo.ipynb` in VS Code (Jupyter) or JupyterLab.
2. Run cells from top to bottom:
   - load or simulate a dataset,
   - compute SNR and a `Plan`,
   - run detection + tracking,
   - compute MSD, α, D, and diagnostics,
   - generate the textual explanation.

On Google Colab, use the “Open in Colab” badge (if defined) pointing to `notebooks/main_demo.ipynb`:

[![Open In Colab](https://colab.research.google.com/assets/(
https://colab.research.google.com/github/Abhishek-Gupta-GitHub/confocal_microscopy-copilot/blob/main/notebooks/main_demo.ipynb
)



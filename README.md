## MicrosCopilot: Confocal Microscopy Copilot

MicrosCopilot is a **physics‑aware framework and GUI** for quantitative confocal microscopy. It helps turn raw 3D/4D confocal stacks into interpretable physical metrics such as mean‑squared displacement (MSD), anomalous diffusion exponents, and imaging diagnostics (bleaching, depth‑loss, crowding) in a transparent, reproducible way.  

The central idea is to combine a configurable digital twin, a modular analysis pipeline, and an interactive interface so that experimentalists can analyse, debug, and interpret confocal experiments without relying on opaque black‑box tools.  

---

## Quick start: demo UI (`ui_demo_final.py`)

1. **Clone the repository**

```python
git clone https://github.com/Abhishek-Gupta-GitHub/confocal_microscopy-copilot.git
cd confocal_microscopy-copilot

```

2. **(Optional) create and activate a virtual environment**

```python
python -m venv .venv
source .venv/bin/activate # Linux/macOS

.venv\Scripts\activate # Windows
```

3. **Install dependencies**

If `requirements.txt` exists:

```python
pip install -r requirements.txt
```

Otherwise, install the main stack (adapt to your environment as needed):

```python
pip install numpy matplotlib pandas tifffile trackpy gradio pyqt5
```

4. **Run the demo UI**

```python
python ui_demo_final.py
```

5. **In the UI you can**

- **Open the local app**  
  - After running `python ui_demo_final.py`, a temporary local URL (e.g. `http://127.0.0.1:7860` or similar) will appear in the terminal.  
  - Click that link or paste it into your browser to access the interface.

- **Load data and basic metadata**  
  - Load an example confocal stack or upload your own 3D/4D dataset.  
  - Enter basic metadata such as voxel size, frame interval, and other acquisition parameters.

- **Choose analysis options**  
  - Select a tracking backend (e.g. Trackpy or other available methods).  
  - Provide an optional natural‑language prompt describing your analysis goal (e.g. “estimate diffusion in the z‑direction and check for bleaching”).  
  - Toggle expert/advanced options, such as:
    - Using or comparing against the physics‑informed digital twin.  
    - Enabling advanced diagnostics (depth‑dependent intensity profiles, crowding metrics).  
    - Choosing an LLM backend (if configured) for richer explanations or planning.

- **Run analysis**  
  - Particle detection and tracking.  
  - MSD and anomalous‑diffusion analysis.  
  - Imaging diagnostics (photobleaching curves, depth‑dependent intensity, crowding metrics). [file:70]

- **Inspect results and explanations**  
  - View plots of trajectories, MSDs, diagnostics, and fitted parameters.  
  - Read an explanation panel that summarises the results, flags possible limitations (e.g. short tracks, strong bleaching), and suggests follow‑up experimental changes.
 

---

## Objectives and design philosophy

MicrosCopilot is designed around recurring challenges in quantitative confocal microscopy:  

- **Bridge raw data and physical insight**  
  - Provide end‑to‑end workflows from image stacks to MSDs, diffusion exponents, and imaging diagnostics, with parameters and assumptions exposed.  

- **Make analysis transparent and physics‑grounded**  
  - Prioritise clear diagnostics (bleaching, depth‑loss, crowding) and explicit model choices over purely data‑driven, black‑box predictions.  

- **Unify simulation and experiment**  
  - Use a digital twin to generate synthetic confocal‑like data with known ground truth, enabling validation and stress‑testing of analysis pipelines before applying them to experimental datasets.  

- **Support collaboration and education**  
  - Offer an interface and explanation layer that encode expert heuristics in a form that students and collaborators can inspect, adapt, and build on.

---

## Core components

MicrosCopilot combines three main components into one workflow.  

### Physics‑informed digital twin

- Simulates 3D Brownian particle trajectories with configurable diffusion coefficient, particle density, voxel size, and frame interval.  
- Renders trajectories into 4D confocal‑like image stacks using an anisotropic 3D Gaussian PSF.  
- Incorporates key imaging artifacts:
  - Depth‑dependent attenuation.  
  - Global photobleaching (exponential intensity decay over time).  
  - Additive Gaussian noise.  
- Enables benchmarking and sensitivity analysis: compare recovered MSDs and exponents to the known ground truth under controlled artifact levels.  

### Modular analysis pipeline (“agents”)

- **Planner**  
  - Ingests user goals, dataset metadata, and preferences to propose an explicit analysis plan (mode, whether to use the digital twin for comparison, initial tracking parameters).  

- **Detection & Tracking**  
  - Converts 3D/4D stacks into particle trajectories using established tools such as Trackpy (feature localization and linking).  
  - Produces trajectory tables and quality metrics (detection counts, track‑length distributions).  

- **Physics Analysis**  
  - Computes ensemble‑averaged MSD curves.  
  - Fits anomalous‑diffusion models to extract exponents and effective diffusion coefficients.  
  - Evaluates imaging diagnostics: bleaching curves, depth‑intensity profiles, crowding metrics based on nearest‑neighbour distances.  

- **Explainer**  
  - Translates numerical outputs into concise, experiment‑focused interpretations.  
  - Highlights limitations (e.g. short tracks, unreliable long‑lag MSDs) and suggests concrete next steps (change frame rate, adjust laser power, alter acquisition depth).  

### Interactive interface

- Lightweight UI (via `ui_demo_final.py` and/or a web interface) that exposes the full pipeline without hiding intermediate results.  
- Intended for day‑to‑day use by experimentalists, method developers, and students working with confocal data.  

---

## Why it is useful for scientists and researchers

MicrosCopilot is particularly suited for soft‑matter physics, microrheology, biophysics, and cell‑imaging workflows where confocal stacks must be turned into quantitative transport metrics.  

- **Reduces manual trial‑and‑error**  
  - Encodes tracking and analysis heuristics into a modular pipeline that can be reused and tuned systematically.  
  - Digital twin allows rapid testing of parameter choices and failure modes before committing microscope time.

- **Improves trust and interpretability**  
  - Pairs physical metrics (MSDs, exponents) with imaging diagnostics to separate experimental artifacts from genuine physical phenomena.  
  - Makes analysis steps, models, and assumptions explicit and inspectable, facilitating review and reproducibility.  

- **Enhances teaching and collaboration**  
  - UI and explanation layer provide an accessible entry point for students and new collaborators to learn quantitative confocal analysis.  
  - Serves as a template for physics‑aware, interpretable “copilot” tools in other microscopy or imaging domains.  

---

## Optional LLM integration (API keys)

The planner and explainer are structured so they can optionally use a large language model for richer natural‑language planning and explanation, but **LLM integration is not required** for the core pipeline.  

To enable LLM‑based features:

1. Choose a provider (e.g. OpenAI, Anthropic, or another supported backend).  
2. Set an API key through environment variables, for example:

```python
export OPENAI_API_KEY="YOUR_API_KEY_HERE" # Linux/macOS
setx OPENAI_API_KEY "YOUR_API_KEY_HERE" # Windows
```

or via a small config module that the code reads:

```python
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
```


3. Ensure any file containing secrets is added to `.gitignore` and not committed to the repository.

Without a key, the digital twin, tracking, physics analysis, diagnostics, and plots remain fully usable.

---

## Repository structure (indicative)

Exact paths may differ, but conceptually the project is organised as follows.  

- `ui_demo_final.py` – main entry point for the demo GUI.  
- Digital‑twin modules – simulation of trajectories and confocal‑like stacks with artifacts.  
- Analysis modules – detection, tracking, MSD computation, diffusion fitting, diagnostics.  
- Agent logic – planner, physics analysis, explainer.  
- Notebooks – example workflows and experiments.  
- `requirements.txt` – Python dependencies (if present).

---

## Contributing

Potential extensions include:

- Improved tracking/localisation and segmentation backends.    
- Richer digital‑twin models (viscoelastic motion, additional artifacts, extended optical models).  
- New visualisations and domain‑specific presets in the UI.  

Issues and pull requests are welcome; contributions that preserve the framework’s transparency, physics‑awareness, and usability for experimentalists are especially encouraged.

# Confocal Microscopy Copilot

An **AI-assisted confocal microscopy copilot** for simulating, analyzing, and explaining Brownian particle movies and soft‑matter experiments. It combines a digital twin of Brownian motion, classical image analysis, particle tracking, and LLM‑based agents to help experimentalists go from raw movies to quantitative physics and plain‑language reports.

## Key Features

- Load or simulate confocal microscopy movies of Brownian particles.
- Detect and track particles over time to build trajectories.
- Estimate physical parameters such as diffusion coefficients and mean‑square displacement.
- Use a configurable digital twin to compare experiment vs simulation.
- Multi‑agent architecture (data I/O, detection/tracking, physics analysis, explanation).
- Optional LLM‑based copilot to explain results, suggest next steps, and help with troubleshooting.

## Project Goals

- Provide a simple, reproducible pipeline for confocal particle tracking experiments.
- Help materials / soft‑matter / tribology researchers quickly extract quantitative information.
- Demonstrate how LLM copilots and agents can assist with microscopy workflows (not replace expert judgment).

## Repository Structure

A typical structure (your exact layout may differ slightly):

- `copilot/` – core multi‑agent logic (data loader, tracker, physics analyst, explainer).
- `notebooks/` – Jupyter notebooks for demos and hackathon workflows.
- `data/` – example datasets and synthetic movies (or download scripts).
- `legacy/` – older scripts and experiments kept for reference.
- `ui_demo.py` / `main_demo.py` – entry points for the interactive demo UI.
- `config.py` – configuration, paths, and default parameters.
- `io_utils.py` – helper functions for loading/saving data and results.

## Installation

Clone the repository
git clone https://github.com/Abhishek-Gupta-GitHub/confocal_microscopy-copilot.git
cd confocal_microscopy-copilot

(Recommended) create and activate a virtual environment, then install dependencies
pip install -r requirements.txt


If a `requirements.txt` is not yet complete, install the main scientific stack manually (e.g. `numpy`, `scipy`, `matplotlib`, `pandas`, `trackpy`, `deeptrack`, and any UI / LLM dependencies you use).

## Quick Start

### 1. Run the main demo


- Loads or generates a test Brownian particle movie.
- Runs detection and tracking.
- Estimates diffusion‑related quantities.
- Optionally launches an interactive UI or notebook interface to inspect results.

### 2. Use the notebook workflow

1. Open the `notebooks/` folder.
2. Start Jupyter Lab / Notebook.
3. Run the main hackathon notebook step by step:
   - Load or simulate data.
   - Detect and track particles.
   - Analyze MSD / diffusion.
   - Compare with the digital twin.
   - Call the copilot agent for explanation and reporting.

## LLM / Copilot Integration

If LLM support is enabled:

- Configure your API key and model name in `config.py` or via environment variables.
- The **ChatExplainer** (or equivalent agent) can:
  - Summarize analysis results.
  - Explain physical meaning (e.g. diffusion coefficient, Brownian motion regimes).
  - Suggest parameter changes or follow‑up experiments.

LLM integration is optional; the classical analysis pipeline works without it.

## Use Cases

- Confocal Brownian motion experiments in soft matter and colloids.
- Educational demos for particle tracking and diffusion analysis.
- Rapid prototyping of microscopy copilot ideas for hackathons and lab projects.

## Keywords

Confocal microscopy, Brownian motion, particle tracking, diffusion coefficient, soft matter physics, AI copilot, multi‑agent system, digital twin, image analysis pipeline, LLM‑driven microscopy assistant

## License

Specify your chosen license here (for example, MIT, BSD, or Apache‑2.0). Add the corresponding `LICENSE` file in the repository root.

## Acknowledgements

- Built during a global microscopy / AI hackathon as a proof‑of‑concept **confocal physics copilot**.
- Inspired by existing work in confocal microscopy, particle tracking, and LLM‑based copilots for scientific workflows.

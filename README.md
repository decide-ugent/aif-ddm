# aif-ddm
This repository contains code to reproduce the results presented in the paper [Cognitive Effort in the Two-Step Task: An Active Inference Drift-Diffusion Model Approach](https://arxiv.org/abs/2508.04435), accepted at [IWAI2025](https://iwaiworkshop.github.io/).
This code is a modified version of the code authored by Dr. Sam Gijsen for the paper [Active inference in the two-step task](https://www.nature.com/articles/s41598-022-21766-4). The original code can be found [here](https://github.com/SamGijsen/AI2step). 


TL;DR

Fit models to data

Compare models (AIC/BIC, CV, etc.)

Recover models/parameters via simulation

Analyze and visualize results

Repo structure
.
├─ data_analysis/           # Exploration, summaries, and plots of datasets & results
├─ model_comparison/        # Metrics and routines to compare fitted models
├─ model_fitting/           # Scripts/pipelines to fit models to data (uses MLE.py)
├─ model_recovery/          # Simulate data from each model and test model identifiability
├─ parameter_recovery/      # Simulate/fit loops to check parameter recoverability
├─ test_model_predictions/  # Generate & plot model predictions vs. empirical behavior
├─ utils/                   # Shared helpers (I/O, plotting, random seeds, etc.)
├─ MLE.py                   # Maximum-likelihood fitting utilities
├─ models.py                # Model definitions and likelihoods
├─ LICENSE
└─ README.md

Folders at a glance
Path	What’s inside	Typical entry point
data_analysis/	Notebooks/scripts for exploratory analysis and figures.	analysis_*.ipynb or make_figures.py
model_fitting/	Fit models to datasets; saves fitted params & logs.	fit_all.py
model_comparison/	Compute AIC/BIC, WAIC, CV; ranking & tables.	compare_models.py
model_recovery/	Simulate from each model, refit, build confusion matrix.	run_model_recovery.py
parameter_recovery/	Simulate with known params, refit, correlate true vs. estimated.	run_parameter_recovery.py
test_model_predictions/	Predict behavior from fitted params; generate plots.	predict_and_plot.py
utils/	Shared utilities: data loading, plotting, seeds, paths.	utils/*.py
MLE.py	Optimizers/wrappers for MLE (bounds, restarts, logging).	Imported by fitting scripts
models.py	Model classes & likelihoods; add new models here.	Imported by most modules

(If your actual script names differ, just swap in the correct filenames.)

Quickstart
# 1) Create env (optional)
python -m venv .venv && source .venv/bin/activate       # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt                          # or: pip install numpy pandas scipy matplotlib

# 3) Fit models to your dataset
python model_fitting/fit_all.py --data path/to/data.csv --out runs/fit/

# 4) Compare models
python model_comparison/compare_models.py --fits runs/fit/ --out runs/compare/

# 5) Check recovery (optional but recommended)
python model_recovery/run_model_recovery.py --out runs/recovery/
python parameter_recovery/run_parameter_recovery.py --out runs/param_recovery/

# 6) Generate predictions & plots
python test_model_predictions/predict_and_plot.py --fits runs/fit/ --out runs/predictions/

# 7) Explore results
python data_analysis/make_figures.py --runs runs/

Typical workflow

Add/clean data → put files under data/ (or pass a path).

Define/modify models in models.py (likelihood, constraints, defaults).

Fit with scripts in model_fitting/ (uses MLE.py).

Compare with model_comparison/ (tables, ranks).

Validate with model_recovery/ and parameter_recovery/.

Predict & visualize in test_model_predictions/ and data_analysis/.

Adding a new model

Implement a class or function in models.py that exposes:

log_likelihood(params, data)

simulate(params, n, rng) (for recovery/tests)

Register the model name so fitting/comparison scripts can find it.

(Optional) Add a default prior or parameter bounds used by MLE.py.

Results layout (suggested)
runs/
  fit/
  compare/
  recovery/
  param_recovery/
  predictions/
  figures/

Reproducibility

Set seeds via utils/seeding.py (or a CLI flag like --seed 123).

Save:

fitted params (CSV/JSON),

optimizer logs,

metrics per subject/model,

environment info (Python, package versions).

License

This project is licensed under the terms in LICENSE.

Citation

If you use this code, please cite:

@misc{<your-key>,
  title        = {<repo-name>},
  author       = {<you>},
  year         = {2025},
  howpublished = {\url{https://github.com/<user>/<repo>}}
}

# aif-ddm
This repository contains code to reproduce the results presented in the paper [Cognitive Effort in the Two-Step Task: An Active Inference Drift-Diffusion Model Approach](https://arxiv.org/abs/2508.04435), accepted at [IWAI2025](https://iwaiworkshop.github.io/).
This code is a modified version of the code authored by Dr. Sam Gijsen for the paper [Active inference in the two-step task](https://www.nature.com/articles/s41598-022-21766-4). The original code can be found [here](https://github.com/SamGijsen/AI2step). 


### TL;DR

- Fit models to data

- Compare models based on AIC and BIC scores

- Perform model and parameter recovery analysis

- Anlyse and visualise results
  

### Repository content

```text
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
```

### Citation

If you use this code, please cite:

```python
@misc{<your-key>,
  title        = {<repo-name>},
  author       = {<you>},
  year         = {2025},
  howpublished = {\url{https://github.com/<user>/<repo>}}
}
```

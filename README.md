# aif-ddm
This repository contains code to reproduce the results presented in the paper [Cognitive Effort in the Two-Step Task: An Active Inference Drift-Diffusion Model Approach](https://arxiv.org/abs/2508.04435), accepted at [IWAI2025](https://iwaiworkshop.github.io/).
This code is a modified version of the code authored by Dr. Sam Gijsen for the paper [Active inference in the two-step task](https://www.nature.com/articles/s41598-022-21766-4). The original code can be found [here](https://github.com/SamGijsen/AI2step). 


### TL;DR

- Fit models to data

- Compare models based on AIC and BIC scores

- Perform model and parameter recovery analysis

- Analyse and visualise results

### Setup

Before running any code, create the conda environment using the provided environment file:

```bash
conda env create -f environment.yml
```

  

### Repository content

```text
.
├─ exploratory_data_analysis/    # Scripts to perform exploratory data analysis
├─ model_comparison/             # Metrics and routines to compare fitted models
├─ model_fitting/                # Scripts/pipelines to fit models to data (uses MLE.py)
├─ model_recovery/               # Simulate data from each model and fit synthetic data to check model recoverability
├─ parameter_recovery/           # Simulate/fit scripts to check parameter recoverability
├─ test_model_predictions/       # Generate & plot model predictions vs. empirical behavior
├─ utils/                        # Utilities for the two-step task environment and helper functions
├─ MLE.py                        # Maximum-likelihood fitting utilities
├─ models.py                     # Model definitions and likelihoods
├─ LICENSE
└─ README.md
```

### Citation

If you use this code, please cite:

```python
@inproceedings{
anonymous2025cognitive,
title={Cognitive Effort in the Two-Step Task: An Active Inference Drift-Diffusion Model Approach},
author={Anonymous},
booktitle={6th International Workshop on Active Inference},
year={2025},
url={https://openreview.net/forum?id=riwG9aNPKR}
}
```

Alternatively, if you only use the pure AIF/HRL models (without DDMs), you may also cite:

```python
@article{gijsen2022active,
  title={Active inference and the two-step task},
  author={Gijsen, Sam and Grundei, Miro and Blankenburg, Felix},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={17682},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

```
### Contact

If you have any questions, please feel free to reach out at:  
`alvaro.garridoperez (at) ugent.be`
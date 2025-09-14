# Scripts for model recovery analysis

### Contents
- **main_script.py** – entry point for running the analysis  
- **helper_functions.py** – utility functions  
- **example_notebook.ipynb** – walkthrough with explanations  

## How to run
1. Navigate to this folder in the root directory:
   ```bash
   cd parameter_recovery
   ```
2. Run the Jupyter notebook ```synthetic_data_generator.ypnb``` to generate a simulated dataset for each of the models you are testing. Note that you must change the appropriate variables every time you run it. Each run will generate a folder with the simulated data.
   ```bash
   model = "RL_ddm_biased" #Class of model. YOU MAY CHANGE THIS to RL, RL_ddm, RL_ddm_biased, AI, or AI_ddm
   mtype = 3 #Subclass of Active Inference model (only relevant if model = AI or AI_ddm). YOU MAY CHANGE THIS to 0, 1, 2 or 3
   drmtype = "linear" # Drift rate model:  YOU MAY CHANGE THIS to linear, sigmoid, sigmoid_single_v_mod, sigmoid_single_v_max
   dataset = 'magic_carpet_2020' # Name of the dataset
   '''
   


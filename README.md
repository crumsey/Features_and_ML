# REQUIREMENTS

- Python major version : Python 3
- Required packages    : numpy, scikit-learn, matplotlib









# MACHINE LEARNING TRAINING

## STEP 1

- Copy the file containing variables (to be used in feature construction) for each flow condition to the `input_files` folder

- Copy the file containing corresponding beta field for each flow condition to the `input_files` folder too

## STEP 2

- Edit `driver_features.py` if and as required

- Run `python driver_features.py`

- This will combine the data from variables and corresponding beta fields from all flow conditions, shuffle (or randomize) the data, and calculate features from the
  provided variables. The final output of this step contains two files - `input_files/features_shuffled.dat` and `input_files/beta_target_shuffled.dat`, which can be used for training now

## STEP 3

- Edit `driver_ML.py` if and as required

- Run `python driver_ML.py`

- The plot of `beta_inverse vs. beta_ML` for latest iteration shall be saved in the `figs` folder as `training_quality_<n_iter>.png`

- A fortran file `output_files/ml_injection.f90` is created based on the latest iteration to be used for injection purposes in the flow solver.









# MACHINE LEARNING INJECTION

## STEP 1

- [IN THE FLOW SOLVER] Ensure that a variable has been created in the residual evaluation function for the beta field

## STEP 3

- [IN THE FLOW SOLVER] Import the `ML_injection` module into the residual evaluation subroutine for the turbulence model in the flow solver as
  ```
  use ML_injection
  ```

- [IN THE FLOW SOLVER] Call the `eval_beta` subroutine (definition can be found in `output_files/ml_injection.f90`) to get the beta field, right 
  before evaluating the residuals inside the turbulence model in the flow solver

- [OPTIONAL STEP] Under-relax beta if needed as `beta_at_this_time_step = alpha * beta_at_last_time_step + (1-alpha) * beta_injected_by_ML`

- [IN THE MAKEFILE FOR THE FLOW SOLVER] Edit the makefile of the flow solver to include the compilation of `ml_injection.f90` and the `Neural_Network` module

- Compile the flow solver

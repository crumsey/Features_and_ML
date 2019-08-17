# MACHINE LEARNING TRAINING

## STEP 1

- Copy the file containing variables (to be used in feature construction) for each flow condition to the input_files folder

- Copy the file containing corresponding beta field for each flow condition to the input_files folder too

## STEP 2

- Edit `generate_features.py` if and as required

- Run `python generate_features.py`

- This will combine the data from variables and corresponding beta fields from all flow conditions, shuffle (or randomize) the data, and calculate features from the
  provided variables. The final output of this step contains two files - `features_shuffled.dat` and `beta_target_shuffled.dat`, which can be used for training now

## STEP 3

- Edit "train.py" if and as required

- Run `python train.py <restart_iteration> <n_iteration>`. Here `restart_iteration` is the iteration to restart the learning from and `n_iteration` is the number of
  iterations to be run. So for example:
  ```
  python train.py 50 70
  ```
  would start the learning from the 50th iteration (provided that the appropriate restart files for weights is available in the `output_files/weights` folder as `weights_50.dat`)
  and run the learning for 70 more iterations, i.e., until the iteration 120 and then save a restart file as `weights_120.dat` in the `output_files/weights` folder.

- The plot of `beta_inverse vs. beta_ML` for latest iteration shall be saved in the `figs` folder as `training_quality_<n_iter>.png`

- A shared library "injection.so" is updated based on the latest iteration to be used for injection purposes in the flow solver.








# MACHINE LEARNING INJECTION

## STEP 1

- [IN THE FLOW SOLVER] Ensure that a variable has been created in the residual evaluation function for the beta field

## STEP 2

- Copy the shared library `injection.so` from the `output_files` folder to the source code of the flow solver

## STEP 3

- [IN THE FLOW SOLVER] Import the `ML_injection` module into the residual evaluation subroutine for the turbulence model in the flow solver as
  ```
  use ML_injection
  ```

- [IN THE FLOW SOLVER] Call the `eval_augmentation` subroutine (definition can be found in `output_files/feature_module.f90`) to get the beta field, right 
  before evaluating the residuals inside the turbulence model in the flow solver

- [OPTIONAL STEP] Under-relax beta if needed as `beta_at_this_time_step = alpha * beta_at_last_time_step + (1-alpha) * beta_injected_by_ML`

- [IN THE MAKEFILE FOR THE FLOW SOLVER] Edit the makefile of the flow solver such that the source file containing the residual evaluation subroutine for the 
  turbulence model is compiled with the shared library mentioned above

- Compile the flow solver
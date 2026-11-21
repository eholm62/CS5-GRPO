# Purdue CS173 Final Project
## Group CS5


### Dependencies
 - PyTorch
 - Pandas
 - NumPy

Install them within a venv via:
`pip install torch pandas numpy`


# Notes On Current State
 - This code has not yet been thoroughly tested, and 
 `train_grpo.py` is currently fully untested
 - `train_grpo.py` currently covers all of the tasks of training
 the model, including the reward function and logging
 - The Gemma4 instance that generated `train_grpo.py` claims it will
 parse any model output via regex to try to find an integer within
 it in case the output contains non-numeric characters
   - This should probably be removed in the future such that any output with
   non-numeric characters should not be rewarded, that way the model
   learns only to output the desired integer and no more
 - `test_grpo.py` currently returns `FAILED (failures=1, errors=2)`
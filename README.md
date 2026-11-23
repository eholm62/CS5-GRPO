# Purdue CS173 Final Project
## Group CS5

### Description
`train_grpo.py` uses a custom implementation of `GRPOTrainer`, found in `grpo.py`, that closely matches the interface for TRL's `GRPOTrainer`, in order to train a small base model, Qwen2.5-0.5B by default, to count the number of a given character in a provided string, while logging reward over time and final weights.

Because the `GRPOTrainer` from `grpo.py` closely matches the interface for the `GRPOTrainer` from trl,
, you only need to replace the `from grpo import GRPOTrainer, GRPOConfig` in `train_grpo.py` with 
`from trl import GRPOTrainer, GRPOConfig` in order to switch to TRL's `GRPOTrainer` implementation if desired.

### Usage
  - Create a venv in the repo via `python3 -m venv .venv`
  - Install the necessary dependencies as shown below in Dependencies.
  - Train a model on `letter_counting_dataset.csv` with `.venv/bin/python3 train_grpo.py --csv letter_counting_dataset.csv`
    - Depending on your hardware and whether you want to run on CPU or GPU or a combination, you may need to adjust 
    certain environment variables, such as CUDA_VISIBLE_DEVICES for example. You also need to install the correct version of torch for your hardware.
    - Any reward logs, weights, or other training data are by default set to be ignored by git
  - After running train_grpo.py, you can display a plot of the logged reward over time with 
  `.venv/bin/python3 show_reward_plot.py reward_log.jsonl`
  - To run any of the 3 tests for grpo.py use `.venv/bin/python3 [test file name]`


### Dependencies
  - For `train_grpo.py` (installation outlined below)
    - PyTorch
    - NumPy
    - Transformers
    - Datasets
    - Accelerate
    - Install deps. with `.venv/bin/pip install numpy transformers datasets accelerate`
      - PyTorch installation outlined below
  - For `show_reward_plot.py`
    - Pandas
    - Matplotlib
      - You may also need to install a GUI backend, PyQt6 for example, if you run into issues
    - Install deps. with `.venv/bin/pip install pandas matplotlib`
  - For `test_grpo.py` and `test_grpo3.py`
    - All deps. for `train_grpo.py`
    - TRL
    - Install TRL with `.venv/bin/pip install trl`
  - For `test_grpo2.py`
    - All deps. for `train_grpo.py` - Datasets

#### PyTorch Installation
Install one of the following versions of torch:

With CUDA (For Nvidia GPUs):
`.venv/bin/pip install torch`

With ROCm (For AMD GPUs):
`.venv/bin/pip install torch --index-url https://https://download.pytorch.org/whl/rocm6.3`

For CPU:
`.venv/bin/pip install torch --index-url https://https://download.pytorch.org/whl/cpu`
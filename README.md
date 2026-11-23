# Purdue CS173 Final Project
## Group CS5


### Usage
  - Create a venv in the repo via `python3 -m venv .venv`
  - Install the necessary dependencies as shown below in Dependencies.
  - Train a model on `letter_counting_dataset.csv` with `.venv/bin/python3 train_grpo.py --csv letter_counting_dataset.csv`
    - Depending on your hardware and whether you want to run on CPU or GPU or a combination, you may need to adjust 
    certain environment variables, such as CUDA_VISIBLE_DEVICES for example. You also need to install the correct version of torch for your hardware.
  - After running train_grpo.py, you can display a plot of the logged reward over time with 
  `venv/bin/python3 show_reward_plot.py reward_log.jsonl`


### Dependencies
  - `train_grpo.py`
    - PyTorch
    - NumPy
    - Transformers
  - `show_reward_plot.py`
    - pandas
    - matplotlib
      - You may also need to install a GUI backend, PyQt6 for example, if you run into issues

To run `train_grpo.py`, install the dependencies within a via:
`.venv/bin/pip install numpy transformers`

Then, to install torch with CUDA (For nVIDIA GPUs):
`.venv/bin/pip install torch`

Or, to install torch with ROCm (For AMD GPUs):
`.venv/bin/pip install torch --index-url https://https://download.pytorch.org/whl/rocm6.3`

Or, to install torch for CPU:
`.venv/bin/pip install torch --index-url https://https://download.pytorch.org/whl/cpu`
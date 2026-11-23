import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

"""
        Usage: 
            python show_reward_plot.py [reward log file]
"""

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Plot GRPO reward curve from a JSONL log file.")
    
    # Add the required positional argument for the file path
    parser.add_argument("log_file", help="Path to the .jsonl reward log file")

    args = parser.parse_args()

    # Check if the file exists
    if not os.path.exists(args.log_file):
        print(f"Error: The file '{args.log_file}' was not found.")
        sys.exit(1)

    # Load the data
    df = pd.read_json(args.log_file, lines=True)

    # Plotting logic
    plt.plot(df["step"], df["reward"], marker="o", markersize=3)
    plt.xlabel("Optimizer step")
    plt.ylabel("Mean reward")
    plt.title(f"GRPO reward over training\n({args.log_file})")
    plt.tight_layout()

if __name__ == "__main__":
    main()
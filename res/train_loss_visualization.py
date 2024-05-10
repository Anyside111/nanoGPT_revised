import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Directory and experiment setup
base_dir = "D:/289L/hw1/nanoGPT_revised/res"
experiment_folders = ["2.1", "2.2", "2.3", "3.1", "3.2", "3.3", "5.1", "5.2", "6.1", "7.1"]
experiment_titles = {
    "2.1": "Key Query Dim = 64 (Baseline)",
    "2.2": "Key Query Dim = 32",
    "2.3": "Key Query Dim = 8",
    "3.1": "Window Size = 100, n_regist = 0",
    "3.2": "Window Size = 10, n_regist = 0",
    "3.3": "Window Size = 3, n_regist = 0",
    "5.1": "Window Size = None, n_regist = 1",
    "5.2": "Window Size = None, n_regist = 5",
    "6.1": "Combined: Window Size = 3 and n_regist = 1",
    "7.1": "Softmax Abs = True" # vs. Baseline
}

def load_and_prepare_data(folder_name):
    file_path = os.path.join(base_dir, folder_name, "losses.csv")
    df = pd.read_csv(file_path, header=None, names=['Iteration', 'Train Loss', 'Validation Loss'])
    df['Validation Loss'] = pd.to_numeric(df['Validation Loss'], errors='coerce')  # Convert to numeric, make errors NaN
    return df

def plot_data(df, folder_name, filter_val_loss_empty=True):
    if filter_val_loss_empty:
        filtered_df = df[df['Validation Loss'].isna()].drop_duplicates(subset='Iteration')
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['Iteration'], filtered_df['Train Loss'], label='Train Loss')
        plt.title(f'Train Loss for Experiment {folder_name}: {experiment_titles[folder_name]}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        valid_df = df.dropna(subset=['Validation Loss'])
        plt.figure(figsize=(10, 6))
        plt.plot(valid_df['Iteration'], valid_df['Train Loss'], label='Train Loss')
        plt.plot(valid_df['Iteration'], valid_df['Validation Loss'], label='Validation Loss')
        plt.title(f'Train and Validation Loss for Experiment {folder_name}: {experiment_titles[folder_name]}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

# Process and plot data for each experiment
# for exp in experiment_folders:
    # df = load_and_prepare_data(exp)
    # plot_data(df, exp, filter_val_loss_empty=True)
    # plot_data(df, exp, filter_val_loss_empty=False)


groupings = {
    'Different key_query_dim Comparison': ["2.1", "2.2", "2.3"],
    'Different Window Size Comparison': ["3.1", "3.2", "3.3"],
    'Different n_regist Comparison': ["5.1", "5.2", "2.1"],
    'Combined Sliding Window Attention and Register Token vs n_regist=1': ["6.1", "5.1"],
    'Combined Sliding Window Attention and Register Token vs wind=3': ["6.1", "3.3"],
    'Exp Softmax vs Abs Softmax': ["7.1", "2.1"]
}

def plot_train_loss(group_name, experiments):

    plt.figure(figsize=(10, 5))
    markers = itertools.cycle(('+', 'o', '*', '.', 'x', 's', 'd'))
    for exp in experiments:
        df = load_and_prepare_data(exp)
        combined_df = df.dropna(subset=['Validation Loss'])  # Only keep rows where validation loss is recorded
        plt.plot(combined_df['Iteration'], combined_df['Train Loss'], label=f'{experiment_titles[exp]}',
                 marker=next(markers), linestyle='-', linewidth=1, markersize=4)
        # nth = 5
        # plt.plot(df['Iteration'][::nth], df['Train Loss'][::nth], label=f'{experiment_titles[exp]}',
        #          marker=next(markers), linestyle='-', linewidth=1, markersize=4)
    plt.title(f'{group_name} - Train Loss Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_validation_loss(group_name, experiments):
    plt.figure(figsize=(10, 5))
    markers = itertools.cycle(('+', 'o', '*', '.', 'x', 's', 'd'))
    for exp in experiments:
        df = load_and_prepare_data(exp)
        valid_df = df.dropna(subset=['Validation Loss'])
        plt.plot(valid_df['Iteration'], valid_df['Validation Loss'], label=f'{experiment_titles[exp]}',
                 marker=next(markers), linestyle='--', linewidth=1, markersize=4)
    plt.title(f'{group_name} - Validation Loss Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for each group
for group_name, experiments in groupings.items():
    plot_train_loss(group_name, experiments)
    plot_validation_loss(group_name, experiments)
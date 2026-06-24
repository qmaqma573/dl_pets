import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys


def analyze_encrypted_model(folder_path, model_type):
    """
    Analyze encrypted model results (CE or MSE) which are split across multiple epoch files
    """
    # Find all history files for this model
    files = sorted(
        glob(os.path.join(folder_path, "he_training*history*.json")))

    if not files:
        print(f"No files found in {folder_path}")
        return None, None, None

    # Extract epoch numbers from filenames
    epochs = []
    accuracies = []
    all_epoch_time = []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract epoch number from filename
            epoch = int(file_path.split('history_')[-1].split('.json')[0])
            epochs.append(epoch)

            # Extract validation accuracy
            if 'val_accuracy' in data:
                accuracies.append(data['val_accuracy'])

            # Extract batch times
            if 'batch_times' in data:
                all_epoch_time.extend(data['batch_times'])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Sort results by epoch
    sorted_indices = np.argsort(epochs)
    epochs = [epochs[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    # Calculate batch time statistics
    avg_batch_time = np.mean(all_epoch_time) if all_epoch_time else 0
    sum_batch_time = np.sum(all_epoch_time) if all_epoch_time else 0

    print(f"\n{model_type} Encrypted Model Results:")
    print(f"Average batch time: {avg_batch_time:.4f} seconds")
    print(f"Total batch time: {sum_batch_time:.4f} seconds")

    return epochs, accuracies, {'avg': avg_batch_time, 'sum': sum_batch_time}

def analyze_plaintext_model(file_path, model_type, epoch=20):
    """
    Analyze plaintext model results (CE or MSE) which are in a single file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract validation accuracies for all epochs
        epochs = list(range(1, len(data.get('val_accuracy', [])) + 1))

        accuracies = data.get('val_accuracy', [])[:epoch]

        # Extract batch times
        all_epoch_time = data.get('batch_times', [])

        times_batch = int(len(all_epoch_time) / len(epochs) * epoch)

        all_epoch_time = all_epoch_time[:times_batch]
        epochs = epochs[:epoch]

        # Calculate batch time statistics
        avg_batch_time = np.mean(all_epoch_time) if all_epoch_time else 0
        sum_batch_time = np.sum(all_epoch_time) if all_epoch_time else 0

        print(f"\n{model_type} Plaintext Model Results:")
        print(f"Average batch time: {avg_batch_time:.4f} seconds")
        print(f"Total batch time: {sum_batch_time:.4f} seconds")

        return epochs, accuracies, {'avg': avg_batch_time, 'sum': sum_batch_time}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def analyze_mpc_model(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Extract validation accuracies for all epochs
    epochs = list(range(1, len(data.get('accuracy', [])) + 1))

    accuracies = data.get('accuracy', [])

    # Extract batch times
    all_epoch_time = data.get('epoch_times', [])

    # Calculate batch time statistics
    avg_epoch_time = np.mean(all_epoch_time) if all_epoch_time else 0
    sum_epoch_time = np.sum(all_epoch_time) if all_epoch_time else 0

    print(f"Average epoch time: {avg_epoch_time:.4f} seconds")
    print(f"Total epoch time: {sum_epoch_time:.4f} seconds")

    return epochs, accuracies, {'avg': avg_epoch_time, 'sum': sum_epoch_time}


def plot_accuracy_curves(results, save_path='accuracy_comparison.png', title="Validation Accuracy Comparison"):
    """
    Plot accuracy curves for all models
    """
    markers      = ['o',  'X',   's',   '^']
    marker_sizes = [7,    12,    7,     10]
    linestyles   = ['-',  '--',  '-.',  ':']
    linewidths   = [2.0,  2.0,   2.0,   2.0]

    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(12, 8))

    for i, (model_name, (epochs, accuracies)) in enumerate(results.items()):
        if epochs and accuracies:
            idx = i % len(markers)
            plt.plot(
                epochs, accuracies,
                marker=markers[idx],
                markersize=marker_sizes[idx],
                linestyle=linestyles[idx],
                linewidth=linewidths[idx],
                label=model_name,
                zorder=len(results) - i,
            )

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Set x-axis to show integer epoch numbers
    plt.xticks(range(1, 21))

    plt.tight_layout()
    plt.savefig(save_path)

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Define paths to model results
    base_dir = "data/models/"

    ce_enc_dir = os.path.join(base_dir, 'he-ce')
    ce_plaintext_file = os.path.join(
        base_dir, 'plaintext-ce', 'training_history.json')

    mse_enc_dir = os.path.join(base_dir, 'he-mse')
    mse_plaintext_file = os.path.join(
        base_dir, 'plaintext-mse', 'training_history.json')

    # Analyze each model
    ce_enc_epochs, ce_enc_accuracies, ce_enc_batch_stats = analyze_encrypted_model(
        ce_enc_dir, "CE")
    mse_enc_epochs, mse_enc_accuracies, mse_enc_batch_stats = analyze_encrypted_model(
        mse_enc_dir, "MSE")

    ce_plain_epochs, ce_plain_accuracies, ce_plain_batch_stats = analyze_plaintext_model(
        ce_plaintext_file, "CE")
    mse_plain_epochs, mse_plain_accuracies, mse_plain_batch_stats = analyze_plaintext_model(
        mse_plaintext_file, "MSE")

    # Compile results for plotting
    accuracy_results = {
        'Semi-private training (CE)': (ce_enc_epochs, ce_enc_accuracies),
        'Semi-private training (MSE)': (mse_enc_epochs, mse_enc_accuracies),
        'Plaintext (CE)': (ce_plain_epochs[:20], ce_plain_accuracies[:20]),
        'Plaintext (MSE)': (mse_plain_epochs[:20], mse_plain_accuracies[:20])
    }

    # Plot accuracy curves
    plot_accuracy_curves(accuracy_results, "accuracy_HE.png", "Validation Accuracy Comparison (HE)")

    print("\n===== Batch Time Summary (HE) =====")
    print("Model               | Average (s) | Total (s)")
    print("-" * 50)
    print(
        f"CE Encrypted        | {ce_enc_batch_stats['avg']:.4f} | {ce_enc_batch_stats['sum']:.4f}")
    print(
        f"MSE Encrypted       | {mse_enc_batch_stats['avg']:.4f} | {mse_enc_batch_stats['sum']:.4f}")
    print(
        f"CE Plaintext        | {ce_plain_batch_stats['avg']:.4f} | {ce_plain_batch_stats['sum']:.4f}")
    print(
        f"MSE Plaintext       | {mse_plain_batch_stats['avg']:.4f} | {mse_plain_batch_stats['sum']:.4f}")

    # MPC vs. plaintext
    ce_mpc_file = os.path.join(
        base_dir, 'mpc', 'mpc_ce.json')
    mse_mpc_file = os.path.join(base_dir, 'mpc', 'mpc_mse.json')
    plaintext_file_ce = os.path.join(base_dir, 'mpc', 'plaintext_training_history_ce.json')
    plaintext_file_mse = os.path.join(base_dir, 'mpc', 'plaintext_training_history_mse.json')

    ce_epochs, ce_accuracies, ce_epoch_stats = analyze_mpc_model(
        ce_mpc_file)
    mse_epochs, mse_accuracies, mse_epoch_stats = analyze_mpc_model(
        mse_mpc_file)
    ce_plain_epochs, ce_plain_accuracies, ce_plain_epoch_stats = analyze_mpc_model(
        plaintext_file_ce)
    mse_plain_epochs, mse_plain_accuracies, mse_plain_batch_stats = analyze_mpc_model(
        plaintext_file_mse)

    mpc_plaintext_result = {
        'Fully private (CE)': (ce_epochs, ce_accuracies),
        'Fully private (MSE)': (mse_epochs, mse_accuracies),
        'Plaintext (CE)': (ce_plain_epochs, ce_plain_accuracies),
        'Plaintext (MSE)': (mse_plain_epochs, mse_plain_accuracies)
    }

    plot_accuracy_curves(mpc_plaintext_result, "accuracy_mpc.png", "Validation Accuracy Comparison (MPC)")

    # MPC fully encrypted vs. plaintext label
    plainlabel_ce_mpc = os.path.join(base_dir, 'mpc', 'mpc_plainlabel_ce.json')
    plainlabel_mse_mpc = os.path.join(base_dir, 'mpc', 'mpc_plainlabel_mse.json')

    ce_plainlabel_epochs, ce_plainlabel_accuracies, ce_plainlabel_epoch_stats = analyze_mpc_model(
        plainlabel_ce_mpc)
    mse_plainlabel_epochs, mse_plainlabel_accuracies, mse_plainlabel_batch_stats = analyze_mpc_model(
        plainlabel_mse_mpc)

    mpc_label_result = {
        'Fully private training': (ce_epochs, ce_accuracies),
        # 'MSE (Fully Enc.)': (mse_epochs, mse_accuracies),
        'Semi-private training': (ce_plainlabel_epochs, ce_plainlabel_accuracies),
        # 'MSE (Enc. Forward, Plain Labels)': (mse_plainlabel_epochs, mse_plainlabel_accuracies)
    }

    plot_accuracy_curves(mpc_label_result, 'encrypted_label_comp.png', "Validation Accuracy Comparison (MPC)")

    print("\n===== Batch Time Summary (MPC) =====")
    print("Model               | Average (s) | Total (s)")
    print("-" * 50)
    print(
        f"MPC CE        | {ce_epoch_stats['avg']:.4f} | {ce_epoch_stats['sum']:.4f}")
    print(
        f"MSE Encrypted       | {mse_epoch_stats['avg']:.4f} | {mse_epoch_stats['sum']:.4f}")
    print(
        f"MPC plaintext label CE        | {ce_plainlabel_epoch_stats['avg']:.4f} | {ce_plainlabel_epoch_stats['sum']:.4f}")
    print(
        f"MPC plaintext label MSE       | {mse_plainlabel_batch_stats['avg']:.4f} | {mse_plainlabel_batch_stats['sum']:.4f}")
    print(
        f"CE Plaintext        | {ce_plain_epoch_stats['avg']:.4f} | {ce_plain_epoch_stats['sum']:.4f}")
    print(
        f"MSE Plaintext       | {mse_plain_batch_stats['avg']:.4f} | {mse_plain_batch_stats['sum']:.4f}")

    plaintext_results_comp  = {
        'MPC Baseline (CE)': (ce_plain_epochs, ce_plain_accuracies),
        'MPC Baseline (MSE)': (mse_plain_epochs, mse_plain_accuracies),
        'HE Baseline (CE)': accuracy_results['Plaintext (CE)'],
        'HE Baseline (MSE)': accuracy_results['Plaintext (MSE)']
    }

    plot_accuracy_curves(plaintext_results_comp, 'baseline_comp.png')
    mpc_he_comp = {
        'MPC (MSE)': (mse_plainlabel_epochs, mse_plainlabel_accuracies),
        'MPC (CE)': (ce_plainlabel_epochs, ce_plainlabel_accuracies),
        'HE (MSE)': (mse_enc_epochs, mse_enc_accuracies),
        'HE (CE)': (ce_enc_epochs, ce_enc_accuracies)
    }
    plot_accuracy_curves(mpc_he_comp, 'he_mpc_comp.png')

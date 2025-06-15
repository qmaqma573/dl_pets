#!/usr/bin/env python3
import os
import random
import time
import json
import numpy as np
import crypten
import crypten.communicator as comm
from crypten.config import cfg
import crypten.optim as crypten_optim
import torch


parent_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..'))
encrypted_label = False
cfg.communicator.verbose = True

def run_benchmarks(
    epochs=20,
    start_epoch=0,
    batch_size=64,
    lr=0.1,
    momentum=0.99,
    loss_func="ce",
    seed=42,
):
    aws = True
    print(encrypted_label)

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    crypten.init()

    rank = comm.get().get_rank()

    ALICE = 0
    BOB = 1

    # load data from two parties
    party_path = f'party{rank}' if aws == False else f'../party{rank}'
    x_path_train = os.path.join(party_path, 'X_train.pt')
    x_path_val = os.path.join(party_path, 'X_val.pt')
    y_path_train = os.path.join(party_path, 'y_train.pt')
    y_path_val = os.path.join(party_path, 'y_val.pt')

    x_train_alice = crypten.load_from_party(x_path_train, src=ALICE)
    x_train_bob = crypten.load_from_party(x_path_train, src=BOB)
    x_val_alice = crypten.load_from_party(x_path_val, src=ALICE)
    x_val_bob = crypten.load_from_party(x_path_val, src=BOB)

    y_train = crypten.load_from_party(y_path_train, src=ALICE)
    y_val = crypten.load_from_party(y_path_val, src=ALICE)

    y_train = y_train.get_plain_text()
    y_val = y_val.get_plain_text()

    unique_labels = np.unique(y_train)
    n_classes = len(unique_labels)

    # If labels don't start at 0 or have gaps, remap them
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[int(label)] for label in y_train])
    y_val_mapped = np.array([label_map[int(label)] for label in y_val])

    y_train = torch.from_numpy(y_train_mapped).long()
    y_val = torch.from_numpy(y_val_mapped).long()

    label_eye = torch.eye(n_classes)
    labels = y_train.long()
    labels_one_hot = label_eye[labels]

    label_eye = torch.eye(n_classes)
    val_labels = y_val.long()
    val_labels_one_hot = label_eye[val_labels]

    x_combined_enc = crypten.cat([x_train_alice, x_train_bob], dim=1)
    x_combined_enc_val = crypten.cat([x_val_alice, x_val_bob], dim=1)

    if encrypted_label:
        y_train = crypten.cryptensor(labels_one_hot)
        y_val = crypten.cryptensor(val_labels_one_hot)
    else:
        y_train = labels_one_hot
        y_val = val_labels_one_hot

    n_features = x_combined_enc.shape[1]

    # create model
    pytorch_model = TorchFashionClassifier(n_features, n_classes)
    crypten.print(f"Optimizer parameter: {lr}, {momentum}")

    # encrypt model
    model = create_model(pytorch_model, n_features)

    optimizer = crypten_optim.SGD(
        model.parameters(), lr, momentum=momentum
    )

    # define loss function (criterion) and optimizer
    crypten.print(f"loss function: {loss_func}")
    if loss_func == 'ce':
        criterion = crypten.nn.CrossEntropyLoss()
    else:
        criterion = crypten.nn.MSELoss()

    train_loss = []
    val_losses = []
    val_accuracy = []
    epoch_times = []

    if aws and rank == 0:
        print("start training")

    for epoch in range(start_epoch, epochs):
        # train for one epoch
        loss, epoch_time = train(
            model,
            x_combined_enc,
            y_train,
            criterion,
            optimizer,
            batch_size,
            verbose=False
        )

        train_loss.append(loss)
        epoch_times.append(epoch_time)

        # evaluate on validation set
        val_loss, accuracy = validate(model, x_combined_enc_val, y_val, criterion, epoch)
        val_losses.append(val_loss)
        val_accuracy.append(accuracy)
        crypten.print(f"Epoch {epoch + 1} --- Training loss: {loss}, epoch time: {epoch_time}, val loss: {val_loss}, val accuracy: {accuracy}")

    crypten.print_communication_stats()

    if rank == 0:
        history = {
            "train_loss": train_loss,
            "val_loss": val_losses,
            "accuracy": val_accuracy,
            "epoch_times": epoch_times
        }

        result_dir = os.path.join(parent_dir, 'data', 'models', 'mpc') if aws == False else 'results'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with open(os.path.join(result_dir, f'mpc_training_history_{loss_func}.json'), 'w') as f:
            crypten.print(f"Write to result file: {os.path.join(result_dir, f'mpc_training_history_{loss_func}.json')}")
            json.dump(history, f)


def preprocess_data():
    # crypten.print("Loading training data...")
    # train_data = np.load(os.path.join(parent_dir, 'data',
    #                      'extracted_features', 'train_features.npz'))
    # X_train, y_train = train_data['X'], train_data['y']

    # crypten.print("Loading validation data...")
    val_data = np.load(os.path.join(parent_dir, 'data',
                       'extracted_features', 'val_features.npz'))
    X_val, y_val = val_data['X'], val_data['y']

    unique_labels = np.unique(y_train)

    # If labels don't start at 0 or have gaps, remap them
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[label] for label in y_train])
    y_val_mapped = np.array([label_map[label] for label in y_val])

    n_classes = len(unique_labels)
    n_features = X_train.shape[1]

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train_mapped).long()

    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val_mapped).long()

    label_eye = torch.eye(n_classes)
    labels = y_train.long()
    labels_one_hot = label_eye[labels]

    label_eye = torch.eye(n_classes)
    val_labels = y_val.long()
    val_labels_one_hot = label_eye[val_labels]

    features_alice = X_train[:10000]
    features_bob = X_train[10000:]

    features_alice_val = X_val[:1991]
    features_bob_val = X_val[1991:]

    features = {
        "features_alice": features_alice,
        "features_bob": features_bob,
        "features_alice_val": features_alice_val,
        "features_bob_val": features_bob_val
    }

    return n_features, n_classes, features, labels_one_hot, val_labels_one_hot


def create_model(pytorch_model, n_features):
    dummy_input = torch.empty(1, n_features)
    model = crypten.nn.from_pytorch(pytorch_model, dummy_input)
    model.encrypt()
    return model


def train(model, x_combined_enc, y_train, criterion, optimizer, batch_size, verbose=False):
    crypten.print(
        f"batch size: {batch_size}, training sample size: {x_combined_enc.size(0)}")
    num_batches = x_combined_enc.size(0) // batch_size

    model.train()
    batch_losses = []
    start_time = time.time()
    for batch in range(num_batches):
        start, end = batch * batch_size, (batch + 1) * batch_size
        x_train = x_combined_enc[start:end]
        y_batch = y_train[start:end]

        optimizer.zero_grad()

        output = model(x_train)

        loss_value = criterion(output, y_batch)

        loss_value.backward()

        optimizer.step()
        batch_loss = loss_value.get_plain_text()
        batch_losses.append(batch_loss.item())
        if verbose:
            crypten.print(f"\tBatch {(batch + 1)} of {num_batches} Loss {batch_loss.item():.4f}")

    epoch_time = time.time() - start_time
    return sum(batch_losses) / num_batches, epoch_time


def validate(model, x_combined_enc_val, y_val, criterion, epoch):
    rank = comm.get().get_rank()
    avg_val_loss = -1
    val_accuracy = -1

    val_batch_size = 64  # Or another manageable size, e.g., 128, 256
    num_val_batches = (x_combined_enc_val.size(
        0) + val_batch_size - 1) // val_batch_size  # Handles partial last batch

    batch_val_losses = []
    all_val_outputs_decrypted_list = []
    all_y_val_decrypted_list = []
    crypten.print("start validation")
    with crypten.no_grad():
        for val_batch_idx in range(num_val_batches):
            val_start = val_batch_idx * val_batch_size
            val_end = min((val_batch_idx + 1) * val_batch_size,
                          x_combined_enc_val.size(0))

            if val_start >= val_end:
                continue

            x_val_batch = x_combined_enc_val[val_start:val_end]
            # y_val is the full one-hot encrypted labels; slice it for the batch
            y_val_batch_one_hot = y_val[val_start:val_end]

            val_outputs_batch_enc = model(x_val_batch)
            val_loss_batch_enc = criterion(
                val_outputs_batch_enc, y_val_batch_one_hot)

            batch_val_losses.append(
                val_loss_batch_enc.get_plain_text().item())
            # For overall accuracy calculation later
            all_val_outputs_decrypted_list.append(
                val_outputs_batch_enc.get_plain_text())
            if encrypted_label:
                all_y_val_decrypted_list.append(
                    y_val_batch_one_hot.get_plain_text())
            else:
                all_y_val_decrypted_list.append(y_val_batch_one_hot)
    if batch_val_losses:
        # Filter NaNs
        valid_losses = [l for l in batch_val_losses if l == l]
        avg_val_loss = sum(valid_losses) / \
            len(valid_losses) if valid_losses else float('nan')
        crypten.print(
            f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

    if all_val_outputs_decrypted_list and all_y_val_decrypted_list:
        try:
            # Concatenate results from all validation batches
            final_val_outputs_decrypted = torch.cat(
                all_val_outputs_decrypted_list, dim=0)
            final_y_val_decrypted = torch.cat(
                all_y_val_decrypted_list, dim=0)

            val_predicted = torch.argmax(
                final_val_outputs_decrypted, dim=1)
            val_true_labels = torch.argmax(final_y_val_decrypted, dim=1)

            correct_predictions = (
                val_predicted == val_true_labels).sum().item()
            total_validation_samples = val_true_labels.numel()
            val_accuracy = 0
            if total_validation_samples > 0:
                val_accuracy = correct_predictions / total_validation_samples

            crypten.print(
                f"Epoch {epoch+1} Validation Accuracy: {val_accuracy:.4f}")
        except Exception as e_acc:
            crypten.print(
                f"Error during final validation accuracy calculation: {e_acc}")
    return avg_val_loss, val_accuracy


class TorchFashionClassifier(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super(TorchFashionClassifier, self).__init__()
        self.classifier = torch.nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.classifier(x)

# run_benchmarks()

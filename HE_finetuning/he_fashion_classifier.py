import numpy as np
import tenseal as ts
import torch
import time
from config import learning_rate, batch_size, epoches
from abc import ABC, abstractmethod


class HEFashionClassifier(ABC):
    def __init__(self, n_features, n_classes):
        self.context = self._create_context()

        self.n_features = n_features
        self.n_classes = n_classes

        std = np.sqrt(2.0 / n_features)

        self.weights = np.random.normal(0, std, (n_features, n_classes))
        self.bias = np.zeros(n_classes)

        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def _create_context(self):
        """Create a TenSEAL context with appropriate parameters for our task."""
        # Higher gives more precision but slower
        poly_mod_degree = 8192
        # levels for deeper computation
        coeff_mod_bit_sizes = [50, 40, 50]

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )

        # higher gives more precision
        context.global_scale = 2**40
        context.generate_galois_keys()

        return context

    def encrypt_data(self, data):
        data_list = data.tolist() if isinstance(data, np.ndarray) else data
        encrypted_data = ts.ckks_vector(self.context, data_list)

        return encrypted_data

    def encrypt_one_hot(self, label):
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1.0
        # TODO: whether to entrypt this
        return self.encrypt_data(one_hot)

    def forward(self, encrypted_x):
        # After dot operation, the output(label) can be decrypted
        outputs = []

        for class_idx in range(self.n_classes):
            class_weights = self.weights[:, class_idx].tolist()
            result = encrypted_x.dot(class_weights)
            result = result + self.bias[class_idx]

            outputs.extend(result.decrypt())

        return np.array(outputs)

    def forward_plaintext(self, x):
        return np.dot(x, self.weights) + self.bias

    def adam_update(self, weight_grads, bias_grads, learning_rate):
        self.t += 1

        # Update weight parameters
        self.m_weights = self.beta1 * self.m_weights + \
            (1 - self.beta1) * weight_grads
        self.v_weights = self.beta2 * self.v_weights + \
            (1 - self.beta2) * (weight_grads ** 2)
        m_weights_hat = self.m_weights / (1 - self.beta1 ** self.t)
        v_weights_hat = self.v_weights / (1 - self.beta2 ** self.t)
        self.weights -= learning_rate * m_weights_hat / \
            (np.sqrt(v_weights_hat) + self.epsilon)

        # Update bias parameters
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * bias_grads
        self.v_bias = self.beta2 * self.v_bias + \
            (1 - self.beta2) * (bias_grads ** 2)
        m_bias_hat = self.m_bias / (1 - self.beta1 ** self.t)
        v_bias_hat = self.v_bias / (1 - self.beta2 ** self.t)
        self.bias -= learning_rate * m_bias_hat / \
            (np.sqrt(v_bias_hat) + self.epsilon)

    @abstractmethod
    def compute_loss(self, outputs, label):
        pass

    @abstractmethod
    def compute_gradients(self, encrypted_x, outputs, lable):
        pass

    def train(self, X_train, y_train, epochs=epoches, learning_rate=learning_rate, batch_size=batch_size, verbose=False):
        X_train = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
        y_train = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train

        n_samples = len(X_train)
        history = {'train_loss': [], 'val_loss': [],
                   'val_accuracy': [], 'batch_times': []}

        print(
            f"Training set size: {n_samples} samples with {X_train.shape[1]} features")
        print(f"Batch size: {batch_size}")
        total_start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)

            n_batches = (n_samples + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                batch_start = time.time()

                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                if start_idx >= end_idx:
                    continue
                batch_indices = indices[start_idx:end_idx]
                actual_batch_size = len(batch_indices)

                if verbose:
                    print(
                        f"  Processing batch {batch_idx+1}/{n_batches} (samples {start_idx+1}-{end_idx})...")

                # Get batch data
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                # Initialize gradient accumulators
                weight_grads_acc = np.zeros_like(self.weights)
                bias_grads_acc = np.zeros_like(self.bias)
                batch_loss = 0.0

                # Process each sample in the batch
                for i in range(actual_batch_size):
                    x = batch_X[i]
                    y = batch_y[i]

                    encrypted_x = self.encrypt_data(x)

                    # Forward pass
                    outputs = self.forward(encrypted_x)

                    # Compute loss
                    loss_value = self.compute_loss(
                        outputs, y)

                    batch_loss += loss_value

                    # Compute gradients
                    weight_grads, bias_grads = self.compute_gradients(
                        encrypted_x, outputs, y)

                    # Accumulate gradients
                    weight_grads_acc += weight_grads
                    bias_grads_acc += bias_grads

                # Average and apply gradients once per batch
                weight_grads_acc /= actual_batch_size
                bias_grads_acc /= actual_batch_size

                self.adam_update(weight_grads_acc,
                                 bias_grads_acc, learning_rate)

                avg_batch_loss = batch_loss / actual_batch_size
                epoch_loss += batch_loss

                # Record batch timing
                batch_time = time.time() - batch_start
                history['batch_times'].append(batch_time)

                # Print batch summary
                if verbose and batch_idx % 5 == 0:
                    print(
                        f"  Batch {batch_idx+1}/{n_batches} completed in {batch_time:.2f}s, loss: {avg_batch_loss:.6f}")

            # Epoch summary
            avg_epoch_loss = epoch_loss / n_samples
            history['train_loss'].append(avg_epoch_loss)
            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Loss: {avg_epoch_loss:.6f}")

            print(
                f"  Time: {epoch_time:.2f}s ({n_samples/epoch_time:.2f} samples/sec)")

        total_time = time.time() - total_start_time
        print(
            f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")

        return history

    def validate(self, X_val, y_val):
        # Validation phase will spend extra time, and it won't effect the weights and bias during the training stage, so skip it to save time
        X_val = X_val.numpy() if isinstance(X_val, torch.Tensor) else X_val
        y_val = y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for i in range(len(X_val)):
            x = X_val[i]
            encrypted_x = self.encrypt_data(x)
            y = y_val[i]

            # Forward pass
            outputs = self.forward(encrypted_x)

            # Compute validation loss
            loss_value = self.compute_loss(outputs, y)
            val_loss += loss_value

            # Compute accuracy
            predicted = np.argmax(outputs)
            val_total += 1
            if predicted == y:
                val_correct += 1

        # Calculate validation metrics
        avg_val_loss = val_loss / len(X_val)
        val_accuracy = val_correct / val_total

        print(
            f"  Val Loss: {avg_val_loss}, Val accuracy: {val_accuracy}")
        return avg_val_loss, val_accuracy

    def evaluate(self, X_test, y_test, verbose=False):
        """Evaluate model on test data(plaintext)"""
        X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_test = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

        n_samples = len(X_test)
        predictions = np.zeros(n_samples, dtype=int)

        print(f"Starting evaluation on {n_samples} samples...")
        start_time = time.time()

        # Find the actual number of classes in the test data
        unique_labels = np.unique(y_test)
        max_label = int(max(unique_labels))
        actual_n_classes = max(self.n_classes, max_label + 1)

        # Track per-class accuracy with arrays large enough for all classes
        class_correct = np.zeros(actual_n_classes)
        class_total = np.zeros(actual_n_classes)

        # Process batches for faster evaluation
        batch_size = 100
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            print(
                f"Evaluating batch {batch_idx+1}/{n_batches} (samples {start_idx+1}-{end_idx})...", end="\r")

            for idx in range(start_idx, end_idx):
                x = X_test[idx]
                true_label = int(y_test[idx])

                # Skip invalid labels
                if true_label >= actual_n_classes:
                    print(
                        f"Warning: Found label {true_label} which is outside the expected range. Skipping.")
                    continue

                outputs = self.forward(x)

                # Get predicted class
                predicted_class = np.argmax(outputs)
                predictions[idx] = predicted_class

                # Update class-specific accuracy
                class_total[true_label] += 1
                if predicted_class == true_label:
                    class_correct[true_label] += 1

        # Compute overall accuracy
        accuracy = np.sum(class_correct) / np.sum(class_total)

        # Display evaluation results
        eval_time = time.time() - start_time
        print(f"\nEvaluation completed in {eval_time:.2f}s")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Print confusion matrix if scikit-learn is available
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, predictions)
            print("\nConfusion Matrix (first 10x10 section):")
            print(cm[:10, :10])  # Show just a portion for readability
        except:
            pass

        if verbose:
            print("\nPer-class accuracy (for classes with samples):")
            for i in range(actual_n_classes):
                if class_total[i] > 0:
                    class_acc = class_correct[i] / class_total[i]
                    print(
                        f"  Class {i}: {class_acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

        return accuracy


class MSE_HEFashionClassifier(HEFashionClassifier):
    def compute_loss(self, outputs, label):
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1.0

        diff = outputs - one_hot
        return np.sum(diff * diff) / self.n_classes

    def compute_gradients(self, encrypted_x, outputs, label):
        # will be slower, compared to the plaintext approach(np.outer), not just because the encryption operation
        weight_gradients = np.zeros_like(self.weights)
        bias_gradients = np.zeros_like(self.bias)

        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1.0

        for class_idx in range(self.n_classes):
            # Compute the output gradient for this class: 2 * (output - target) / n_classes
            output_val = outputs[class_idx]
            target_val = one_hot[class_idx]
            output_grad = 2 * (output_val - target_val) / self.n_classes

            bias_gradients[class_idx] = output_grad

            # For weight gradients: multiply each encrypted feature by the output gradient (scalar)
            encrypted_grad_vector = encrypted_x * output_grad

            decrypted_grad_vector = encrypted_grad_vector.decrypt()

            # Set the weight gradients for this class
            for feature_idx in range(self.n_features):
                weight_gradients[feature_idx,
                                 class_idx] = decrypted_grad_vector[feature_idx]

        return weight_gradients, bias_gradients


class CE_HEFashionClassifier(HEFashionClassifier):
    def softmax(self, x):
        shifted_x = x - np.max(x)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x)

    def compute_loss(self, outputs, label):
        # Since we keep the labels in plaintext, we can use softmax itself instead of approximation
        probabilities = self.softmax(outputs)
        return -np.log(probabilities[label] + self.epsilon)

    def compute_gradients(self, encrypted_x, outputs, label):
        """Compute gradients for cross entropy loss."""
        probabilities = self.softmax(outputs.flatten())
        weight_gradients = np.zeros_like(self.weights)

        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1.0

        output_grad = probabilities - one_hot
        bias_gradients = output_grad

        for class_idx in range(self.n_classes):
            # For weight gradients: multiply each encrypted feature by the output gradient (scalar)
            encrypted_grad_vector = encrypted_x * output_grad[class_idx]

            decrypted_grad_vector = encrypted_grad_vector.decrypt()

            # Set the weight gradients for this class
            for feature_idx in range(self.n_features):
                weight_gradients[feature_idx,
                                 class_idx] = decrypted_grad_vector[feature_idx]

        return weight_gradients, bias_gradients

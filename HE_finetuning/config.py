import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(
    os.path.dirname(current_dir), "data", "extracted_features")

model_path = os.path.join(os.path.dirname(current_dir), "data", "models")

learning_rate = 0.001
weight_decay = 1e-5
epoches = 20
batch_size = 64

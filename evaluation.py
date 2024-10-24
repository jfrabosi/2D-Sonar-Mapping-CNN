import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time

# First copy over the model architecture classes
class MicrophoneAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([channels])))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention, dim=-1)
        return torch.matmul(attention, V)


class SpatialAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels // reduction_ratio, 1)
        self.conv2 = nn.Conv1d(channels // reduction_ratio, channels, 1)
        self.spatial_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        context = F.adaptive_avg_pool1d(x, 1)
        context = self.conv1(context)
        context = F.relu(context)
        context = self.conv2(context)
        spatial = self.spatial_conv(x)
        attention = F.sigmoid(context + spatial)
        return x * attention


class ContinuousFieldOutput(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = 128

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )

        self.distance_pred = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )

        self.confidence_pred = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared_features = self.shared(x)
        distances = self.distance_pred(shared_features)
        confidence = self.confidence_pred(shared_features)
        return distances, confidence


class AcousticNN(nn.Module):
    def __init__(self, input_channels=7, sequence_length=3695, output_size=61):
        super().__init__()
        self.l2_lambda = 0.001

        self.feature_extraction = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1)
        )

        self.mic_attention = MicrophoneAttention(64)
        self.spatial_attention = SpatialAttention(64)
        self.attention_dropout = nn.Dropout(0.2)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.2),
            nn.Conv1d(96, 128, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.field_output = ContinuousFieldOutput(128, output_size)  # 128 = 2*64 (bidirectional)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.mic_attention(x)
        x = self.attention_dropout(x)
        x = self.spatial_attention(x)
        x = self.attention_dropout(x)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = F.dropout(x.mean(dim=1), p=0.3, training=self.training)
        distances, confidence = self.field_output(x)
        return distances, confidence


def scale_microphone(data, min_val=-200, max_val=200):
    return (data - min_val) / (max_val - min_val) * 2 - 1


def scale_laser(data, min_val=0, max_val=0.1):
    return np.where(data != 0, (data - min_val) / (max_val - min_val), 0)


def load_and_preprocess_data(filename, num_simulations, start_index=0):
    with h5py.File(filename, 'r') as f:
        # Calculate max_length from the specified range
        max_length = max(f[f"simulation_{i}"]["microphone_data"].shape[1]
                        for i in range(start_index, start_index + num_simulations))

        X, y_dist, y_mask = [], [], []
        for i in range(start_index, start_index + num_simulations):
            sim_group = f[f"simulation_{i}"]
            microphone_data = sim_group['microphone_data'][:]
            laser_distances = sim_group['laser_distances'][:]

            padded_data = np.zeros((microphone_data.shape[0], max_length))
            padded_data[:, :microphone_data.shape[1]] = microphone_data

            X.append(padded_data)
            dist = np.where(laser_distances == -1, 0, laser_distances)
            mask = (laser_distances != -1).astype(float)

            y_dist.append(dist)
            y_mask.append(mask)

    return np.array(X), np.array(y_dist), np.array(y_mask)


def visualize_predictions(model, X_val, y_dist_val, y_mask_val, sample_idx=None):
    model.eval()

    if sample_idx is None:
        sample_idx = np.random.randint(0, len(X_val))

    # Get predictions for the sample
    with torch.no_grad():
        input_sample = torch.FloatTensor(X_val[sample_idx:sample_idx + 1]).to(next(model.parameters()).device)
        pred_dist, pred_conf = model(input_sample)
        pred_dist = pred_dist[0].cpu().numpy()
        pred_conf = pred_conf[0].cpu().numpy()

    # Get actual values
    actual_dist = y_dist_val[sample_idx]
    actual_mask = y_mask_val[sample_idx]

    # Calculate dot products
    actual_dot_product = actual_dist * actual_mask
    predicted_dot_product = pred_dist * pred_conf

    # Create visualization
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'Model Predictions for Sample {sample_idx}')

    # Distance predictions
    plt.subplot(3, 1, 1)
    plt.plot(actual_dist, 'b-', label='Actual Distance', alpha=0.7)
    plt.plot(pred_dist, 'r--', label='Predicted Distance', alpha=0.7)
    plt.title('Distance Field Prediction vs Actual')
    plt.xlabel('Laser Index')
    plt.ylabel('Scaled Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confidence/Mask predictions
    plt.subplot(3, 1, 2)
    plt.plot(actual_mask, 'b-', label='Actual Mask', alpha=0.7)
    plt.plot(pred_conf, 'r--', label='Predicted Confidence', alpha=0.7)
    plt.title('Object Presence Prediction vs Actual')
    plt.xlabel('Laser Index')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Dot product visualization
    plt.subplot(3, 1, 3)
    plt.plot(actual_dot_product, 'b-', label='Actual Masked Distance', alpha=0.7)
    plt.plot(predicted_dot_product, 'r--', label='Predicted Masked Distance', alpha=0.7)
    plt.title('Masked Distance (Dot Product)')
    plt.xlabel('Laser Index')
    plt.ylabel('Masked Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate metrics
    valid_mask = actual_mask > 0
    if valid_mask.any():
        mse_distance = np.mean((actual_dist[valid_mask] - pred_dist[valid_mask]) ** 2)
        mae_distance = np.mean(np.abs(actual_dist[valid_mask] - pred_dist[valid_mask]))
        mse_masked = np.mean((actual_dot_product - predicted_dot_product) ** 2)
    else:
        mse_distance = mae_distance = mse_masked = 0

    bce_mask = -np.mean(actual_mask * np.log(pred_conf + 1e-10) +
                        (1 - actual_mask) * np.log(1 - pred_conf + 1e-10))

    print(f"\nMetrics for sample {sample_idx}:")
    print(f"Distance MSE: {mse_distance:.4f}")
    print(f"Distance MAE: {mae_distance:.4f}")
    print(f"Mask BCE: {bce_mask:.4f}")
    print(f"Masked Distance MSE: {mse_masked:.4f}")

    return sample_idx


def calculate_flops(model, input_shape):
    """
    Calculate FLOPs for the model using a dummy input of the given shape.
    Also measures inference time.
    """
    device = next(model.parameters()).device

    # Create dummy input with batch size of 2 to satisfy BatchNorm
    dummy_input = torch.randn(2, *input_shape).to(device)

    # Ensure model is in eval mode
    model.eval()

    # Measure inference time (using batch size 1 for realistic timing)
    real_input = torch.randn(1, *input_shape).to(device)

    # Warm-up run
    with torch.no_grad():
        _ = model(real_input)

    # Actual timing
    times = []
    num_runs = 100  # Increase this for more stable timing

    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(real_input)
            torch.cuda.synchronize()  # Ensure GPU computation is complete
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    inference_time = sum(times) / len(times)

    # Calculate FLOPs (using batch size 2 for proper BatchNorm operation)
    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False)  # Suppress warnings about unsupported ops
    total_flops = flops.total() // 2  # Divide by 2 since we used batch size 2

    # Get detailed analysis
    flops_table = flop_count_table(flops)

    print("\nModel Computation Analysis:")
    print(f"Total FLOPs per inference: {total_flops:,}")
    print(f"Average inference time: {inference_time:.2f} ms")
    print(f"Standard deviation: {np.std(times):.2f} ms")
    print("\nPerformance Metrics:")
    print(f"FLOPs per second: {(total_flops / (inference_time / 1000)):,.2f}")
    print(f"Theoretical max throughput: {1000 / inference_time:.2f} inferences/second")
    print("\nDetailed FLOPs breakdown:")
    print(flops_table)

    return total_flops, inference_time


def main(sim_start=7500, sim_end=7800):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print(f"Loading data from simulation {sim_start} to {sim_end}...")
    num_simulations = sim_end - sim_start
    X, y_dist, y_mask = load_and_preprocess_data(
        "simulation_data/simulation_data_parallel.h5",
        num_simulations=num_simulations,
        start_index=sim_start
    )

    # Scale the data
    X_scaled = scale_microphone(X)
    y_dist_scaled = scale_laser(y_dist)

    # Split the data
    X_train, X_val, y_dist_train, y_dist_val, y_mask_train, y_mask_val = train_test_split(
        X_scaled, y_dist_scaled, y_mask, test_size=0.2, random_state=42
    )

    # Load the model
    print("Loading model...")
    model = AcousticNN(
        input_channels=X_scaled.shape[1],
        sequence_length=X_scaled.shape[2],
        output_size=y_dist.shape[1]
    ).to(device)

    try:
        model.load_state_dict(torch.load('best_model.pth', weights_only=True))
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Ensure model is in eval mode
    model.eval()

    # Calculate FLOPs and inference time
    input_shape = (X_scaled.shape[1], X_scaled.shape[2])
    flops, inference_time = calculate_flops(model, input_shape)

    # Visualize predictions for a random sample
    sample_idx = visualize_predictions(model, X_val, y_dist_val, y_mask_val)

    # Allow the user to view more samples if desired
    while True:
        response = input("\nWould you like to see another sample? (y/n): ")
        if response.lower() != 'y':
            break
        sample_idx = visualize_predictions(model, X_val, y_dist_val, y_mask_val)

if __name__ == "__main__":
    main(sim_start=7500, sim_end=7800)
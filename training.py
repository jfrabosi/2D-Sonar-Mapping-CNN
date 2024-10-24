import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_and_preprocess_data(filename, num_simulations):
    with h5py.File(filename, 'r') as f:
        max_length = max(f[f"simulation_{i}"]["microphone_data"].shape[1] for i in range(num_simulations))

        X, y_dist, y_mask = [], [], []
        for i in range(num_simulations):
            sim_group = f[f"simulation_{i}"]
            microphone_data = sim_group['microphone_data'][:]
            laser_distances = sim_group['laser_distances'][:]

            padded_data = np.zeros((microphone_data.shape[0], max_length))
            padded_data[:, :microphone_data.shape[1]] = microphone_data

            X.append(padded_data)

            # Create distance and mask arrays
            dist = np.where(laser_distances == -1, 0, laser_distances)
            mask = (laser_distances != -1).astype(float)

            y_dist.append(dist)
            y_mask.append(mask)

    X = np.array(X)
    y_dist = np.array(y_dist)
    y_mask = np.array(y_mask)
    print(f"X shape: {X.shape}")
    print(f"y_dist shape: {y_dist.shape}")
    print(f"y_mask shape: {y_mask.shape}")
    return X, y_dist, y_mask


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
        self.spatial_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels
        )

    def forward(self, x):
        # Global context
        context = F.adaptive_avg_pool1d(x, 1)
        context = self.conv1(context)
        context = F.relu(context)
        context = self.conv2(context)

        # Local spatial relationship
        spatial = self.spatial_conv(x)

        # Combine global and local
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

        # Distance prediction with positive output
        self.distance_pred = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensures positive distances
        )

        # Confidence prediction (0 to 1)
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

        # Feature extraction
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

        # Attention mechanisms
        self.mic_attention = MicrophoneAttention(64)
        self.spatial_attention = SpatialAttention(64)
        self.attention_dropout = nn.Dropout(0.2)

        # Temporal processing
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

        # Bidirectional LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            128, 64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Continuous field output
        self.field_output = ContinuousFieldOutput(128, output_size)  # 128 = 2*64 (bidirectional)

    def forward(self, x):
        # Feature extraction
        x = self.feature_extraction(x)

        # Apply microphone attention
        x = self.mic_attention(x)
        x = self.attention_dropout(x)

        # Apply spatial attention
        x = self.spatial_attention(x)
        x = self.attention_dropout(x)

        # Temporal processing
        x = self.temporal_conv(x)

        # LSTM processing
        x = x.transpose(1, 2)  # (batch, time, channels)
        x, _ = self.lstm(x)

        # Global average pooling with dropout
        x = F.dropout(x.mean(dim=1), p=0.3, training=self.training)

        # Get continuous field output
        distances, confidence = self.field_output(x)

        return distances, confidence

    def get_l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_lambda * l2_loss

    def print_param_count(self):
        print("Parameter count per layer:")
        for name, module in self.named_children():
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name}: {param_count:,} parameters")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal trainable parameters: {total_params:,}")


class ContinuousFieldLoss(nn.Module):
    def __init__(self, distance_weight=1.0, confidence_weight=2.0, consistency_weight=0.1):
        super().__init__()
        self.distance_weight = distance_weight
        self.confidence_weight = confidence_weight
        self.consistency_weight = consistency_weight

    def forward(self, distance_pred, confidence_pred, distance_true, mask_true):
        # Only consider points where both mask is 1 and distance is non-zero
        valid_points = (mask_true > 0) & (distance_true > 0)

        # Distance loss only where objects exist
        if valid_points.sum() > 0:
            masked_distance_loss = F.mse_loss(
                distance_pred[valid_points],
                distance_true[valid_points]
            )
        else:
            masked_distance_loss = torch.tensor(0.0).to(distance_pred.device)

        # Confidence loss (binary cross entropy)
        confidence_loss = F.binary_cross_entropy(
            confidence_pred,
            mask_true
        )

        # Consistency loss between distance and confidence
        predicted_presence = (distance_pred > 0.05).float()
        consistency_loss = F.mse_loss(
            predicted_presence,
            confidence_pred
        )

        total_loss = (
                self.distance_weight * masked_distance_loss +
                self.confidence_weight * confidence_loss +
                self.consistency_weight * consistency_loss
        )

        return total_loss, {
            'distance_loss': masked_distance_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'consistency_loss': consistency_loss.item()
        }


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=50, patience=10):
    criterion = ContinuousFieldLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    max_grad_norm = 1.0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_distance_loss': [], 'train_confidence_loss': [], 'train_consistency_loss': [],
        'val_distance_loss': [], 'val_confidence_loss': [], 'val_consistency_loss': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_metrics = {k: 0.0 for k in ['total', 'distance', 'confidence', 'consistency']}

        for inputs, targets_dist, targets_mask in train_loader:
            inputs = inputs.to(device)
            targets_dist = targets_dist.to(device)
            targets_mask = targets_mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_dist, pred_conf = model(inputs)
            loss, component_losses = criterion(pred_dist, pred_conf, targets_dist, targets_mask)
            reg_loss = model.get_l2_regularization_loss()
            total_loss = loss + reg_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Track losses
            train_metrics['total'] += total_loss.item()
            train_metrics['distance'] += component_losses['distance_loss']
            train_metrics['confidence'] += component_losses['confidence_loss']
            train_metrics['consistency'] += component_losses['consistency_loss']

        # Validation phase
        model.eval()
        val_metrics = {k: 0.0 for k in ['total', 'distance', 'confidence', 'consistency']}

        with torch.no_grad():
            for inputs, targets_dist, targets_mask in val_loader:
                inputs = inputs.to(device)
                targets_dist = targets_dist.to(device)
                targets_mask = targets_mask.to(device)

                pred_dist, pred_conf = model(inputs)
                loss, component_losses = criterion(pred_dist, pred_conf, targets_dist, targets_mask)
                reg_loss = model.get_l2_regularization_loss()
                total_loss = loss + reg_loss

                val_metrics['total'] += total_loss.item()
                val_metrics['distance'] += component_losses['distance_loss']
                val_metrics['confidence'] += component_losses['confidence_loss']
                val_metrics['consistency'] += component_losses['consistency_loss']

        # Calculate average losses
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            val_metrics[k] /= len(val_loader)

        # Update learning rate
        scheduler.step(val_metrics['total'])
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['train_distance_loss'].append(train_metrics['distance'])
        history['train_confidence_loss'].append(train_metrics['confidence'])
        history['train_consistency_loss'].append(train_metrics['consistency'])
        history['val_distance_loss'].append(val_metrics['distance'])
        history['val_confidence_loss'].append(val_metrics['confidence'])
        history['val_consistency_loss'].append(val_metrics['consistency'])

        # Print epoch metrics
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['total']:.4f} (Dist: {train_metrics['distance']:.4f}, "
              f"Conf: {train_metrics['confidence']:.4f}, Cons: {train_metrics['consistency']:.4f})")
        print(f"Val Loss: {val_metrics['total']:.4f} (Dist: {val_metrics['distance']:.4f}, "
              f"Conf: {val_metrics['confidence']:.4f}, Cons: {val_metrics['consistency']:.4f})")
        print(f"Learning Rate: {current_lr:.6f}")

        # Early stopping
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break

    return history


def scale_microphone(data, min_val=-200, max_val=200):
    return (data - min_val) / (max_val - min_val) * 2 - 1


def scale_laser(data, min_val=0, max_val=0.1):
    return np.where(data != 0, (data - min_val) / (max_val - min_val), 0)


def plot_training_history(history):
    plt.figure(figsize=(15, 15))

    # Overall losses
    plt.subplot(3, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Overall Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Distance and Confidence losses
    plt.subplot(3, 1, 2)
    plt.plot(history['train_distance_loss'], label='Train Distance Loss')
    plt.plot(history['val_distance_loss'], label='Val Distance Loss')
    plt.plot(history['train_confidence_loss'], label='Train Confidence Loss')
    plt.plot(history['val_confidence_loss'], label='Val Confidence Loss')
    plt.title('Distance and Confidence Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Consistency loss
    plt.subplot(3, 1, 3)
    plt.plot(history['train_consistency_loss'], label='Train Consistency Loss')
    plt.plot(history['val_consistency_loss'], label='Val Consistency Loss')
    plt.title('Consistency Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_val, y_dist_val, y_mask_val, device):
    criterion = ContinuousFieldLoss()

    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Successfully loaded best model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Continuing with current model state")

    model.eval()
    with torch.no_grad():
        # Get predictions
        val_dist_pred, val_conf_pred = model(X_val)

        # Calculate losses
        loss, component_losses = criterion(
            val_dist_pred,
            val_conf_pred,
            y_dist_val,
            y_mask_val
        )

        # Calculate additional metrics
        val_dist_mae = nn.L1Loss()(val_dist_pred * y_mask_val, y_dist_val * y_mask_val)

    print("\nValidation Metrics:")
    print(f"Distance Loss: {component_losses['distance_loss']:.4f}")
    print(f"Confidence Loss: {component_losses['confidence_loss']:.4f}")
    print(f"Consistency Loss: {component_losses['consistency_loss']:.4f}")
    print(f"Distance MAE: {val_dist_mae.item():.4f}")

    return val_dist_pred, val_conf_pred


def visualize_predictions(val_dist_pred, val_conf_pred, y_dist_val, y_mask_val, sample_idx=2):
    plt.figure(figsize=(15, 10))

    # Distance predictions with confidence overlay
    plt.subplot(2, 1, 1)
    actual_dist = y_dist_val[sample_idx].cpu().numpy()
    predicted_dist = val_dist_pred[sample_idx].cpu().numpy()
    confidence = val_conf_pred[sample_idx].cpu().numpy()

    plt.plot(actual_dist, 'b-', label='Actual Distance', alpha=0.7)
    plt.plot(predicted_dist, 'r-', label='Predicted Distance', alpha=0.7)

    # Add confidence bands
    plt.fill_between(
        range(len(predicted_dist)),
        predicted_dist * (1 - confidence),
        predicted_dist * (1 + confidence),
        color='r',
        alpha=0.2,
        label='Confidence Interval'
    )

    plt.title('Distance Predictions with Confidence')
    plt.xlabel('Laser Index')
    plt.ylabel('Distance')
    plt.legend()

    # Confidence vs actual mask
    plt.subplot(2, 1, 2)
    actual_mask = y_mask_val[sample_idx].cpu().numpy()
    plt.plot(actual_mask, 'b-', label='Actual Object Presence', alpha=0.7)
    plt.plot(confidence, 'r-', label='Predicted Confidence', alpha=0.7)
    plt.title('Object Presence and Prediction Confidence')
    plt.xlabel('Laser Index')
    plt.ylabel('Probability')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate sample-specific metrics
    mask_present = actual_mask > 0.5
    if mask_present.any():
        sample_dist_mse = np.mean((actual_dist[mask_present] - predicted_dist[mask_present]) ** 2)
        sample_dist_mae = np.mean(np.abs(actual_dist[mask_present] - predicted_dist[mask_present]))
    else:
        sample_dist_mse = sample_dist_mae = 0.0

    sample_conf_bce = F.binary_cross_entropy(
        torch.tensor(confidence),
        torch.tensor(actual_mask)
    ).item()

    print("\nSample-specific Metrics:")
    print(f"Distance MSE: {sample_dist_mse:.4f}")
    print(f"Distance MAE: {sample_dist_mae:.4f}")
    print(f"Confidence BCE: {sample_conf_bce:.4f}")


def visualize_data(X, y_dist, y_mask, X_scaled, y_dist_scaled, index):
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Data Visualization for Simulation {index}")

    # Original microphone data
    plt.subplot(2, 1, 1)
    for i in range(X.shape[1]):
        plt.plot(X[index, i], label=f'Microphone {i + 1}')
    plt.title("Original Microphone Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()

    # Distance field visualization
    plt.subplot(2, 1, 2)
    laser_indices = range(len(y_dist_scaled[index]))

    # Plot the complete distance field
    plt.plot(laser_indices, y_dist_scaled[index], 'b-', label='Distance Field', alpha=0.7)

    # Highlight only non-zero measurements where mask is 1
    mask = y_mask[index]
    distances = y_dist_scaled[index]
    valid_indices = np.where((mask > 0) & (distances > 0))[0]

    if len(valid_indices) > 0:
        plt.scatter(valid_indices, distances[valid_indices],
                    c='red', s=50, alpha=0.5, label='Valid Measurements')

    plt.title("Distance Field with Valid Measurements")
    plt.xlabel("Laser Index")
    plt.ylabel("Scaled Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print quick validation
    print(f"\nNumber of actual valid measurements: {len(valid_indices)}")


def main():
    X, y_dist, y_mask = load_and_preprocess_data("simulation_data/simulation_data_parallel.h5", num_simulations=7500)

    # Apply simple scaling to microphone data
    X_scaled = scale_microphone(X)

    # Scale laser distances
    y_dist_scaled = scale_laser(y_dist)

    # Visualize a random simulation
    random_index = np.random.randint(0, X.shape[0])
    visualize_data(X, y_dist, y_mask, X_scaled, y_dist_scaled, random_index)

    # Split the data
    X_train, X_val, y_dist_train, y_dist_val, y_mask_train, y_mask_val = train_test_split(
        X_scaled, y_dist_scaled, y_mask, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_dist_train = torch.FloatTensor(y_dist_train).to(device)
    y_mask_train = torch.FloatTensor(y_mask_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_dist_val = torch.FloatTensor(y_dist_val).to(device)
    y_mask_val = torch.FloatTensor(y_mask_val).to(device)

    # Create datasets and dataloaders
    batch_size = 16
    train_dataset = TensorDataset(X_train, y_dist_train, y_mask_train)
    val_dataset = TensorDataset(X_val, y_dist_val, y_mask_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the new AcousticNN model
    input_channels = X_scaled.shape[1]  # Number of microphones (7)
    sequence_length = X_scaled.shape[2]  # Length of the time series
    output_size = y_dist.shape[1]  # Number of distance/mask points (61)

    # Initialize model
    model = AcousticNN(
        input_channels=7,
        sequence_length=3695,
        output_size=61
    ).to(device)

    model.print_param_count()

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Setup scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-5,
    )

    # Train model
    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_epochs=500,
        patience=50
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    val_dist_pred, val_conf_pred = evaluate_model(
        model,
        X_val,
        y_dist_val,
        y_mask_val,
        device
    )

    # After loading and preprocessing data
    random_index = np.random.randint(0, X.shape[0])
    visualize_data(X, y_dist, y_mask, X_scaled, y_dist_scaled, random_index)
    visualize_predictions(val_dist_pred, val_conf_pred, y_dist_val, y_mask_val, random_index)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()
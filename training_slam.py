import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_and_preprocess_data(filename, num_simulations):
    with h5py.File(filename, 'r') as f:
        # Fix: Changed from simulation{i} to simulation_{i} to match your data format
        maxlength = max(f[f"simulation_{i}"]["microphone_data"].shape[1] for i in range(num_simulations))

        X, y_dist, y_mask = [], [], []
        for i in range(num_simulations):
            # Fix: Changed simgroup to sim_group to match variable name below
            sim_group = f[f"simulation_{i}"]  # Changed from simulation{i}
            microphone_data = sim_group['microphone_data'][:]
            laser_distances = sim_group['laser_distances'][:]

            # Fix: Changed max_length to maxlength to match variable name above
            padded_data = np.zeros((microphone_data.shape[0], maxlength))
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

    # Distance and Validity losses
    plt.subplot(3, 1, 2)
    plt.plot(history['train_distance_loss'], label='Train Distance Loss')
    plt.plot(history['val_distance_loss'], label='Val Distance Loss')
    plt.plot(history['train_validity_loss'], label='Train Validity Loss')
    plt.plot(history['val_validity_loss'], label='Val Validity Loss')
    plt.title('Distance and Validity Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Invalid Prediction loss
    plt.subplot(3, 1, 3)
    plt.plot(history['train_invalid_pred_loss'], label='Train Invalid Pred Loss')
    plt.plot(history['val_invalid_pred_loss'], label='Val Invalid Pred Loss')
    plt.title('Invalid Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


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

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            128, 64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # SLAM-optimized output
        self.field_output = SLAMOptimizedOutput(128, output_size)

    def forward(self, x):
        # Feature extraction
        x = self.feature_extraction(x)

        # Apply attention
        x = self.mic_attention(x)
        x = self.attention_dropout(x)
        x = self.spatial_attention(x)
        x = self.attention_dropout(x)

        # Temporal processing
        x = self.temporal_conv(x)

        # LSTM processing
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)

        # Global average pooling with dropout
        x = F.dropout(x.mean(dim=1), p=0.3, training=self.training)

        # Get SLAM-optimized output
        return self.field_output(x)

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


class SLAMOptimizedOutput(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = 128

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Distance prediction branch
        self.distance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Measurement validity prediction
        self.validity_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

        # Uncertainty prediction
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, x):
        shared_features = self.shared(x)

        # Get raw predictions
        distances = self.distance_net(shared_features)
        validity = self.validity_net(shared_features)
        uncertainty = self.uncertainty_net(shared_features)

        # Create masked distances (NaN where invalid)
        masked_distances = distances.clone()
        masked_distances[validity < 0.5] = float('nan')

        return {
            'distances': masked_distances,
            'raw_distances': distances,
            'validity': validity,
            'uncertainty': uncertainty
        }


class SLAMOptimizedLoss(nn.Module):
    def __init__(self, valid_threshold=0.5):
        super().__init__()
        self.valid_threshold = valid_threshold

    def forward(self, predictions, targets, target_mask):
        # Unpack predictions
        raw_distances = predictions['raw_distances']
        validity = predictions['validity']
        uncertainty = predictions['uncertainty']

        # Convert target mask to float and handle -1 values
        valid_measurements = (target_mask > 0).float()

        # Validity loss (binary cross entropy)
        validity_loss = F.binary_cross_entropy(
            validity,
            valid_measurements,
            reduction='mean'
        )

        # Only compute distance loss for valid measurements
        valid_indices = (target_mask > 0)
        if valid_indices.any():
            # Distance error term
            distance_error = F.mse_loss(
                raw_distances[valid_indices],
                targets[valid_indices],
                reduction='none'
            )

            # Modified uncertainty weighting
            uncertainty_term = torch.exp(-uncertainty[valid_indices])
            weighted_distance_loss = (
                    distance_error * uncertainty_term +
                    0.5 * uncertainty[valid_indices]
            ).mean()
        else:
            weighted_distance_loss = torch.tensor(0.0).to(raw_distances.device)

        # Penalize predicting distances for invalid measurements
        invalid_indices = ~valid_indices
        invalid_prediction_loss = torch.abs(raw_distances[invalid_indices]).mean() if invalid_indices.any() else 0

        # Combine losses with positive weights
        total_loss = (
                weighted_distance_loss +
                validity_loss +
                0.1 * invalid_prediction_loss
        )

        return total_loss, {
            'distance_loss': weighted_distance_loss.item(),
            'validity_loss': validity_loss.item(),
            'invalid_pred_loss': invalid_prediction_loss if isinstance(invalid_prediction_loss,
                                                                       float) else invalid_prediction_loss.item()
        }


def process_for_slam(model_output, confidence_threshold=0.5):
    """
    Convert model outputs to SLAM-friendly format
    """
    distances = model_output['distances']  # Already contains NaN for invalid
    uncertainties = model_output['uncertainty']

    # Create structured array for SLAM
    slam_measurements = {
        'distances': distances.cpu().numpy(),
        'uncertainties': uncertainties.cpu().numpy(),
        'valid_mask': ~torch.isnan(distances).cpu().numpy()
    }

    return slam_measurements


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=50, patience=10):
    criterion = SLAMOptimizedLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    max_grad_norm = 1.0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_distance_loss': [], 'train_validity_loss': [], 'train_invalid_pred_loss': [],
        'val_distance_loss': [], 'val_validity_loss': [], 'val_invalid_pred_loss': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_metrics = {k: 0.0 for k in ['total', 'distance', 'validity', 'invalid_pred']}

        for inputs, targets_dist, targets_mask in train_loader:
            inputs = inputs.to(device)
            targets_dist = targets_dist.to(device)
            targets_mask = targets_mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(inputs)
            loss, component_losses = criterion(predictions, targets_dist, targets_mask)
            reg_loss = model.get_l2_regularization_loss()
            total_loss = loss + reg_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Track losses
            train_metrics['total'] += total_loss.item()
            train_metrics['distance'] += component_losses['distance_loss']
            train_metrics['validity'] += component_losses['validity_loss']
            train_metrics['invalid_pred'] += component_losses['invalid_pred_loss']

        # Validation phase
        model.eval()
        val_metrics = {k: 0.0 for k in ['total', 'distance', 'validity', 'invalid_pred']}

        with torch.no_grad():
            for inputs, targets_dist, targets_mask in val_loader:
                inputs = inputs.to(device)
                targets_dist = targets_dist.to(device)
                targets_mask = targets_mask.to(device)

                predictions = model(inputs)
                loss, component_losses = criterion(predictions, targets_dist, targets_mask)
                reg_loss = model.get_l2_regularization_loss()
                total_loss = loss + reg_loss

                val_metrics['total'] += total_loss.item()
                val_metrics['distance'] += component_losses['distance_loss']
                val_metrics['validity'] += component_losses['validity_loss']
                val_metrics['invalid_pred'] += component_losses['invalid_pred_loss']

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
        history['train_validity_loss'].append(train_metrics['validity'])
        history['train_invalid_pred_loss'].append(train_metrics['invalid_pred'])
        history['val_distance_loss'].append(val_metrics['distance'])
        history['val_validity_loss'].append(val_metrics['validity'])
        history['val_invalid_pred_loss'].append(val_metrics['invalid_pred'])

        # Print epoch metrics
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['total']:.4f} (Dist: {train_metrics['distance']:.4f}, "
              f"Valid: {train_metrics['validity']:.4f}, Invalid: {train_metrics['invalid_pred']:.4f})")
        print(f"Val Loss: {val_metrics['total']:.4f} (Dist: {val_metrics['distance']:.4f}, "
              f"Valid: {val_metrics['validity']:.4f}, Invalid: {val_metrics['invalid_pred']:.4f})")
        print(f"Learning Rate: {current_lr:.6f}")

        # Update the model saving part
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0
            save_model(model, 'best_model_slam.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break

    return history


def enhanced_evaluate_model(model, X_val, y_dist_val, y_mask_val):
    criterion = SLAMOptimizedLoss()

    load_model(model, 'best_model_slam.pth')
    model.eval()

    metrics = {}
    with torch.no_grad():
        predictions = model(X_val)
        loss, component_losses = criterion(predictions, y_dist_val, y_mask_val)

        # Calculate enhanced metrics
        valid_measurements = (y_mask_val > 0)
        if valid_measurements.any():
            metrics['mae'] = F.l1_loss(
                predictions['raw_distances'][valid_measurements],
                y_dist_val[valid_measurements]
            ).item()

            # Calculate uncertainty calibration
            uncertainty = predictions['uncertainty'][valid_measurements]
            errors = torch.abs(
                predictions['raw_distances'][valid_measurements] -
                y_dist_val[valid_measurements]
            )
            metrics['uncertainty_correlation'] = torch.corrcoef(
                torch.stack([uncertainty.flatten(), errors.flatten()])
            )[0, 1].item()

        # Calculate validity prediction metrics
        metrics['validity_accuracy'] = ((predictions['validity'] > 0.5) ==
                                        (y_mask_val > 0)).float().mean().item()

    print("\nEnhanced Validation Metrics:")
    print(f"Distance Loss: {component_losses['distance_loss']:.4f}")
    print(f"Validity Loss: {component_losses['validity_loss']:.4f}")
    print(f"Invalid Prediction Loss: {component_losses['invalid_pred_loss']:.4f}")
    print(f"Distance MAE: {metrics['mae']:.4f}")
    print(f"Validity Accuracy: {metrics['validity_accuracy']:.4f}")
    print(f"Uncertainty Correlation: {metrics['uncertainty_correlation']:.4f}")

    return predictions, metrics


def enhanced_visualize_predictions(predictions, y_dist_val, y_mask_val, sample_idx=2):
    plt.figure(figsize=(15, 20))

    # Extract predictions
    distances = predictions['raw_distances'][sample_idx].cpu().numpy()
    masked_distances = predictions['distances'][sample_idx].cpu().numpy()
    validity = predictions['validity'][sample_idx].cpu().numpy()
    uncertainty = predictions['uncertainty'][sample_idx].cpu().numpy()

    # Ground truth
    actual_dist = y_dist_val[sample_idx].cpu().numpy()
    actual_mask = y_mask_val[sample_idx].cpu().numpy()

    # Distance predictions with uncertainty
    plt.subplot(4, 1, 1)
    plt.plot(actual_dist, 'b-', label='Actual Distance', alpha=0.7)
    plt.plot(distances, 'r--', label='Predicted Distance', alpha=0.7)
    plt.fill_between(
        range(len(distances)),
        distances - 2 * uncertainty,
        distances + 2 * uncertainty,
        color='r',
        alpha=0.2,
        label='Uncertainty (2σ)'
    )
    plt.title('Distance Predictions with Uncertainty')
    plt.xlabel('Laser Index')
    plt.ylabel('Distance')
    plt.legend()

    # Validity predictions
    plt.subplot(4, 1, 2)
    plt.plot(actual_mask, 'b-', label='Actual Valid Measurements', alpha=0.7)
    plt.plot(validity, 'r--', label='Predicted Validity', alpha=0.7)
    plt.title('Measurement Validity Prediction')
    plt.xlabel('Laser Index')
    plt.ylabel('Probability')
    plt.legend()

    # Uncertainty visualization
    plt.subplot(4, 1, 3)
    plt.plot(uncertainty, 'g-', label='Prediction Uncertainty', alpha=0.7)
    valid_errors = np.abs(distances - actual_dist) * (actual_mask > 0)
    plt.plot(valid_errors, 'r--', label='Actual Errors', alpha=0.7)
    plt.title('Uncertainty vs Actual Errors')
    plt.xlabel('Laser Index')
    plt.ylabel('Magnitude')
    plt.legend()

    # SLAM-ready output
    plt.subplot(4, 1, 4)
    plt.plot(actual_dist * (actual_mask > 0), 'b-', label='Actual Valid Distances', alpha=0.7)
    plt.plot(masked_distances, 'r--', label='SLAM-ready Distances', alpha=0.7)
    plt.title('SLAM-ready Distance Output')
    plt.xlabel('Laser Index')
    plt.ylabel('Distance')
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_model(model, filename):
    """
    Safely save model weights
    """
    torch.save(
        model.state_dict(),
        filename,
        _use_new_zipfile_serialization=True
    )


def load_model(model, filename):
    """
    Safely load model weights
    """
    try:
        state_dict = torch.load(
            filename,
            weights_only=True  # Safe loading mode
        )
        model.load_state_dict(state_dict)
        print("Successfully loaded best model weights")
        return True
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Continuing with current model state")
        return False


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

    # Print validation info
    total_points = len(y_mask[index])
    valid_points = len(valid_indices)
    print(f"\nStatistics for Visualized Sample:")
    print(f"Total laser points: {total_points}")
    print(f"Valid measurements: {valid_points} ({valid_points/total_points*100:.1f}%)")
    print(f"Invalid/No detection: {total_points-valid_points} ({(total_points-valid_points)/total_points*100:.1f}%)\n")


def main():
    # Load and preprocess data
    X, y_dist, y_mask = load_and_preprocess_data("simulation_data/simulation_data_parallel.h5", num_simulations=12000)

    # Apply scaling
    X_scaled = scale_microphone(X)
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

    # Initialize model
    model = AcousticNN(
        input_channels=7,
        sequence_length=3695,
        output_size=61
    ).to(device)

    model.print_param_count()

    # Update optimizer configuration
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # Slightly higher initial LR for warmup
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Update scheduler for cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval each time
        eta_min=1e-6
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
    predictions, metrics = enhanced_evaluate_model(  # Use this instead of evaluate_model
        model,
        X_val,
        y_dist_val,
        y_mask_val
    )

    # Visualize predictions
    random_val_idx = np.random.randint(0, len(X_val))

    # And update the visualization call
    enhanced_visualize_predictions(  # Use this instead of visualize_predictions
        predictions,
        y_dist_val,
        y_mask_val,
        random_val_idx
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()

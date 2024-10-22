import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_and_preprocess_data(filename, num_simulations=1000):
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


class SimplifiedCNNLSTM(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimplifiedCNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_shape[0], 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4)
        )

        self.lstm = nn.LSTM(64, 32, num_layers=1, batch_first=True, bidirectional=True)

        self.fc_shared = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc_distance = nn.Linear(64, num_classes)
        self.fc_mask = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)  # Global average pooling

        shared_features = self.fc_shared(x)

        distance_output = self.fc_distance(shared_features)
        mask_output = torch.sigmoid(self.fc_mask(shared_features))

        return distance_output, mask_output

    def print_param_count(self):
        print("Parameter count per layer:")
        for name, module in self.named_children():
            if isinstance(module, nn.Sequential):
                print(f"  {name}:")
                for i, layer in enumerate(module):
                    param_count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    print(f"    Layer {i}: {param_count:,} parameters")
            else:
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"  {name}: {param_count:,} parameters")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal trainable parameters: {total_params:,}")


class BalancedCNNLSTM(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(BalancedCNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_shape[0], 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4)
        )

        self.cnn_output_size = 128 * (input_shape[1] // 64)

        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True)

        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)

        self.fc_shared = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc_distance = nn.Linear(128, num_classes)
        self.fc_mask = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        x = x.mean(dim=1)  # Global average pooling

        shared_features = self.fc_shared(x)

        distance_output = self.fc_distance(shared_features)
        mask_output = torch.sigmoid(self.fc_mask(shared_features))

        return distance_output, mask_output

    def print_param_count(self):
        print("Parameter count per layer:")
        for name, module in self.named_children():
            if isinstance(module, nn.Sequential):
                print(f"  {name}:")
                for i, layer in enumerate(module):
                    param_count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    print(f"    Layer {i}: {param_count:,} parameters")
            else:
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"  {name}: {param_count:,} parameters")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal trainable parameters: {total_params:,}")


def train_model(model, train_loader, val_loader, criterion_dist, criterion_mask, optimizer, scheduler, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets_dist, targets_mask in train_loader:
            inputs, targets_dist, targets_mask = inputs.to(device), targets_dist.to(device), targets_mask.to(device)
            optimizer.zero_grad()
            outputs_dist, outputs_mask = model(inputs)
            loss_dist = criterion_dist(outputs_dist, targets_dist)
            loss_mask = criterion_mask(outputs_mask, targets_mask)
            loss = loss_dist + loss_mask
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets_dist, targets_mask in val_loader:
                inputs, targets_dist, targets_mask = inputs.to(device), targets_dist.to(device), targets_mask.to(device)
                outputs_dist, outputs_mask = model(inputs)
                loss_dist = criterion_dist(outputs_dist, targets_dist)
                loss_mask = criterion_mask(outputs_mask, targets_mask)
                loss = loss_dist + loss_mask
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    return train_losses, val_losses


def scale_microphone(data, min_val=-200, max_val=200):
    return (data - min_val) / (max_val - min_val) * 2 - 1


def scale_laser(data, min_val=0, max_val=0.1):
    return np.where(data != 0, (data - min_val) / (max_val - min_val), 0)


def visualize_data(X, y_dist, y_mask, X_scaled, y_dist_scaled, index):
    plt.figure(figsize=(15, 15))
    plt.suptitle(f"Data Visualization for Simulation {index}")

    # Original microphone data
    plt.subplot(3, 1, 1)
    for i in range(X.shape[1]):
        plt.plot(X[index, i], label=f'Microphone {i + 1}')
    plt.title("Original Microphone Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()

    # Scaled microphone data
    plt.subplot(3, 1, 2)
    for i in range(X_scaled.shape[1]):
        plt.plot(X_scaled[index, i], label=f'Microphone {i + 1}')
    plt.title("Scaled Microphone Data")
    plt.xlabel("Time")
    plt.ylabel("Scaled Amplitude")
    plt.legend()

    # Laser distances and mask
    plt.subplot(3, 1, 3)
    plt.scatter(y_dist_scaled[index], range(len(y_dist_scaled[index])), c='blue', s=20, label='Distance')
    plt.scatter(y_mask[index], range(len(y_mask[index])), c='red', s=10, alpha=0.5, label='Mask')
    plt.title("Scaled Laser Distances and Mask")
    plt.xlabel("Scaled Distance / Mask Value")
    plt.ylabel("Laser Index")
    plt.legend()
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    # Print the laser measurements and mask
    print("Scaled laser measurements (1D vector):")
    print(y_dist_scaled[index])
    print("\nLaser measurement mask (1D vector):")
    print(y_mask[index])


def main():
    X, y_dist, y_mask = load_and_preprocess_data("simulation_data_simple.h5", num_simulations=1000)

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
    train_dataset = TensorDataset(X_train, y_dist_train, y_mask_train)
    val_dataset = TensorDataset(X_val, y_dist_val, y_mask_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize the model
    input_shape = (X_scaled.shape[1], X_scaled.shape[2])
    num_classes = y_dist.shape[1]
    model = SimplifiedCNNLSTM(input_shape, num_classes).to(device)

    # Print parameter count
    model.print_param_count()

    # Define loss functions
    criterion_dist = nn.MSELoss()
    criterion_mask = nn.BCELoss()

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion_dist, criterion_mask, optimizer,
                                           scheduler, num_epochs=200, patience=30)

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate the model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        val_dist_pred, val_mask_pred = model(X_val)
        val_dist_mse = criterion_dist(val_dist_pred, y_dist_val)
        val_mask_bce = criterion_mask(val_mask_pred, y_mask_val)
        val_dist_mae = nn.L1Loss()(val_dist_pred, y_dist_val)

    print(f"Validation Distance MSE: {val_dist_mse.item():.4f}")
    print(f"Validation Distance MAE: {val_dist_mae.item():.4f}")
    print(f"Validation Mask BCE: {val_mask_bce.item():.4f}")

    # Visualize predictions
    sample_idx = 2
    plt.figure(figsize=(12, 6))

    # Distance predictions
    plt.subplot(2, 1, 1)
    actual_dist = y_dist_val[sample_idx].cpu().numpy()
    predicted_dist = val_dist_pred[sample_idx].cpu().numpy()
    plt.plot(actual_dist, label='Actual Distance')
    plt.plot(predicted_dist, label='Predicted Distance')
    plt.title('Laser Distance Predictions vs Actual')
    plt.xlabel('Laser Index')
    plt.ylabel('Distance')
    plt.legend()

    # Mask predictions
    plt.subplot(2, 1, 2)
    actual_mask = y_mask_val[sample_idx].cpu().numpy()
    predicted_mask = val_mask_pred[sample_idx].cpu().numpy()
    plt.plot(actual_mask, label='Actual Mask')
    plt.plot(predicted_mask, label='Predicted Mask')
    plt.title('Object Presence Mask Predictions vs Actual')
    plt.xlabel('Laser Index')
    plt.ylabel('Probability')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate and print sample-specific metrics
    sample_dist_mse = np.mean((actual_dist - predicted_dist) ** 2)
    sample_dist_mae = np.mean(np.abs(actual_dist - predicted_dist))
    sample_mask_bce = nn.BCELoss()(torch.tensor(predicted_mask), torch.tensor(actual_mask)).item()

    print(f"Sample Distance MSE: {sample_dist_mse:.4f}")
    print(f"Sample Distance MAE: {sample_dist_mae:.4f}")
    print(f"Sample Mask BCE: {sample_mask_bce:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()
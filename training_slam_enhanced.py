import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
from dataclasses import dataclass
from typing import Dict, Tuple


# Data Processing Functions
def load_data(filename, num_simulations):
    """Load and preprocess data from HDF5 file."""
    with h5py.File(filename, 'r') as f:
        maxlength = max(f[f"simulation_{i}"]["microphone_data"].shape[1] for i in range(num_simulations))

        X, y_dist, y_mask = [], [], []
        for i in range(num_simulations):
            sim_group = f[f"simulation_{i}"]
            mic_data = sim_group['microphone_data'][:]
            laser_dist = sim_group['laser_distances'][:]

            # Pad microphone data
            padded_data = np.zeros((mic_data.shape[0], maxlength))
            padded_data[:, :mic_data.shape[1]] = mic_data

            # Process laser data
            dist = np.where(laser_dist == -1, 0, laser_dist)
            mask = (laser_dist != -1).astype(float)

            X.append(padded_data)
            y_dist.append(dist)
            y_mask.append(mask)

    return map(np.array, [X, y_dist, y_mask])


def add_acoustic_noise(x, noise_params):
    """
    Add realistic acoustic noise to input signals.
    """
    device = x.device
    noisy = x.clone()

    # Gaussian noise (environmental/thermal noise)
    if noise_params['gaussian_std'] > 0:
        noise = torch.randn_like(x) * noise_params['gaussian_std']
        noisy = noisy + noise

    # Impulse noise (sudden acoustic events)
    if noise_params['impulse_prob'] > 0:
        mask = torch.rand_like(x) < noise_params['impulse_prob']
        impulses = torch.randn_like(x) * noise_params['impulse_strength']
        noisy = torch.where(mask, noisy + impulses, noisy)

    # Phase noise (timing jitter)
    if noise_params['phase_noise_std'] > 0:
        batch_size, channels, seq_len = x.shape

        # Create time shifts for each channel in each batch
        phase_shift = torch.randn(batch_size, channels, device=device) * noise_params['phase_noise_std']

        # Create time base
        t = torch.arange(seq_len, device=device).float()

        # Apply shifts channel-wise
        noisy_phase = torch.zeros_like(noisy)
        for b in range(batch_size):
            for c in range(channels):
                # Interpolate shifted signal
                shift = phase_shift[b, c]
                shifted_t = t + shift

                # Ensure shifted time points are within bounds
                shifted_t = torch.clamp(shifted_t, 0, seq_len - 1)

                # Linear interpolation
                shifted_t_floor = shifted_t.floor().long()
                shifted_t_ceil = shifted_t.ceil().long()
                weight_ceil = shifted_t - shifted_t_floor
                weight_floor = 1 - weight_ceil

                noisy_phase[b, c] = (noisy[b, c, shifted_t_floor] * weight_floor +
                                     noisy[b, c, shifted_t_ceil] * weight_ceil)

        noisy = noisy_phase

    return noisy


def calculate_frequency_weights(n_fft, sample_rate=5000000, freq_range=(20000, 100000)):
    """
    Calculate frequency-dependent weights based on chirp characteristics.

    Args:
        n_fft: FFT size
        sample_rate: Sampling rate in Hz (default 1MHz based on data)
        freq_range: Tuple of (min_freq, max_freq) of chirp in Hz

    Returns:
        numpy array of weights for each frequency bin
    """
    # Calculate frequency bins
    freqs = np.fft.rfftfreq(n_fft, d=1 / sample_rate)

    # Create weights that emphasize the chirp frequency range
    weights = np.ones_like(freqs)

    # Find indices corresponding to chirp frequency range
    min_idx = np.searchsorted(freqs, freq_range[0])
    max_idx = np.searchsorted(freqs, freq_range[1])

    # Apply frequency-dependent weighting
    # Higher frequencies get higher weights to compensate for attenuation
    chirp_weights = np.linspace(1.0, 2.0, max_idx - min_idx)
    weights[min_idx:max_idx] = chirp_weights

    # Smoothly taper weights outside chirp range
    window = signal.windows.hann(min_idx * 2)
    weights[:min_idx] = window[:min_idx]
    if max_idx < len(weights):
        window = signal.windows.hann(2 * (len(weights) - max_idx))
        weights[max_idx:] = window[len(window) // 2:]

    return weights


def frequency_normalize_batch(mic_data, n_fft=256):
    """
    Apply frequency-based normalization to a batch of microphone data.

    Args:
        mic_data: Tensor of shape (batch_size, n_channels, time_steps)
        n_fft: FFT size for STFT

    Returns:
        Normalized tensor of same shape
    """
    device = mic_data.device
    batch_size, n_channels, time_steps = mic_data.shape

    # Calculate frequency weights (only need to do this once)
    weights = calculate_frequency_weights(n_fft)
    weights = torch.from_numpy(weights).float().to(device)

    # Prepare for STFT
    hop_length = n_fft // 4  # 75% overlap
    window = torch.hann_window(n_fft).to(device)

    # Process each channel separately to avoid memory issues
    normalized = torch.zeros_like(mic_data)

    for b in range(batch_size):
        for c in range(n_channels):
            # Get single channel data
            single_channel = mic_data[b, c]

            # Ensure the input length is sufficient for STFT
            if single_channel.shape[0] < n_fft:
                pad_size = n_fft - single_channel.shape[0]
                single_channel = F.pad(single_channel, (0, pad_size))

            # Apply STFT
            stft = torch.stft(
                single_channel,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
                normalized=True  # Add normalization
            )

            # Apply frequency-dependent weights
            stft *= weights[:stft.shape[0], None]

            # Inverse STFT
            channel_normalized = torch.istft(
                stft,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                length=time_steps,
                normalized=True  # Add normalization
            )

            normalized[b, c] = channel_normalized

    # Scale to original range
    max_val = torch.max(torch.abs(normalized))
    if max_val > 0:
        normalized = normalized * (torch.max(torch.abs(mic_data)) / max_val)

    return normalized


def scale_data(mic_data, laser_data):
    """Scale input and output data to appropriate ranges."""
    # Convert to torch tensor if needed
    if isinstance(mic_data, np.ndarray):
        mic_data = torch.from_numpy(mic_data).float()

    # Move calculations to CPU initially to avoid CUDA issues
    mic_data_cpu = mic_data.cpu()

    # Apply frequency-based normalization
    mic_data_normalized = frequency_normalize_batch(mic_data_cpu)

    # Scale to [-1, 1] range properly
    mic_max = torch.max(torch.abs(mic_data_normalized))
    if mic_max > 0:
        mic_scaled = mic_data_normalized / mic_max
    else:
        mic_scaled = mic_data_normalized

    # Move back to original device
    mic_scaled = mic_scaled.to(mic_data.device)

    # Scale laser data as before
    laser_scaled = np.where(laser_data != 0, laser_data / 0.1, 0)

    return mic_scaled, laser_scaled


# Model Components

@dataclass
class PhysicsConstants:
    """Physical constants for acoustic validation"""
    speed_of_sound: float = 343.0  # m/s
    sampling_rate: int = 5000000  # Hz (5MHz from data)
    chirp_duration: float = 0.0001  # seconds
    mic_spacing: float = 0.01  # meters
    freq_range: Tuple[int, int] = (20000, 100000)  # Hz
    max_distance: float = 0.1  # meters


class PhysicsMetrics:
    def __init__(self, constants: PhysicsConstants = PhysicsConstants()):
        self.c = constants
        self.samples_per_meter = self.c.sampling_rate / self.c.speed_of_sound
        self.min_samples_between_reflections = int(2 * self.c.mic_spacing * self.samples_per_meter)

    def time_of_flight_validation(self, mic_data: torch.Tensor,
                                  predicted_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, n_channels, time_steps = mic_data.shape
        device = mic_data.device

        # Process analytic signal calculation (unchanged)
        analytic_signals = []
        for b in range(batch_size):
            channel_signals = []
            for c in range(n_channels):
                single_channel = mic_data[b, c]
                stft = torch.stft(single_channel,
                                  n_fft=256,
                                  hop_length=128,
                                  window=torch.hann_window(256, device=device),
                                  return_complex=True)
                channel_signals.append(stft)
            analytic_signals.append(torch.stack(channel_signals))
        analytic_signal = torch.stack(analytic_signals)
        envelope = torch.abs(analytic_signal).max(dim=2)[0]

        # Find actual peaks
        peaks = []
        for i in range(1, envelope.shape[-1] - 1):
            is_peak = (envelope[..., i] > envelope[..., i - 1]) & \
                      (envelope[..., i] > envelope[..., i + 1])
            if is_peak.any():  # If there's a peak in any batch/channel
                peaks.append(i * 128)

        # Handle case where no peaks are found
        if not peaks:
            return {
                'tof_violations': torch.zeros(batch_size, device=device).cpu(),
                'tof_violation_rate': torch.zeros(batch_size, device=device).cpu()
            }

        peaks = torch.tensor(peaks, device=device)

        # Calculate violations with safe handling
        violations = torch.zeros(batch_size, device=device)
        valid_predictions = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            valid_mask = predicted_distances[i] > 0
            valid_predictions[i] = valid_mask.sum()

            for pred_delay in predicted_distances[i][valid_mask]:
                diffs = torch.abs(peaks - pred_delay * self.samples_per_meter)
                min_diff = torch.min(diffs)
                if min_diff > self.samples_per_meter * 0.01:
                    violations[i] += 1

        # Safe division for violation rate
        violation_rate = torch.where(valid_predictions > 0,
                                     violations / valid_predictions,
                                     torch.zeros_like(violations))

        return {
            'tof_violations': violations.cpu(),
            'tof_violation_rate': violation_rate.cpu()
        }

    def amplitude_decay_validation(self, mic_data: torch.Tensor,
                                   predicted_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, n_channels, time_steps = mic_data.shape
        device = mic_data.device

        # Process analytic signal (unchanged)
        analytic_signals = []
        for b in range(batch_size):
            channel_signals = []
            for c in range(n_channels):
                single_channel = mic_data[b, c]
                stft = torch.stft(single_channel,
                                  n_fft=256,
                                  hop_length=128,
                                  window=torch.hann_window(256, device=device),
                                  return_complex=True)
                channel_signals.append(stft)
            analytic_signals.append(torch.stack(channel_signals))
        analytic_signal = torch.stack(analytic_signals)
        envelope = torch.abs(analytic_signal).max(dim=2)[0]

        # Expected decay calculation with safe handling
        expected_amplitudes = torch.where(predicted_distances > 0,
                                          1 / (predicted_distances ** 2 + 1e-6),
                                          torch.zeros_like(predicted_distances))

        amplitude_errors = torch.zeros(batch_size, device=device)
        valid_measurements = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            valid_mask = predicted_distances[i] > 0
            valid_measurements[i] = valid_mask.sum()

            if valid_mask.any():
                delays = (predicted_distances[i] * self.samples_per_meter).long()
                valid_delays = delays[valid_mask]
                valid_delays_clipped = torch.clamp(valid_delays // 128, 0, envelope.size(-1) - 1)

                if valid_delays_clipped.numel() > 0:
                    amps = envelope[i, :, valid_delays_clipped].max(dim=0)[0]
                    if amps.max() > 0:
                        expected_norm = expected_amplitudes[i, valid_mask]
                        expected_norm = expected_norm / (expected_norm.max() + 1e-6)
                        actual_norm = amps / (amps.max() + 1e-6)
                        amplitude_errors[i] = torch.mean(torch.abs(expected_norm - actual_norm))

        # Safe mean calculation
        batch_mean_error = torch.where(valid_measurements > 0,
                                       amplitude_errors.sum() / (valid_measurements.sum() + 1e-6),
                                       torch.tensor(0., device=device))

        # Take the mean across the batch to get a single scalar
        mean_error = batch_mean_error.mean()

        return {
            'amplitude_errors': amplitude_errors.cpu(),
            'mean_amplitude_error': mean_error.cpu()
        }

    def spatial_consistency_validation(self, predicted_distances: torch.Tensor,
                                       valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        max_diff = self.c.mic_spacing

        # Calculate differences between adjacent measurements
        diffs = torch.abs(predicted_distances[:, 1:] - predicted_distances[:, :-1])
        valid_pairs = valid_mask[:, 1:] * valid_mask[:, :-1]

        # Find violations with safe handling
        violations = (diffs > max_diff) * valid_pairs
        violation_counts = torch.sum(violations, dim=1)
        valid_pair_counts = torch.sum(valid_pairs, dim=1)

        # Safe division for violation rate
        violation_rate = torch.where(valid_pair_counts > 0,
                                     violation_counts / valid_pair_counts,
                                     torch.zeros_like(violation_counts))

        return {
            'spatial_violations': violation_counts.cpu(),
            'spatial_violation_rate': violation_rate.cpu()
        }

    def multi_path_validation(self, mic_data: torch.Tensor,
                              predicted_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, n_channels, time_steps = mic_data.shape
        device = mic_data.device

        # Calculate delays
        primary_delays = predicted_distances * self.samples_per_meter
        secondary_delays = primary_delays * 2

        # Process analytic signal (unchanged)
        analytic_signals = []
        for b in range(batch_size):
            channel_signals = []
            for c in range(n_channels):
                single_channel = mic_data[b, c]
                stft = torch.stft(single_channel,
                                  n_fft=256,
                                  hop_length=128,
                                  window=torch.hann_window(256, device=device),
                                  return_complex=True)
                channel_signals.append(stft)
            analytic_signals.append(torch.stack(channel_signals))
        analytic_signal = torch.stack(analytic_signals)
        envelope = torch.abs(analytic_signal).max(dim=2)[0]

        reflection_scores = torch.zeros(batch_size, device=device)
        valid_primaries = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            valid_mask = primary_delays[i] > 0
            valid_primaries[i] = valid_mask.sum()

            if valid_mask.any():
                valid_secondary = secondary_delays[i][valid_mask]
                indices = (valid_secondary // 128).long()
                valid_indices = indices[indices < envelope.shape[-1]]

                if valid_indices.numel() > 0:
                    signal_max = torch.max(envelope[i])
                    if signal_max > 0:
                        for idx in valid_indices:
                            window = 5
                            start = max(0, idx - window)
                            end = min(envelope.shape[-1], idx + window + 1)
                            window_max = torch.max(envelope[i, :, start:end])
                            if window_max > 0.1 * signal_max:
                                reflection_scores[i] += 1

        # Safe division for detection rate
        detection_rate = torch.where(valid_primaries > 0,
                                     reflection_scores / valid_primaries,
                                     torch.zeros_like(reflection_scores))

        return {
            'multipath_detections': reflection_scores.cpu(),
            'multipath_detection_rate': detection_rate.cpu()
        }

    def __call__(self, mic_data: torch.Tensor,
                 predicted_distances: torch.Tensor,
                 valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        metrics = {}
        metrics.update(self.time_of_flight_validation(mic_data, predicted_distances))
        metrics.update(self.amplitude_decay_validation(mic_data, predicted_distances))
        metrics.update(self.spatial_consistency_validation(predicted_distances, valid_mask))
        metrics.update(self.multi_path_validation(mic_data, predicted_distances))

        # Calculate overall physics score with safe handling
        weights = {
            'tof_violation_rate': 0.4,
            'mean_amplitude_error': 0.2,
            'spatial_violation_rate': 0.3,
            'multipath_detection_rate': 0.1
        }

        # Ensure all metrics are on CPU and handled safely
        overall_score = sum(weights[k] * (1 - metrics[k].float().mean())
                            for k in weights.keys())

        metrics['overall_physics_score'] = overall_score.cpu()

        return metrics


class ConfidenceCalibrationMetrics:
    def __init__(self, num_bins=10, distance_threshold=0.005):
        """
        Initialize confidence calibration tracking.

        Args:
            num_bins: Number of confidence bins to use
            distance_threshold: Maximum distance error to consider prediction "correct" (in meters)
        """
        self.num_bins = num_bins
        self.distance_threshold = distance_threshold
        self.bin_edges = torch.linspace(0, 1, num_bins + 1)
        self.predictions = []
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.predictions = []  # Store (confidence, correctness, uncertainty) tuples

    def update(self, predicted_distances, validity_scores, uncertainty_scores,
               target_distances, target_mask):
        """
        Update calibration statistics with batch of predictions.

        Args:
            predicted_distances: Model's distance predictions
            validity_scores: Model's validity scores (confidence)
            uncertainty_scores: Model's uncertainty estimates
            target_distances: Ground truth distances
            target_mask: Ground truth validity mask
        """
        # Only consider points with ground truth
        valid_mask = target_mask > 0

        if not valid_mask.any():
            return

        # Calculate correctness (within threshold)
        errors = torch.abs(predicted_distances - target_distances)
        correctness = (errors < self.distance_threshold).float()

        # Gather valid predictions
        valid_confidences = validity_scores[valid_mask]
        valid_correctness = correctness[valid_mask]
        valid_uncertainty = uncertainty_scores[valid_mask]

        # Store results
        self.predictions.extend(zip(
            valid_confidences.cpu().numpy(),
            valid_correctness.cpu().numpy(),
            valid_uncertainty.cpu().numpy()
        ))

    def compute_metrics(self):
        """
        Compute calibration metrics from accumulated statistics.
        """
        if not self.predictions:
            return {}

        # Convert to numpy arrays
        confidences, correctness, uncertainties = zip(*self.predictions)
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        uncertainties = np.array(uncertainties)

        # Bin predictions by confidence
        bin_indices = np.digitize(confidences, self.bin_edges.cpu().numpy()) - 1

        # Calculate statistics per bin
        metrics = {
            'ece': 0.0,  # Expected Calibration Error
            'mce': 0.0,  # Maximum Calibration Error
            'bin_stats': []
        }

        total_samples = len(confidences)

        for bin_idx in range(self.num_bins):
            bin_mask = bin_indices == bin_idx
            if not np.any(bin_mask):
                continue

            # Compute bin statistics
            bin_confidence = confidences[bin_mask].mean()
            bin_accuracy = correctness[bin_mask].mean()
            bin_uncertainty = uncertainties[bin_mask].mean()
            bin_count = np.sum(bin_mask)

            # Update ECE
            metrics['ece'] += (bin_count / total_samples) * abs(bin_confidence - bin_accuracy)

            # Update MCE
            metrics['mce'] = max(metrics['mce'], abs(bin_confidence - bin_accuracy))

            # Store bin statistics
            metrics['bin_stats'].append({
                'confidence': bin_confidence,
                'accuracy': bin_accuracy,
                'uncertainty': bin_uncertainty,
                'count': bin_count
            })

        # Calculate uncertainty correlation
        metrics['uncertainty_correlation'] = np.corrcoef(uncertainties,
                                                         np.abs(1 - correctness))[0, 1]

        return metrics


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [3, 7, 11, 15, 19]
        # Ensure divisible number of channels
        self.channels_per_branch = out_channels // len(kernel_sizes)
        # Adjust output channels to be perfectly divisible
        self.total_channels = self.channels_per_branch * len(kernel_sizes)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, self.channels_per_branch, k, padding=k // 2),
                nn.BatchNorm1d(self.channels_per_branch),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        weight = self.fc(avg).view(b, c, 1)
        return x * weight


class DilatedConvPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Calculate intermediate channel size
        mid_channels = (in_channels + out_channels) // 2

        # Create parallel dilated convolutions with increasing dilation
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=3,
                          dilation=2 ** i, padding=2 ** i),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for i in range(4)  # Dilations: 1, 2, 4, 8
        ])

        # 1x1 conv to combine and match output channels
        self.combine = nn.Conv1d(mid_channels * 4, out_channels, 1)

    def forward(self, x):
        # Process through dilated convolutions
        dilated_outputs = [conv(x) for conv in self.dilated_convs]
        # Concatenate along channel dimension
        combined = torch.cat(dilated_outputs, dim=1)
        # Combine to match output channels
        return self.combine(combined)


class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            bidirectional=True, batch_first=True)
        self.norm = nn.LayerNorm(input_size)
        self.proj = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        identity = x
        out, _ = self.lstm(x)
        out = self.proj(out)
        return self.norm(out + identity)


class SLAMOutput(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden = input_dim // 2

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.distance = nn.Linear(hidden, output_dim)
        self.validity = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.Sigmoid()
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        features = self.shared(x)
        dist = self.distance(features)
        valid = self.validity(features)
        uncert = self.uncertainty(features)

        # Mask distances with validity
        masked_dist = dist * (valid > 0.5).float()

        return {
            'distances': masked_dist,
            'raw_distances': dist,
            'validity': valid,
            'uncertainty': uncert
        }


class CurriculumScheduler:
    def __init__(self, train_dataset, initial_threshold=0.7, final_threshold=0.0,
                 warmup_epochs=10, phases=4):
        self.dataset = train_dataset
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.warmup_epochs = warmup_epochs
        self.phases = phases

        # Calculate complexity scores for each sample
        self.complexity_scores = self._calculate_complexity_scores()

        # Calculate threshold schedule
        self.thresholds = np.linspace(initial_threshold, final_threshold, phases)

    def _calculate_complexity_scores(self):
        """Calculate complexity score for each training sample."""
        scores = []
        for i in range(len(self.dataset)):
            mic_data, distances, mask = self.dataset[i]

            # 1. Number of valid measurements (more = harder)
            valid_ratio = mask.mean()

            # 2. Variation in distances (more variation = harder)
            distance_std = distances[mask > 0].std() if mask.sum() > 0 else 0

            # 3. Signal complexity (using FFT amplitude variation)
            # Fix: Add return_complex=True and proper handling of complex output
            try:
                # Handle different mic_data shapes
                if len(mic_data.shape) == 1:
                    mic_data = mic_data.unsqueeze(0)  # Add channel dimension if needed

                # Perform STFT with proper parameters
                stft_out = torch.stft(
                    mic_data,
                    n_fft=256,
                    hop_length=128,
                    window=torch.hann_window(256, device=mic_data.device),
                    return_complex=True
                )

                # Calculate complexity using magnitude spectrum
                fft_complexity = torch.abs(stft_out).std()

            except RuntimeError as e:
                print(f"Warning: STFT failed, using fallback complexity measure. Error: {e}")
                # Fallback: Use simple standard deviation of the signal
                fft_complexity = mic_data.std()

            # Combine factors (can be weighted differently)
            complexity = (0.4 * valid_ratio +
                          0.3 * distance_std +
                          0.3 * fft_complexity)
            scores.append(complexity)

        return torch.tensor(scores)

    def get_sample_weights(self, epoch):
        """Get sample weights based on current epoch."""
        if epoch < self.warmup_epochs:
            threshold = self.initial_threshold
        else:
            phase = min((epoch - self.warmup_epochs) //
                        ((self.phases - 1) * self.warmup_epochs),
                        self.phases - 1)
            threshold = self.thresholds[phase]

        # Create binary weights based on threshold
        weights = (self.complexity_scores <= threshold).float()
        # Ensure some samples are always included
        weights = weights + 0.1
        # Normalize weights
        weights = weights / weights.sum()

        return weights


class CurriculumSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, curriculum_scheduler, batch_size):
        self.data_source = data_source
        self.curriculum_scheduler = curriculum_scheduler
        self.batch_size = batch_size
        self.epoch = 0

    def __iter__(self):
        # Get weights for current epoch
        weights = self.curriculum_scheduler.get_sample_weights(self.epoch)
        # Get current batch size from the batch sampler
        current_batch_size = self.batch_size
        # Ensure we sample enough indices for complete batches
        num_samples = len(self.data_source)
        num_batches = (num_samples + current_batch_size - 1) // current_batch_size
        total_samples = num_batches * current_batch_size

        indices = torch.multinomial(weights,
                                    total_samples,
                                    replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch


class CyclicBatchSizer:
    def __init__(self, min_batch_size, max_batch_size,
                 cycle_length=10, decay_rate=0.9):
        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.cycle_length = cycle_length
        self.decay_rate = decay_rate

    def get_batch_size(self, epoch):
        """Calculate batch size for given epoch."""
        # Calculate cycle progress
        cycle = epoch // self.cycle_length
        cycle_epoch = epoch % self.cycle_length

        # Calculate progress within cycle (0 to 1)
        progress = cycle_epoch / self.cycle_length

        # Calculate decay factor
        decay = self.decay_rate ** cycle

        # Calculate current range
        current_max = self.max_batch * decay
        current_range = current_max - self.min_batch

        # Cosine annealing within cycle
        cos_factor = 0.5 * (1 + np.cos(progress * np.pi))

        # Calculate batch size
        batch_size = self.min_batch + current_range * cos_factor

        return int(batch_size)


class NoiseScheduler:
    """Manages noise parameters during training."""

    def __init__(self,
                 initial_gaussian_std=0.02,
                 final_gaussian_std=0.01,
                 initial_impulse_prob=0.001,
                 final_impulse_prob=0.0005,
                 initial_impulse_strength=0.1,
                 final_impulse_strength=0.05,
                 initial_phase_std=0.5,
                 final_phase_std=0.2):
        self.initial_params = {
            'gaussian_std': initial_gaussian_std,
            'impulse_prob': initial_impulse_prob,
            'impulse_strength': initial_impulse_strength,
            'phase_noise_std': initial_phase_std
        }
        self.final_params = {
            'gaussian_std': final_gaussian_std,
            'impulse_prob': final_impulse_prob,
            'impulse_strength': final_impulse_strength,
            'phase_noise_std': final_phase_std
        }

    def get_noise_params(self, epoch, total_epochs):
        """Calculate noise parameters for current epoch."""
        progress = min(epoch / (total_epochs * 0.8), 1.0)  # Transition over first 80% of training

        current_params = {}
        for key in self.initial_params:
            initial = self.initial_params[key]
            final = self.final_params[key]
            current_params[key] = initial + (final - initial) * progress

        return current_params


# Main Model
class AcousticMapper(nn.Module):
    def __init__(self, input_channels=7, output_size=61):
        super().__init__()
        self.l2_lambda = 0.001

        # Feature extraction (unchanged)
        self.feature_net = nn.Sequential(
            MultiScaleConv(input_channels, 100),  # Changed from 96 to 100 (divisible by 5)
            ChannelAttention(100),
            nn.Conv1d(100, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Original temporal path
        self.temporal_net = nn.Sequential(
            nn.Conv1d(128, 160, 5, stride=2, padding=2),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(160, 192, 5, stride=2, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # New dilated path
        self.dilated_net = DilatedConvPath(128, 192)

        # Channel attention for combining paths
        self.path_attention = ChannelAttention(192 * 2)

        # 1x1 conv to combine paths
        self.path_combine = nn.Conv1d(192 * 2, 192, 1)

        # Rest remains unchanged
        self.sequence_net = ResidualLSTM(192, 96)
        self.output_net = SLAMOutput(192, output_size)

    def forward(self, x):
        # Feature extraction
        x = self.feature_net(x)

        # Process through both paths
        temporal_features = self.temporal_net(x)
        dilated_features = self.dilated_net(x)

        # Match temporal dimensions (dilated path has no stride)
        dilated_features = F.adaptive_max_pool1d(
            dilated_features,
            temporal_features.size(-1)
        )

        # Combine paths
        combined = torch.cat([temporal_features, dilated_features], dim=1)
        combined = self.path_attention(combined)
        x = self.path_combine(combined)

        # Sequence processing
        x = x.transpose(1, 2)
        x = self.sequence_net(x)

        # Global pooling
        x = x.mean(1)

        # Generate outputs
        return self.output_net(x)

    def get_l2_loss(self):
        """Calculate L2 regularization loss."""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_lambda * l2_loss

    def print_param_count(self):
        """Update parameter counting to include new components"""
        print("Parameter count per layer:")
        for name, module in self.named_children():
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name}: {param_count:,} parameters")

            # Detailed count for temporal paths
            if name in ['temporal_net', 'dilated_net']:
                print(f"    Details for {name}:")
                for sub_name, sub_module in module.named_children():
                    sub_count = sum(p.numel() for p in sub_module.parameters() if p.requires_grad)
                    print(f"      {sub_name}: {sub_count:,} parameters")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal trainable parameters: {total_params:,}")


# Loss Function
class MappingLoss(nn.Module):
    def __init__(self, spatial_lambda=0.1):  # Add weight parameter
        super().__init__()
        self.spatial_lambda = spatial_lambda

    def spatial_consistency_loss(self, distances, valid_mask):
        """
        Calculate loss based on spatial consistency between adjacent measurements.

        Args:
            distances: Predicted distances tensor (batch_size, num_measurements)
            valid_mask: Binary mask of valid measurements (batch_size, num_measurements)
        """
        # Calculate differences between adjacent measurements
        diffs = torch.abs(distances[:, 1:] - distances[:, :-1])

        # Only consider pairs where both measurements are valid
        valid_pairs = valid_mask[:, 1:] * valid_mask[:, :-1]

        # Calculate mean difference where valid, avoid division by zero
        valid_sum = valid_pairs.sum()
        if valid_sum > 0:
            return (diffs * valid_pairs).sum() / valid_sum
        return torch.tensor(0.0, device=distances.device)

    def forward(self, pred, target_dist, target_mask):
        # Unpack predictions (existing code)
        dist = pred['raw_distances']
        valid = pred['validity']
        uncert = pred['uncertainty']

        # Valid measurements mask (existing code)
        valid_mask = (target_mask > 0).float()

        # Existing loss components (existing code)
        dist_loss = torch.where(valid_mask == 1,
                                F.mse_loss(dist, target_dist, reduction='none'),
                                torch.zeros_like(dist)
                                ).mean()

        valid_loss = F.binary_cross_entropy(valid, valid_mask)

        uncert_loss = torch.where(valid_mask == 1,
                                  (torch.exp(-uncert) * (dist - target_dist).pow(2) + uncert),
                                  torch.zeros_like(uncert)
                                  ).mean()

        invalid_loss = torch.where(valid_mask == 0,
                                   torch.abs(dist),
                                   torch.zeros_like(dist)
                                   ).mean()

        # Add spatial consistency loss
        spatial_loss = self.spatial_consistency_loss(dist, valid_mask)

        # Store individual loss components
        loss_components = {
            'distance': dist_loss.item(),
            'validity': valid_loss.item(),
            'uncertainty': uncert_loss.item(),
            'invalid': invalid_loss.item(),
            'spatial': spatial_loss.item()  # Add to components
        }

        # Combine losses with spatial consistency
        total_loss = (dist_loss +
                      valid_loss +
                      0.1 * uncert_loss +
                      0.1 * invalid_loss +
                      self.spatial_lambda * spatial_loss)

        return total_loss, loss_components


# Training Functions
def train_epoch(model, loader, criterion, optimizer, device, noise_scheduler, epoch, total_epochs):
    model.train()
    total_losses = {
        'total': 0,
        'distance': 0,
        'validity': 0,
        'uncertainty': 0,
        'invalid': 0,
        'spatial': 0,
        'l2': 0
    }

    num_batches = 0

    # Get noise parameters for current epoch
    noise_params = noise_scheduler.get_noise_params(epoch, total_epochs)

    for X, y_dist, y_mask in loader:
        X, y_dist, y_mask = [t.to(device) for t in (X, y_dist, y_mask)]

        # Add noise to input data
        X_noisy = add_acoustic_noise(X, noise_params)

        optimizer.zero_grad()
        pred = model(X_noisy)
        loss, components = criterion(pred, y_dist, y_mask)

        # Add L2 regularization
        l2_loss = model.get_l2_loss()
        total_loss = loss + l2_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update running losses
        total_losses['total'] += total_loss.item()
        total_losses['l2'] += l2_loss.item()
        for k, v in components.items():
            total_losses[k] += v

        num_batches += 1

    # Average losses
    for k in total_losses:
        total_losses[k] /= num_batches

    return total_losses


def validate(model, loader, criterion, device, physics_metrics):
    calibration_tracker = ConfidenceCalibrationMetrics()
    model.eval()
    total_losses = {
        'total': 0,
        'distance': 0,
        'validity': 0,
        'uncertainty': 0,
        'invalid': 0,
        'spatial': 0,
        'l2': 0
    }

    physics_scores = []

    with torch.no_grad():
        for X, y_dist, y_mask in loader:
            X, y_dist, y_mask = [t.to(device) for t in (X, y_dist, y_mask)]
            pred = model(X)
            loss, components = criterion(pred, y_dist, y_mask)
            l2_loss = model.get_l2_loss()

            calibration_tracker.update(
                pred['distances'],
                pred['validity'],
                pred['uncertainty'],
                y_dist,
                y_mask
            )

            total_losses['total'] += (loss + l2_loss).item()
            total_losses['l2'] += l2_loss.item()
            for k, v in components.items():
                total_losses[k] += v

            # Add physics validation
            physics_results = physics_metrics(X, pred['distances'], y_mask)
            physics_scores.append(physics_results)

    # Average losses
    for k in total_losses:
        total_losses[k] /= len(loader)

    # Aggregate physics scores
    avg_physics_scores = {}
    for key in physics_scores[0].keys():
        avg_physics_scores[key] = sum(s[key] for s in physics_scores) / len(physics_scores)

    calibration_metrics = calibration_tracker.compute_metrics()

    return total_losses, avg_physics_scores, calibration_metrics


def train_model(model, train_loader, val_loader, device,
                batch_sizer, curriculum_sampler, epochs=100):
    criterion = MappingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduler()

    # Initialize physics metrics
    physics_metrics = PhysicsMetrics()

    best_loss = float('inf')
    patience = 200
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        # Update curriculum sampler epoch
        curriculum_sampler.set_epoch(epoch)

        # Get new batch size
        new_batch_size = batch_sizer.get_batch_size(epoch)

        # Update DataLoader batch size
        train_loader.batch_sampler.batch_size = new_batch_size

        # Train epoch
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device,
                                   noise_scheduler, epoch, epochs)
        # Updated validation call
        val_losses, physics_scores, calibration_metrics = validate(model, val_loader,
                                                                   criterion, device, physics_metrics)

        # Step the learning rate scheduler
        scheduler.step()

        # Store results
        history.append({
            'train': train_losses,
            'val': val_losses,
            'physics': physics_scores,
            'calibration': calibration_metrics,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Print detailed metrics
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("Training Losses:")
        print(f"  Total: {train_losses['total']:.4f}")
        print(f"  Distance: {train_losses['distance']:.4f}")
        print(f"  Validity: {train_losses['validity']:.4f}")
        print(f"  Uncertainty: {train_losses['uncertainty']:.4f}")
        print(f"  Invalid: {train_losses['invalid']:.4f}")
        print(f"  L2: {train_losses['l2']:.4f}")

        print("\nValidation Losses:")
        print(f"  Total: {val_losses['total']:.4f}")
        print(f"  Distance: {val_losses['distance']:.4f}")
        print(f"  Validity: {val_losses['validity']:.4f}")
        print(f"  Uncertainty: {val_losses['uncertainty']:.4f}")
        print(f"  Invalid: {val_losses['invalid']:.4f}")
        print(f"  L2: {val_losses['l2']:.4f}")

        # Print metrics including physics scores
        print(f"\nPhysics Validation Metrics:")
        print(f"  Overall Score: {physics_scores['overall_physics_score'].item():.4f}")
        print(f"  ToF Violations: {physics_scores['tof_violation_rate'].mean().item():.4f}")
        print(f"  Amplitude Error: {physics_scores['mean_amplitude_error'].item():.4f}")
        print(f"  Spatial Violations: {physics_scores['spatial_violation_rate'].mean().item():.4f}")

        print("\nCalibration Metrics:")
        print(f"  ECE: {calibration_metrics['ece']:.4f}")
        print(f"  MCE: {calibration_metrics['mce']:.4f}")
        print(f"  Uncertainty Correlation: {calibration_metrics['uncertainty_correlation']:.4f}")

        print(f"\nLearning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'best_model_slam_mega.pt')

        # Early stopping check
        if val_losses['total'] < best_loss:
            best_loss = val_losses['total']
            torch.save(model.state_dict(), 'best_model_slam_mega.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered")
                break

    return history


def plot_training_history(history):
    epochs = range(1, len(history) + 1)

    # Plot 1: Total Loss (unchanged)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [h['train']['total'] for h in history], 'b-', label='Train Total')
    plt.plot(epochs, [h['val']['total'] for h in history], 'r-', label='Val Total')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Training and Validation Components
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Update components list to include spatial loss
    components = ['distance', 'validity', 'uncertainty', 'invalid', 'spatial']

    # Rest of plotting code remains the same
    for comp in components:
        ax1.plot(epochs, [h['train'][comp] for h in history],
                 label=f'Train {comp.capitalize()}')
    ax1.set_title('Training Loss Components')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    for comp in components:
        ax2.plot(epochs, [h['val'][comp] for h in history],
                 label=f'Val {comp.capitalize()}')
    ax2.set_title('Validation Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_physics_metrics(history):
    plt.figure(figsize=(12, 8))
    metrics = ['tof_violation_rate', 'mean_amplitude_error',
               'spatial_violation_rate', 'multipath_detection_rate']

    colors = ['blue', 'red', 'green', 'purple']  # Different color for each metric

    for metric, color in zip(metrics, colors):
        # Extract the metric values and ensure they're scalars
        values = []
        for h in history:
            metric_value = h['physics'][metric]
            if torch.is_tensor(metric_value):
                # If it's a tensor, take the mean and convert to scalar
                metric_value = metric_value.mean().item()
            values.append(metric_value)

        plt.plot(values, label=metric.replace('_', ' ').title(), color=color)

    plt.title('Physics Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_path_contributions(model, loader, device):
    """Plot relative contributions of temporal and dilated paths"""
    model.eval()
    temporal_norms = []
    dilated_norms = []

    with torch.no_grad():
        for X, _, _ in loader:
            X = X.to(device)
            features = model.feature_net(X)

            temporal = model.temporal_net(features)
            dilated = model.dilated_net(features)

            temporal_norms.append(torch.norm(temporal, dim=1).mean().item())
            dilated_norms.append(torch.norm(dilated, dim=1).mean().item())

    plt.figure(figsize=(10, 5))
    plt.plot(temporal_norms, label='Temporal Path')
    plt.plot(dilated_norms, label='Dilated Path')
    plt.title('Path Contribution Analysis')
    plt.xlabel('Batch')
    plt.ylabel('Feature Norm')
    plt.legend()
    plt.show()


# Add frequency content visualization
def plot_frequency_content(data, sample_rate=5000000):
    plt.figure(figsize=(10, 6))
    freqs = np.fft.rfftfreq(data.shape[-1], d=1 / sample_rate)
    fft = np.fft.rfft(data.mean(axis=0))
    plt.semilogy(freqs / 1000, np.abs(fft))  # Convert to kHz
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude')
    plt.title('Average Frequency Content')
    plt.grid(True)
    plt.show()


def plot_calibration_results(history):
    plt.figure(figsize=(15, 5))

    # Plot 1: Reliability Diagram
    plt.subplot(131)
    latest_stats = history[-1]['calibration']['bin_stats']
    confidences = [s['confidence'] for s in latest_stats]
    accuracies = [s['accuracy'] for s in latest_stats]
    counts = [s['count'] for s in latest_stats]

    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    plt.scatter(confidences, accuracies, s=counts, alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')

    # Plot 2: ECE over time
    plt.subplot(132)
    ece_values = [h['calibration']['ece'] for h in history]
    plt.plot(ece_values)
    plt.xlabel('Epoch')
    plt.ylabel('ECE')
    plt.title('Expected Calibration Error')

    # Plot 3: Uncertainty Correlation
    plt.subplot(133)
    corr_values = [h['calibration']['uncertainty_correlation'] for h in history]
    plt.plot(corr_values)
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.title('Uncertainty-Error Correlation')

    plt.tight_layout()
    plt.show()


# Main execution
def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process data
    X, y_dist, y_mask = load_data("simulation_data/simulation_data_parallel.h5", 19600)

    # Convert to torch tensor on CPU first
    X = torch.from_numpy(X).float()

    # Apply scaling including frequency normalization
    X_scaled, y_dist_scaled = scale_data(X, y_dist)

    # Split data
    split = train_test_split(X_scaled.cpu().numpy(), y_dist_scaled, y_mask, test_size=0.2, random_state=42)
    X_train, X_val, y_dist_train, y_dist_val, y_mask_train, y_mask_val = split

    # Create dataloaders
    train_data = TensorDataset(*map(torch.FloatTensor, [X_train, y_dist_train, y_mask_train]))
    val_data = TensorDataset(*map(torch.FloatTensor, [X_val, y_dist_val, y_mask_val]))

    # Initialize curriculum scheduler
    curriculum_scheduler = CurriculumScheduler(
        train_data,
        initial_threshold=0.7,
        final_threshold=0.0,
        warmup_epochs=10,
        phases=4
    )

    # Initialize batch sizer
    batch_sizer = CyclicBatchSizer(
        min_batch_size=8,
        max_batch_size=32,
        cycle_length=10,
        decay_rate=0.9
    )

    # Create sampler for curriculum learning
    sampler = CurriculumSampler(train_data, curriculum_scheduler, 16)

    # Modify DataLoader to use sampler
    train_loader = DataLoader(
        train_data,
        batch_size=16,  # This will be adjusted dynamically
        sampler=sampler,
        num_workers=4
    )

    val_loader = DataLoader(val_data, batch_size=16)

    # Initialize and train model
    model = AcousticMapper().to(device)
    model.print_param_count()
    history = train_model(model, train_loader, val_loader, device, batch_sizer, sampler, epochs=1000)

    # Plot results
    plot_training_history(history)
    plot_path_contributions(model, val_loader, device)
    plot_physics_metrics(history)
    plot_calibration_results(history)

    # Plot frequency content of first batch
    plot_frequency_content(X_scaled[0])


if __name__ == "__main__":
    main()
import os
import csv
import numpy as np
import argparse
from datetime import datetime
from typing import List, Optional

def generate_eeg_signal(duration_sec: float = 1.028, sampling_rate: int = 250, state: str = 'relaxed', channels: int = 16):
    """
    Generate realistic EEG signals with state-specific frequency characteristics.
    
    States and their characteristics (unified with training script):
    - relaxed: High alpha (8-13 Hz), low beta, moderate theta
    - focused: High beta (15-25 Hz), moderate alpha, low theta
    - stressed: Very high beta (20-40 Hz), high gamma, low alpha, elevated theta
    """
    num_samples = int(duration_sec * sampling_rate)
    time = np.linspace(0, duration_sec, num_samples)
    
    # Initialize signal for all channels
    signals = np.zeros((num_samples, channels))
    
    # Standard EEG bands
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # State-specific amplitude profiles (arbitrary units)
    STATE_PROFILES = {
        'relaxed': {'alpha': (20, 35), 'beta': (3, 10), 'theta': (8, 15), 'delta': (2, 6), 'gamma': (1, 4)},
        'focused': {'alpha': (10, 18), 'beta': (18, 30), 'theta': (3, 8), 'delta': (1, 5), 'gamma': (5, 12)},
        'stressed': {'alpha': (3, 10), 'beta': (30, 50), 'theta': (10, 20), 'delta': (4, 8), 'gamma': (15, 30)}
    }
    
    profile = STATE_PROFILES.get(state, STATE_PROFILES['relaxed'])
    
    for ch in range(channels):
        # Base pink noise (1/f) simulation
        signal = np.zeros(num_samples)
        for i in range(1, num_samples // 2):
            freq = i * (sampling_rate / num_samples)
            amplitude = 1.0 / (freq ** 0.8) # Pink-ish noise
            signal += amplitude * np.sin(2 * np.pi * freq * time + np.random.uniform(0, 2*np.pi))
        
        # Normalize baseline noise
        signal = (signal - np.mean(signal)) / np.std(signal) * 5.0
        
        # Add state-specific band peaks
        for band, (low, high) in BANDS.items():
            amp_min, amp_max = profile[band]
            
            # Generate 2-3 sine waves within the band for more realism than a single frequency
            num_sines = 3
            for _ in range(num_sines):
                freq = np.random.uniform(low, high)
                amp = np.random.uniform(amp_min, amp_max) / num_sines
                signal += amp * np.sin(2 * np.pi * freq * time + np.random.uniform(0, 2*np.pi))
        
        # Add channel-specific variations
        if ch < 4:  # Frontal - more frontal alpha/beta variation
            signal *= np.random.uniform(0.9, 1.1)
        elif ch >= 12:  # Occipital - more alpha
            if state == 'relaxed':
                signal *= 1.2
        
        # Add some random high-frequency "spikes"
        if np.random.random() > 0.9:
            spike_idx = np.random.randint(0, num_samples)
            signal[spike_idx:spike_idx+5] += np.random.uniform(50, 100)
            
        signals[:, ch] = signal
    
    return signals

def generate_dataset(
    state: str, 
    num_epochs: int, 
    duration_sec: float, 
    sampling_rate: int, 
    output_dir: str,
    channel_names: List[str]
):
    """Generate and save a dataset for a specific state."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"test_{state}_{num_epochs}epochs.csv")
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + channel_names)
        
        timestamp = 0.0
        for epoch in range(num_epochs):
            signals = generate_eeg_signal(duration_sec, sampling_rate, state, len(channel_names))
            
            for sample_idx in range(signals.shape[0]):
                row = [f"{timestamp:.4f}"] + [f"{val:.2f}" for val in signals[sample_idx]]
                writer.writerow(row)
                timestamp += 1.0 / sampling_rate
    
    print(f"✓ Generated {state} dataset: {filename} ({num_epochs} epochs)")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Generate realistic EEG test datasets.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs per state (default: 50)")
    parser.add_argument("--rate", type=int, default=250, help="Sampling rate in Hz (default: 250)")
    parser.add_argument("--duration", type=float, default=1.028, help="Epoch duration in seconds (default: 1.028)")
    parser.add_argument("--outdir", type=str, default="data/testing_data", help="Output directory")
    parser.add_argument("--mixed", action="store_true", help="Generate a mixed states dataset")
    parser.add_argument("--states", nargs="+", default=["relaxed", "focused", "stressed"], help="States to generate")
    parser.add_argument("--channels", type=int, default=16, help="Number of channels (max 16 for standard names)")
    
    args = parser.parse_args()
    
    all_channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", 
                    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"]
    channels = all_channels[:args.channels]
    
    print("=" * 60)
    print("EEG Test Dataset Generator")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    if args.mixed:
        filename = os.path.join(args.outdir, "test_mixed_states.csv")
        os.makedirs(args.outdir, exist_ok=True)
        
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + channels)
            
            timestamp = 0.0
            for state in args.states:
                print(f"  Generating {args.epochs} epochs of '{state}'...")
                for _ in range(args.epochs):
                    signals = generate_eeg_signal(args.duration, args.rate, state, len(channels))
                    for sample_idx in range(signals.shape[0]):
                        row = [f"{timestamp:.4f}"] + [f"{val:.2f}" for val in signals[sample_idx]]
                        writer.writerow(row)
                        timestamp += 1.0 / args.rate
        print(f"✓ Generated mixed states dataset: {filename}")
    else:
        for state in args.states:
            generate_dataset(state, args.epochs, args.duration, args.rate, args.outdir, channels)
            
    print("=" * 60)
    print("✓ Done!")

if __name__ == "__main__":
    main()

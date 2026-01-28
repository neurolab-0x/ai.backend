"""
Generate realistic EEG test datasets for different mental states.
Creates distinct frequency patterns for relaxed, focused, and stressed states.
"""
import os
import csv
import numpy as np
from datetime import datetime

def generate_eeg_signal(duration_sec=1.0, sampling_rate=250, state='relaxed', channels=16):
    """
    Generate realistic EEG signals with state-specific frequency characteristics.
    
    States:
    - relaxed: High alpha (8-13 Hz), low beta
    - focused: High beta (13-30 Hz), moderate alpha
    - stressed: High beta, high gamma (30-50 Hz), low alpha
    """
    num_samples = int(duration_sec * sampling_rate)
    time = np.linspace(0, duration_sec, num_samples)
    
    # Initialize signal for all channels
    signals = np.zeros((num_samples, channels))
    
    for ch in range(channels):
        # Base noise
        signal = np.random.normal(0, 2, num_samples)
        
        if state == 'relaxed':
            # High alpha (8-13 Hz) - relaxed, eyes closed
            alpha_freq = np.random.uniform(8, 13)
            alpha_amp = np.random.uniform(15, 25)
            signal += alpha_amp * np.sin(2 * np.pi * alpha_freq * time + np.random.uniform(0, 2*np.pi))
            
            # Low beta (13-20 Hz)
            beta_freq = np.random.uniform(13, 20)
            beta_amp = np.random.uniform(3, 8)
            signal += beta_amp * np.sin(2 * np.pi * beta_freq * time + np.random.uniform(0, 2*np.pi))
            
            # Theta (4-8 Hz) - drowsiness
            theta_freq = np.random.uniform(4, 8)
            theta_amp = np.random.uniform(5, 10)
            signal += theta_amp * np.sin(2 * np.pi * theta_freq * time + np.random.uniform(0, 2*np.pi))
            
        elif state == 'focused':
            # Moderate alpha
            alpha_freq = np.random.uniform(9, 12)
            alpha_amp = np.random.uniform(8, 15)
            signal += alpha_amp * np.sin(2 * np.pi * alpha_freq * time + np.random.uniform(0, 2*np.pi))
            
            # High beta (15-25 Hz) - concentration
            beta_freq = np.random.uniform(15, 25)
            beta_amp = np.random.uniform(12, 20)
            signal += beta_amp * np.sin(2 * np.pi * beta_freq * time + np.random.uniform(0, 2*np.pi))
            
            # Low gamma (30-40 Hz) - cognitive processing
            gamma_freq = np.random.uniform(30, 40)
            gamma_amp = np.random.uniform(3, 7)
            signal += gamma_amp * np.sin(2 * np.pi * gamma_freq * time + np.random.uniform(0, 2*np.pi))
            
        elif state == 'stressed':
            # Low alpha - anxiety
            alpha_freq = np.random.uniform(8, 11)
            alpha_amp = np.random.uniform(3, 8)
            signal += alpha_amp * np.sin(2 * np.pi * alpha_freq * time + np.random.uniform(0, 2*np.pi))
            
            # High beta (20-30 Hz) - anxiety, stress
            beta_freq = np.random.uniform(20, 30)
            beta_amp = np.random.uniform(15, 25)
            signal += beta_amp * np.sin(2 * np.pi * beta_freq * time + np.random.uniform(0, 2*np.pi))
            
            # High gamma (35-50 Hz) - high arousal
            gamma_freq = np.random.uniform(35, 50)
            gamma_amp = np.random.uniform(8, 15)
            signal += gamma_amp * np.sin(2 * np.pi * gamma_freq * time + np.random.uniform(0, 2*np.pi))
            
            # Add more noise for stressed state
            signal += np.random.normal(0, 5, num_samples)
        
        # Add channel-specific variations
        if ch < 4:  # Frontal channels (Fp1, Fp2, F3, F4)
            signal *= np.random.uniform(0.9, 1.1)
        elif ch < 8:  # Central and Parietal (C3, C4, P3, P4)
            signal *= np.random.uniform(0.95, 1.05)
        else:  # Occipital and Temporal
            signal *= np.random.uniform(0.85, 1.15)
        
        signals[:, ch] = signal
    
    return signals

def generate_test_dataset(state='relaxed', num_epochs=50, duration_sec=1.028, 
                         sampling_rate=250, output_dir='data/testing_data'):
    """
    Generate a complete test dataset for a specific mental state.
    
    Args:
        state: 'relaxed', 'focused', or 'stressed'
        num_epochs: Number of 1-second epochs to generate
        duration_sec: Duration of each epoch (default 1.028s = 257 samples at 250Hz)
        sampling_rate: Sampling rate in Hz
        output_dir: Directory to save the file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Channel names (standard 10-20 system)
    channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", 
                "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"]
    
    filename = f"{output_dir}/test_{state}_{num_epochs}epochs.csv"
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["timestamp"] + channels)
        
        timestamp = 0.0
        for epoch in range(num_epochs):
            # Generate EEG signal for this epoch
            signals = generate_eeg_signal(duration_sec, sampling_rate, state, len(channels))
            
            # Write each sample
            for sample_idx in range(signals.shape[0]):
                row = [f"{timestamp:.4f}"] + [f"{val:.2f}" for val in signals[sample_idx]]
                writer.writerow(row)
                timestamp += 1.0 / sampling_rate
    
    print(f"✓ Generated {state} dataset: {filename}")
    print(f"  - {num_epochs} epochs × {int(duration_sec * sampling_rate)} samples = {num_epochs * int(duration_sec * sampling_rate)} total samples")
    return filename

def generate_mixed_state_dataset(num_epochs_per_state=30, output_dir='data/testing_data'):
    """
    Generate a dataset with mixed mental states for comprehensive testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", 
                "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"]
    
    filename = f"{output_dir}/test_mixed_states.csv"
    duration_sec = 1.028
    sampling_rate = 250
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + channels)
        
        timestamp = 0.0
        states = ['relaxed', 'focused', 'stressed']
        
        for state in states:
            print(f"  Generating {num_epochs_per_state} epochs of '{state}' state...")
            for epoch in range(num_epochs_per_state):
                signals = generate_eeg_signal(duration_sec, sampling_rate, state, len(channels))
                
                for sample_idx in range(signals.shape[0]):
                    row = [f"{timestamp:.4f}"] + [f"{val:.2f}" for val in signals[sample_idx]]
                    writer.writerow(row)
                    timestamp += 1.0 / sampling_rate
    
    total_samples = num_epochs_per_state * 3 * int(duration_sec * sampling_rate)
    print(f"✓ Generated mixed states dataset: {filename}")
    print(f"  - {num_epochs_per_state} epochs per state × 3 states = {total_samples} total samples")
    return filename

if __name__ == "__main__":
    print("=" * 60)
    print("EEG Test Dataset Generator")
    print("=" * 60)
    print()
    
    # Generate individual state datasets
    print("Generating individual state datasets...")
    generate_test_dataset('relaxed', num_epochs=50)
    generate_test_dataset('focused', num_epochs=50)
    generate_test_dataset('stressed', num_epochs=50)
    print()
    
    # Generate mixed state dataset
    print("Generating mixed state dataset...")
    generate_mixed_state_dataset(num_epochs_per_state=30)
    print()
    
    print("=" * 60)
    print("✓ All test datasets generated successfully!")
    print("=" * 60)
    print()
    print("Test these files with your API:")
    print("  - data/testing_data/test_relaxed_50epochs.csv")
    print("  - data/testing_data/test_focused_50epochs.csv")
    print("  - data/testing_data/test_stressed_50epochs.csv")
    print("  - data/testing_data/test_mixed_states.csv")

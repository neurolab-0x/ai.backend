"""
Generate comprehensive training dataset for EEG state classification.
Creates realistic frequency band features for relaxed, focused, and stressed states.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def generate_realistic_band_powers(state, num_samples=1000):
    """
    Generate realistic frequency band power values for different mental states.
    
    States and their characteristics:
    - Relaxed (0): High alpha, low beta, moderate theta
    - Focused (1): High beta, moderate alpha, low theta
    - Stressed (2): Very high beta, elevated gamma, low alpha, moderate theta
    
    Args:
        state: Mental state ('relaxed', 'focused', or 'stressed')
        num_samples: Number of samples to generate
    
    Returns:
        List of sample dictionaries with frequency band powers
    """
    samples = []
    
    for _ in range(num_samples):
        if state == 'relaxed':
            # Relaxed state: High alpha rhythm
            alpha = np.random.uniform(15, 35)  # High alpha (8-13 Hz)
            beta = np.random.uniform(3, 12)    # Low beta (13-30 Hz)
            theta = np.random.uniform(5, 15)   # Moderate theta (4-8 Hz)
            delta = np.random.uniform(2, 8)    # Low delta (0.5-4 Hz)
            gamma = np.random.uniform(1, 5)    # Low gamma (30-45 Hz)
            
        elif state == 'focused':
            # Focused state: elevated beta, moderate alpha
            alpha = np.random.uniform(8, 20)   # Moderate alpha
            beta = np.random.uniform(15, 35)   # High beta - concentration
            theta = np.random.uniform(2, 8)    # Low theta
            delta = np.random.uniform(1, 5)    # Low delta
            gamma = np.random.uniform(5, 15)   # Moderate gamma - cognitive processing
            
        elif state == 'stressed':
            # Stressed state: very high beta, low alpha
            alpha = np.random.uniform(3, 12)   # Low alpha - anxiety
            beta = np.random.uniform(25, 50)   # Very high beta - stress/anxiety
            theta = np.random.uniform(8, 18)   # Elevated theta - mental fatigue
            delta = np.random.uniform(3, 10)   # Moderate delta
            gamma = np.random.uniform(12, 30)  # High gamma - high arousal
            
        else:
            raise ValueError(f"Unknown state: {state}")
        
        # Add some natural variation
        alpha += np.random.normal(0, 2)
        beta += np.random.normal(0, 3)
        theta += np.random.normal(0, 2)
        delta += np.random.normal(0, 2)
        gamma += np.random.normal(0, 2)
        
        # Ensure all values are positive
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
        theta = max(0.1, theta)
        delta = max(0.1, delta)
        gamma = max(0.1, gamma)
        
        samples.append({
            'alpha': alpha,
            'beta': beta,
            'theta': theta,
            'delta': delta,
            'gamma': gamma,
            'state': 0 if state == 'relaxed' else (1 if state == 'focused' else 2)
        })
    
    return samples


def add_transition_samples(num_samples=300):
    """
    Generate samples representing transitions between states.
    These help the model learn boundaries between states.
    
    Args:
        num_samples: Number of transition samples to generate
    
    Returns:
        List of transition sample dictionaries
    """
    samples = []
    
    for _ in range(num_samples):
        # Random transition type
        transition = np.random.choice([
            'relaxed_to_focused',
            'focused_to_relaxed',
            'focused_to_stressed',
            'stressed_to_focused',
            'relaxed_to_stressed',
            'stressed_to_relaxed'
        ])
        
        if 'relaxed' in transition and 'focused' in transition:
            # Transition between relaxed and focused
            alpha = np.random.uniform(10, 25)
            beta = np.random.uniform(8, 25)
            theta = np.random.uniform(3, 12)
            delta = np.random.uniform(1, 7)
            gamma = np.random.uniform(2, 10)
            
        elif 'focused' in transition and 'stressed' in transition:
            # Transition between focused and stressed
            alpha = np.random.uniform(5, 18)
            beta = np.random.uniform(20, 40)
            theta = np.random.uniform(5, 15)
            delta = np.random.uniform(2, 8)
            gamma = np.random.uniform(8, 22)
            
        else:  # relaxed to/from stressed
            # Transition between relaxed and stressed
            alpha = np.random.uniform(5, 20)
            beta = np.random.uniform(12, 35)
            theta = np.random.uniform(6, 16)
            delta = np.random.uniform(2, 9)
            gamma = np.random.uniform(3, 18)
        
        # Determine state label based on dominant characteristics
        if beta > 25:
            state = 2  # stressed
        elif alpha > 18:
            state = 0  # relaxed
        else:
            state = 1  # focused
        
        samples.append({
            'alpha': alpha,
            'beta': beta,
            'theta': theta,
            'delta': delta,
            'gamma': gamma,
            'state': state
        })
    
    return samples



def generate_training_data(samples_per_state=1000, include_transitions=True):
    """
    Generate complete training dataset with all states.
    
    Args:
        samples_per_state: Number of samples to generate per state
        include_transitions: Whether to include transition samples
    
    Returns:
        pandas DataFrame with training data
    """
    all_samples = []
    
    # Generate samples for each state
    print(f"Generating {samples_per_state} samples for 'relaxed' state...")
    all_samples.extend(generate_realistic_band_powers('relaxed', samples_per_state))
    
    print(f"Generating {samples_per_state} samples for 'focused' state...")
    all_samples.extend(generate_realistic_band_powers('focused', samples_per_state))
    
    print(f"Generating {samples_per_state} samples for 'stressed' state...")
    all_samples.extend(generate_realistic_band_powers('stressed', samples_per_state))
    
    # Add transition samples
    if include_transitions:
        transition_count = int(samples_per_state * 0.3)  # 30% of per-state samples
        print(f"Generating {transition_count} transition samples...")
        all_samples.extend(add_transition_samples(transition_count))
    
    # Convert to DataFrame
    df = pd.DataFrame(all_samples)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nTotal samples generated: {len(df)}")
    print(f"State distribution:\n{df['state'].value_counts().sort_index()}")
    
    return df


def save_training_data(df, filename='data/training_data/training.csv'):
    """
    Save training data to CSV file.
    
    Args:
        df: DataFrame with training data
        filename: Output filename
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\nTraining data saved to: {filename}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.groupby('state')[['alpha', 'beta', 'theta', 'delta', 'gamma']].mean())


if __name__ == '__main__':
    print("=" * 60)
    print("EEG Training Data Generator")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Generate training data
    df = generate_training_data(
        samples_per_state=10000,
        include_transitions=True
    )
    
    # Save to file
    save_training_data(df)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from typing import List, Dict

# Standard EEG bands (unified across scripts)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# State-specific amplitude profiles (unified across scripts)
STATE_PROFILES = {
    'relaxed': {'alpha': (20, 35), 'beta': (3, 10), 'theta': (8, 15), 'delta': (2, 6), 'gamma': (1, 4)},
    'focused': {'alpha': (10, 18), 'beta': (18, 30), 'theta': (3, 8), 'delta': (1, 5), 'gamma': (5, 12)},
    'stressed': {'alpha': (3, 10), 'beta': (30, 50), 'theta': (10, 20), 'delta': (4, 8), 'gamma': (15, 30)}
}

def generate_band_powers(state: str, num_samples: int = 1000) -> List[Dict]:
    """Generate realistic frequency band power values for mental states."""
    samples = []
    profile = STATE_PROFILES.get(state, STATE_PROFILES['relaxed'])
    state_label = 0 if state == 'relaxed' else (1 if state == 'focused' else 2)
    
    for _ in range(num_samples):
        sample = {'state': state_label}
        for band, (low, high) in profile.items():
            # Generate random power within the profile range plus some variance
            val = np.random.uniform(low, high)
            val += np.random.normal(0, (high - low) * 0.1)
            sample[band] = max(0.1, val)
            
        samples.append(sample)
    
    return samples

def generate_transitions(num_samples: int = 300) -> List[Dict]:
    """Generate samples representing transitions between states."""
    samples = []
    states = list(STATE_PROFILES.keys())
    
    for _ in range(num_samples):
        # Pick two states to interpolate between
        s1, s2 = np.random.choice(states, 2, replace=False)
        p1, p2 = STATE_PROFILES[s1], STATE_PROFILES[s2]
        
        # Interpolation factor
        alpha_interp = np.random.random()
        
        sample = {}
        for band in BANDS.keys():
            v1_range = p1[band]
            v2_range = p2[band]
            v1 = np.random.uniform(*v1_range)
            v2 = np.random.uniform(*v2_range)
            sample[band] = v1 * (1 - alpha_interp) + v2 * alpha_interp
            
        # Assign label based on dominance
        if sample['beta'] > 25:
            sample['state'] = 2 # stressed
        elif sample['alpha'] > 18:
            sample['state'] = 0 # relaxed
        else:
            sample['state'] = 1 # focused
            
        samples.append(sample)
        
    return samples

def main():
    parser = argparse.ArgumentParser(description="Generate EEG training datasets.")
    parser.add_argument("--samples", type=int, default=10000, help="Samples per state (default: 10000)")
    parser.add_argument("--transitions", type=int, default=3000, help="Number of transition samples")
    parser.add_argument("--outfile", type=str, default="data/training_data/training.csv", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("EEG Training Data Generator")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    all_samples = []
    for state in STATE_PROFILES.keys():
        print(f"Generating {args.samples} samples for '{state}' state...")
        all_samples.extend(generate_band_powers(state, args.samples))
        
    if args.transitions > 0:
        print(f"Generating {args.transitions} transition samples...")
        all_samples.extend(generate_transitions(args.transitions))
        
    df = pd.DataFrame(all_samples)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    df.to_csv(args.outfile, index=False)
    
    print(f"\n✓ Saved {len(df)} samples to {args.outfile}")
    print("\nState distribution:")
    print(df['state'].value_counts().sort_index())
    
    print("\nMean band powers per state:")
    print(df.groupby('state')[list(BANDS.keys())].mean())
    print("=" * 60)

if __name__ == "__main__":
    main()

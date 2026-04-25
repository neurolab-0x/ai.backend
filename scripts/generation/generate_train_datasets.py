import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import multiprocessing as mp
from functools import partial
import signal
import sys

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

def generate_chunk(chunk_size: int, transition_ratio: float = 0.1) -> pd.DataFrame:
    """Generate a chunk of randomized EEG samples using vectorized NumPy operations."""
    num_transitions = int(chunk_size * transition_ratio)
    num_base = chunk_size - num_transitions
    
    # 1. Generate base states (equally distributed)
    samples_per_state = num_base // 3
    all_base_data = []
    all_base_labels = []
    
    for state_idx, (state_name, profile) in enumerate(STATE_PROFILES.items()):
        n = samples_per_state if state_idx < 2 else num_base - (samples_per_state * 2)
        state_data = np.zeros((n, len(BANDS)))
        
        for i, band in enumerate(list(BANDS.keys())):
            low, high = profile[band]
            # Vectorized generation for efficiency
            vals = np.random.uniform(low, high, size=n)
            # Add some natural variance/noise
            vals += np.random.normal(0, (high - low) * 0.1, size=n)
            state_data[:, i] = np.maximum(0.1, vals)
            
        all_base_data.append(state_data)
        all_base_labels.append(np.full(n, state_idx))
        
    # 2. Generate transitions
    states = list(STATE_PROFILES.keys())
    transition_data = np.zeros((num_transitions, len(BANDS)))
    transition_labels = np.zeros(num_transitions, dtype=int)
    
    for i in range(num_transitions):
        # Pick two random states to interpolate between
        s1_idx, s2_idx = np.random.choice(len(states), 2, replace=False)
        p1, p2 = STATE_PROFILES[states[s1_idx]], STATE_PROFILES[states[s2_idx]]
        
        alpha_interp = np.random.random()
        
        row = {}
        for b_idx, band in enumerate(BANDS.keys()):
            v1 = np.random.uniform(*p1[band])
            v2 = np.random.uniform(*p2[band])
            val = v1 * (1 - alpha_interp) + v2 * alpha_interp
            transition_data[i, b_idx] = val
            row[band] = val
            
        # Assign label based on dominance rules
        if row['beta'] > 25:
            transition_labels[i] = 2 # stressed
        elif row['alpha'] > 18:
            transition_labels[i] = 0 # relaxed
        else:
            transition_labels[i] = 1 # focused

    # Combine and shuffle
    combined_data = np.vstack(all_base_data + [transition_data])
    combined_labels = np.concatenate(all_base_labels + [transition_labels])
    
    df = pd.DataFrame(combined_data, columns=list(BANDS.keys()))
    df['state'] = combined_labels
    
    return df.sample(frac=1).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="Generate large-scale EEG training datasets.")
    parser.add_argument("--total_samples", "--samples", type=int, default=None, help="Total samples to generate")
    parser.add_argument("--transitions", type=int, default=None, help="Number of transition samples (overrides transition_ratio)")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Samples per processing chunk")
    parser.add_argument("--transition_ratio", type=float, default=0.1, help="Ratio of transition samples (0.0 to 1.0)")
    parser.add_argument("--outfile", type=str, default="data/training_data/training.csv", help="Output file path")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Handle backward compatibility and argument prioritization
    if args.total_samples is None:
        # Default if nothing provided
        total_samples = 100000
    else:
        total_samples = args.total_samples

    transition_ratio = args.transition_ratio
    if args.transitions is not None:
        # If specific transition count is provided, recalculate ratio
        transition_ratio = args.transitions / total_samples if total_samples > 0 else 0
    
    print("=" * 60)
    print("NeuroLab Scalable Data Generator")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Target: {total_samples:,} samples")
    print(f"Transition Ratio: {transition_ratio:.2%}")
    print(f"Workers: {args.workers}")
    print("=" * 60)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    
    # Calculate chunks
    num_chunks = max(1, total_samples // args.chunk_size)
    samples_per_worker_chunk = total_samples // num_chunks
    
    print(f"Generating in {num_chunks} chunks of ~{samples_per_worker_chunk:,} samples each...")
    
    # Use Pool for parallel generation
    first_chunk = True
    processed_count = 0
    
    with mp.Pool(processes=args.workers) as pool:
        # Create a partial function with fixed transition ratio
        gen_func = partial(generate_chunk, transition_ratio=transition_ratio)
        
        # Generator for chunk sizes
        chunk_sizes = [samples_per_worker_chunk] * (num_chunks - 1)
        chunk_sizes.append(total_samples - sum(chunk_sizes))
        
        for df_chunk in pool.imap_unordered(gen_func, chunk_sizes):
            # Save chunk to CSV
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            df_chunk.to_csv(args.outfile, mode=mode, header=header, index=False)
            
            processed_count += len(df_chunk)
            first_chunk = False
            
            percent = (processed_count / total_samples) * 100 if total_samples > 0 else 100
            print(f"Progress: [{processed_count:,}/{total_samples:,}] {percent:.1f}% complete")

    print(f"\n✓ Successfully saved {processed_count:,} samples to {args.outfile}")
    
    # Final check on a sample of the data
    try:
        final_check = pd.read_csv(args.outfile, nrows=10000)
        print("\nDistribution (first 10k samples):")
        print(final_check['state'].value_counts(normalize=True).sort_index())
    except Exception:
        pass
        
    print(f"Finished at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    # Fix for multiprocessing on some systems
    if sys.platform == 'win32':
        mp.freeze_support()
    main()
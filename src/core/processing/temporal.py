import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def temporal_smoothing(states: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply temporal smoothing to state predictions using a moving average window.
    
    Args:
        states (np.ndarray): Array of state predictions
        window_size (int): Size of the smoothing window
        
    Returns:
        np.ndarray: Smoothed state predictions
    """
    try:
        if len(states) < window_size:
            logger.warning(f"Input length ({len(states)}) is less than window size ({window_size}). Returning original states.")
            return states
            
        # Use convolution for efficient moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(states, kernel, mode='same')
        
        # Round to nearest integer to get valid state indices
        return np.round(smoothed).astype(int)
        
    except Exception as e:
        logger.error(f"Temporal smoothing failed: {str(e)}")
        return states

def calculate_state_durations(states: np.ndarray, sampling_rate: float = 1.0) -> dict:
    """
    Calculate the duration of each state in the sequence.
    
    Args:
        states (np.ndarray): Array of state predictions
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        dict: Dictionary mapping states to their durations in seconds
    """
    try:
        if len(states) == 0:
            return {}
            
        # Find state transitions
        transitions = np.where(np.diff(states) != 0)[0] + 1
        transitions = np.concatenate(([0], transitions, [len(states)]))
        
        # Calculate durations
        durations = {}
        for i in range(len(transitions) - 1):
            state = states[transitions[i]]
            duration = (transitions[i + 1] - transitions[i]) / sampling_rate
            durations[state] = durations.get(state, 0) + duration
            
        return durations
        
    except Exception as e:
        logger.error(f"Duration calculation failed: {str(e)}")
        return {}

def detect_state_transitions(states: np.ndarray, min_duration: float = 1.0, sampling_rate: float = 1.0) -> List[Tuple[int, int, int]]:
    """
    Detect significant state transitions with minimum duration threshold.
    
    Args:
        states (np.ndarray): Array of state predictions
        min_duration (float): Minimum duration for a state to be considered significant
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        List[Tuple[int, int, int]]: List of (start_idx, end_idx, state) tuples
    """
    try:
        if len(states) == 0:
            return []
            
        # Find state transitions
        transitions = np.where(np.diff(states) != 0)[0] + 1
        transitions = np.concatenate(([0], transitions, [len(states)]))
        
        # Filter transitions based on duration
        significant_states = []
        for i in range(len(transitions) - 1):
            start_idx = transitions[i]
            end_idx = transitions[i + 1]
            state = states[start_idx]
            duration = (end_idx - start_idx) / sampling_rate
            
            if duration >= min_duration:
                significant_states.append((start_idx, end_idx, state))
                
        return significant_states
        
    except Exception as e:
        logger.error(f"State transition detection failed: {str(e)}")
        return []

def analyze_temporal_patterns(states: np.ndarray, sampling_rate: float = 1.0) -> dict:
    """
    Analyze temporal patterns in state sequences.
    
    Args:
        states (np.ndarray): Array of state predictions
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        dict: Dictionary containing temporal pattern analysis
    """
    try:
        if len(states) == 0:
            return {
                "total_duration": 0,
                "state_durations": {},
                "transition_count": 0,
                "dominant_state": None,
                "state_sequence": []
            }
            
        # Calculate basic metrics
        durations = calculate_state_durations(states, sampling_rate)
        transitions = np.where(np.diff(states) != 0)[0]
        transition_count = len(transitions)
        
        # Find dominant state
        if durations:
            dominant_state = max(durations.items(), key=lambda x: x[1])[0]
        else:
            dominant_state = None
            
        # Analyze state sequence
        state_sequence = []
        current_state = states[0]
        current_duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                state_sequence.append((current_state, current_duration / sampling_rate))
                current_state = states[i]
                current_duration = 1
                
        state_sequence.append((current_state, current_duration / sampling_rate))
        
        return {
            "total_duration": sum(durations.values()),
            "state_durations": durations,
            "transition_count": transition_count,
            "dominant_state": dominant_state,
            "state_sequence": state_sequence
        }
        
    except Exception as e:
        logger.error(f"Temporal pattern analysis failed: {str(e)}")
        return {
            "error": str(e),
            "total_duration": 0,
            "state_durations": {},
            "transition_count": 0,
            "dominant_state": None,
            "state_sequence": []
        } 
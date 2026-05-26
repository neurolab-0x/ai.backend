import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load EEG data from supported backend file formats (.csv, .edf)
    
    Parameters:
    -----------
    file_path : str
        Path to the EEG data file
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing EEG data
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    logger.info(f"Loading file: {file_path} with format {extension}")
    
    try:
        if extension == '.csv':
            return load_csv_data(file_path)
        elif extension == '.edf':
            return load_edf_data(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise


def load_csv_data(file_path):
    """Load data from CSV file format."""
    try:
        df = None
        # Try different delimiters
        for delimiter in [',', ';', '\t']:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter)
                if len(df.columns) > 1:  # Successful parsing
                    break
            except Exception:
                continue
        
        # Check if successful
        if df is None or len(df.columns) <= 1:
            raise ValueError("Could not parse CSV file with standard delimiters")
        
        # Check if column names are numeric and convert them
        if all(col.replace('.', '').isdigit() for col in df.columns if isinstance(col, str)):
            df.columns = [f"channel_{i}" for i in range(len(df.columns))]
        
        # Ensure numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                # Skip non-numeric columns (like timestamps, labels, etc.)
                pass
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise


def load_edf_data(file_path):
    """Load data from EDF (European Data Format) file."""
    try:
        import pyedflib
        
        with pyedflib.EdfReader(file_path) as f:
            n_channels = f.signals_in_file
            signal_labels = f.getSignalLabels()
            
            # Read all signals
            signals = np.zeros((n_channels, f.getNSamples()[0]))
            for i in range(n_channels):
                signals[i, :] = f.readSignal(i)
            
            # Create DataFrame
            df = pd.DataFrame(signals.T, columns=signal_labels)
            
            # Add metadata if available
            try:
                # Extract sampling frequency
                sfreq = f.getSampleFrequency(0)
                df.attrs['sfreq'] = sfreq
                
                # Extract recording info
                recording_info = {
                    'startdate': f.getStartdatetime(),
                    'patient': f.getPatientCode(),
                    'gender': f.getGender(),
                    'birthdate': f.getBirthdate(),
                    'equipment': f.getEquipment()
                }
                df.attrs['recording_info'] = recording_info
            except:
                pass
                
            return df
        
    except ImportError:
        logger.error("pyedflib not installed. Please install with: pip install pyedflib")
        raise
    except Exception as e:
        logger.error(f"Error loading EDF file: {str(e)}")
        raise

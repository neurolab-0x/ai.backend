import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load EEG data from various file formats (.csv, .edf, .bdf, .gdf, .mat)
    
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
        elif extension in ['.bdf', '.gdf']:
            return load_biosignal_data(file_path)
        elif extension == '.mat':
            return load_matlab_data(file_path)
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


def load_biosignal_data(file_path):
    """Load data from BDF/GDF file formats."""
    try:
        import mne
        
        # Load raw data
        raw = mne.io.read_raw(file_path, preload=True)
        
        # Extract data and channel names
        data = raw.get_data()
        ch_names = raw.ch_names
        
        # Create DataFrame
        df = pd.DataFrame(data.T, columns=ch_names)
        
        # Store metadata
        df.attrs['sfreq'] = raw.info['sfreq']
        df.attrs['ch_types'] = [raw.info['chs'][i]['kind'] for i in range(len(ch_names))]
        
        return df
    
    except ImportError:
        logger.error("mne not installed. Please install with: pip install mne")
        raise
    except Exception as e:
        logger.error(f"Error loading biosignal file: {str(e)}")
        raise


def load_matlab_data(file_path):
    """Load data from MATLAB .mat file format."""
    try:
        import scipy.io as sio
        
        # Load mat file
        mat_data = sio.loadmat(file_path)
        
        # Find main data arrays (usually the largest numeric arrays)
        data_candidates = {}
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray) and value.size > 10 and not key.startswith('__'):
                data_candidates[key] = value.size
        
        # Get key with largest array
        if data_candidates:
            main_key = max(data_candidates, key=data_candidates.get)
            data = mat_data[main_key]
            
            # Reshape if needed
            if data.ndim > 2:
                # Flatten all but last dimension for multi-dimensional data
                data = data.reshape(-1, data.shape[-1])
            
            # Create column names
            if data.shape[1] > data.shape[0]:
                # Transpose if we have more columns than rows 
                # (assuming channels are usually fewer than timepoints)
                data = data.T
            
            columns = [f"channel_{i}" for i in range(data.shape[1])]
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)
            
            # Store metadata
            if 'srate' in mat_data:
                df.attrs['sfreq'] = float(mat_data['srate'])
            elif 'fs' in mat_data:
                df.attrs['sfreq'] = float(mat_data['fs'])
                
            # Look for channel names
            for key in mat_data:
                if 'label' in key.lower() or 'channel' in key.lower() or 'ch' in key.lower():
                    try:
                        ch_names = [str(x[0]) for x in mat_data[key]]
                        if len(ch_names) == data.shape[1]:
                            df.columns = ch_names
                            break
                    except Exception:
                        pass
            
            return df
        else:
            raise ValueError("No suitable data arrays found in .mat file")
    
    except ImportError:
        logger.error("scipy not installed. Please install with: pip install scipy")
        raise
    except Exception as e:
        logger.error(f"Error loading MATLAB file: {str(e)}")
        raise

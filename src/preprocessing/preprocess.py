import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from src.preprocessing.features import extract_features
from src.core.processing.artifacts import clean_eeg
from src.core.processing.filters import apply_eeg_preprocessing, filter_eeg_bands
import logging
from typing import Tuple, Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import warnings
from tqdm import tqdm
import joblib
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass

def validate_input_data(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Enhanced input data validation with detailed quality metrics"""
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'quality_metrics': {}
    }
    
    try:
        if not isinstance(df, pd.DataFrame):
            raise PreprocessingError("Input must be a pandas DataFrame")
        
        if target_column not in df.columns:
            raise PreprocessingError(f"Target column '{target_column}' not found in DataFrame")
        
        if df.empty:
            raise PreprocessingError("Empty DataFrame provided")
        
        # Check for infinite values
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]).values)
        if inf_mask.any():
            validation_results['warnings'].append(f"Found {inf_mask.sum()} infinite values")
            validation_results['is_valid'] = False
        
        # Check for null values
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if not null_cols.empty:
            validation_results['warnings'].append(f"Found null values in columns: {null_cols.to_dict()}")
        
        # Check for constant columns
        constant_cols = df.columns[df.nunique() == 1].tolist()
        if constant_cols:
            validation_results['warnings'].append(f"Found constant columns: {constant_cols}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_results['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # Compute basic statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != target_column:
                validation_results['quality_metrics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'null_count': int(df[col].isnull().sum()),
                    'unique_count': int(df[col].nunique())
                }
        
        # Check class distribution
        if target_column in df.columns:
            class_dist = df[target_column].value_counts().to_dict()
            validation_results['quality_metrics']['class_distribution'] = class_dist
            
            # Check for class imbalance
            min_class_count = min(class_dist.values())
            max_class_count = max(class_dist.values())
            if max_class_count / min_class_count > 3:
                validation_results['warnings'].append(
                    f"Severe class imbalance detected: {max_class_count/min_class_count:.2f}x"
                )
        
        return validation_results
        
    except Exception as e:
        raise PreprocessingError(f"Error in data validation: {str(e)}")

def compute_signal_quality_metrics(signal: np.ndarray) -> Dict[str, float]:
    """Compute signal quality metrics for EEG data"""
    metrics = {}
    try:
        # Signal-to-noise ratio (SNR)
        signal_power = np.mean(signal ** 2)
        noise_power = np.var(signal - np.mean(signal))
        metrics['snr'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Peak-to-peak amplitude
        metrics['peak_to_peak'] = np.max(signal) - np.min(signal)
        
        # Zero crossing rate
        metrics['zero_crossing_rate'] = np.sum(np.diff(np.signbit(signal).astype(int)) != 0) / len(signal)
        
        # Signal energy
        metrics['energy'] = np.sum(signal ** 2)
        
        # Signal entropy
        hist, _ = np.histogram(signal, bins=50, density=True)
        metrics['entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return metrics
    except Exception as e:
        logger.warning(f"Error computing signal quality metrics: {str(e)}")
        return {k: 0 for k in ['snr', 'peak_to_peak', 'zero_crossing_rate', 'energy', 'entropy']}

def analyze_feature_importance(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Analyze feature importance using multiple methods"""
    importance_scores = {}
    
    try:
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = dict(zip(feature_names, rf.feature_importances_))
        
        # F-score importance
        f_scores, _ = f_classif(X, y)
        f_importance = dict(zip(feature_names, f_scores))
        
        # Mutual information importance
        mi_scores = mutual_info_classif(X, y)
        mi_importance = dict(zip(feature_names, mi_scores))
        
        # Combine scores
        for feature in feature_names:
            importance_scores[feature] = {
                'random_forest': float(rf_importance[feature]),
                'f_score': float(f_importance[feature]),
                'mutual_info': float(mi_importance[feature]),
                'combined_score': float(
                    (rf_importance[feature] + f_importance[feature] + mi_importance[feature]) / 3
                )
            }
        
        return importance_scores
    except Exception as e:
        logger.warning(f"Error in feature importance analysis: {str(e)}")
        return {}

def split_data(df: pd.DataFrame, target_column: str = 'eeg_state', 
               test_size: float = 0.2, random_state: int = 42, 
               stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enhanced data splitting with validation"""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    except Exception as e:
        raise PreprocessingError(f"Error in data splitting: {str(e)}")

def augment_data(X: np.ndarray, y: np.ndarray, noise_level: float = 0.05, 
                shift_max: int = 5, num_augmented_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced data augmentation with parallel processing"""
    if num_augmented_samples is None:
        num_augmented_samples = X.shape[0] // 4
    
    def augment_single_sample(idx: int) -> Tuple[np.ndarray, int]:
        x = X[idx].copy()
        
        # Apply random noise
        noise = np.random.normal(0, noise_level, x.shape)
        x_noise = x + noise
        
        # Apply time shift
        shift = np.random.randint(-shift_max, shift_max + 1)
        if shift != 0:
            x_shift = np.roll(x, shift)
            if shift > 0:
                # Clamp shift if it exceeds array length
                s = min(shift, len(x) - 1)
                x_shift[:s] = x_shift[s]
            else:
                # Clamp shift if it exceeds array length
                s = max(shift, -len(x) + 1)
                x_shift[s:] = x_shift[s-1]
        else:
            x_shift = x
        
        # Combine transformations with reduced noise
        x_augmented = x_shift + (noise * 0.5)
        return x_augmented, y[idx]
    
    # Use parallel processing for augmentation
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            augment_single_sample,
            np.random.randint(0, X.shape[0], num_augmented_samples)
        ))
    
    augmented_X = np.vstack([X] + [r[0] for r in results])
    augmented_y = np.concatenate([y] + [np.atleast_1d(r[1]) for r in results])
    
    return augmented_X, augmented_y

def remove_outliers(X: np.ndarray, y: np.ndarray, contamination: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced outlier removal with validation"""
    from sklearn.ensemble import IsolationForest
    
    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = iso_forest.fit_predict(X)
        
        mask = outlier_pred == 1
        X_clean = X[mask]
        y_clean = y[mask]
        
        removed_count = len(y) - len(y_clean)
        removed_pct = (1 - len(y_clean)/len(y)) * 100
        
        logger.info(f"Removed {removed_count} outliers ({removed_pct:.2f}%)")
        
        if removed_pct > 50:  # Warning if more than 50% removed
            warnings.warn(f"High percentage of outliers removed: {removed_pct:.2f}%")
        
        return X_clean, y_clean
    except Exception as e:
        raise PreprocessingError(f"Error in outlier removal: {str(e)}")

def preprocess_data(df: pd.DataFrame, target_column: str = 'eeg_state', 
                   num_features: Optional[int] = None, clean_artifacts: bool = True,
                   use_robust_scaler: bool = False, balance_method: str = 'smote',
                   parallel_processing: bool = True, cache_dir: Optional[str] = None,
                   n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Enhanced preprocessing pipeline with improved error handling, caching, and cross-validation
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe containing EEG data and state labels
    target_column : str
        Column name for the target variable
    num_features : int or None
        Number of features to select (if None, uses all features)
    clean_artifacts : bool
        Whether to apply artifact cleaning
    use_robust_scaler : bool
        Whether to use RobustScaler instead of StandardScaler
    balance_method : str
        Method to balance classes ('smote', 'adasyn', or None)
    parallel_processing : bool
        Whether to use parallel processing for feature extraction
    cache_dir : str or None
        Directory to cache intermediate results
    n_splits : int
        Number of cross-validation splits
        
    Returns:
    -------
    X_train, X_test, y_train, y_test, metadata
    """
    try:
        logger.info("Starting enhanced preprocessing pipeline...")
        metadata = {
            'preprocessing_steps': [],
            'feature_importance': {},
            'signal_quality': {},
            'validation_results': {}
        }
        
        # Create cache directory if specified
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
        
        # Validate input data (only if target column exists)
        if target_column in df.columns:
            validation_results = validate_input_data(df, target_column)
            metadata['validation_results'] = validation_results
            
            if not validation_results['is_valid']:
                logger.warning("Data validation failed. Proceeding with caution.")
                for warning in validation_results['warnings']:
                    logger.warning(warning)
        
        # Extract features if needed
        if 'eeg_state' in df.columns and len(df.columns) < 10:
            logger.info("Extracting features from raw data...")
            
            if cache_dir and (cache_path / 'features.pkl').exists():
                logger.info("Loading cached features...")
                df_features = joblib.load(cache_path / 'features.pkl')
            else:
                if parallel_processing:
                    # Parallel feature extraction with progress bar
                    with ThreadPoolExecutor() as executor:
                        futures = []
                        for i in tqdm(range(len(df)), desc="Extracting features"):
                            row_df = df.iloc[i:i+1]
                            if clean_artifacts:
                                futures.append(executor.submit(
                                    _process_row_with_artifacts, row_df, target_column
                                ))
                            else:
                                futures.append(executor.submit(extract_features, row_df))
                        
                        df_features = pd.concat([f.result() for f in futures], ignore_index=True)
                else:
                    # Sequential feature extraction with progress bar
                    df_features = pd.DataFrame()
                    for i in tqdm(range(len(df)), desc="Extracting features"):
                        row_df = df.iloc[i:i+1]
                        if clean_artifacts:
                            processed_row = _process_row_with_artifacts(row_df, target_column)
                        else:
                            processed_row = extract_features(row_df)
                        df_features = pd.concat([df_features, processed_row], ignore_index=True)
                
                if cache_dir:
                    joblib.dump(df_features, cache_path / 'features.pkl')
        else:
            logger.info("Using provided feature dataframe...")
            df_features = df.copy()
        
        logger.info(f"Feature extraction complete. Shape: {df_features.shape}")
        metadata['preprocessing_steps'].append('feature_extraction')
        
        # Compute signal quality metrics
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != target_column:
                signal = df_features[col].values
                metadata['signal_quality'][col] = compute_signal_quality_metrics(signal)
        
        # Check if we're in inference mode (no target column)
        inference_mode = target_column not in df_features.columns
        
        if inference_mode:
            # Inference mode: just prepare features without splitting
            logger.info("Inference mode detected - skipping train/test split")
            X_inference = df_features.values
            
            # Handle missing values
            logger.info("Handling missing values...")
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            X_inference = imputer.fit_transform(X_inference)
            metadata['preprocessing_steps'].append('missing_value_imputation')
            
            # Scale features
            logger.info("Scaling features...")
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X_inference)
            metadata['preprocessing_steps'].append('feature_scaling')
            
            logger.info(f"Preprocessing complete! Inference data shape: {X_normalized.shape}")
            # Return in format expected by caller (X_train, X_test, y_train, y_test, metadata)
            # For inference, return X as first element, rest as None
            return X_normalized, None, None, None, metadata
        
        X_train, X_test, y_train, y_test = split_data(df_features, target_column)
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        metadata['preprocessing_steps'].append('data_splitting')
        
        # Handle missing values
        logger.info("Handling missing values...")
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        metadata['preprocessing_steps'].append('missing_value_imputation')
        
        # Scale features
        logger.info("Scaling features...")
        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        metadata['preprocessing_steps'].append('feature_scaling')
        
        # Augment training data
        logger.info("Augmenting training data...")
        X_train, y_train = augment_data(X_train, y_train)
        logger.info(f"After augmentation - Training set: {X_train.shape}")
        metadata['preprocessing_steps'].append('data_augmentation')
        
        # Balance classes
        if balance_method:
            logger.info(f"Balancing classes using {balance_method}...")
            logger.info(f"Class distribution before balancing: {Counter(y_train)}")
            
            balancer = SMOTE(sampling_strategy='auto', random_state=42) if balance_method.lower() == 'smote' else \
                      ADASYN(sampling_strategy='auto', random_state=42) if balance_method.lower() == 'adasyn' else \
                      None
            
            if balancer is None:
                raise ValueError(f"Unknown balance method: {balance_method}")
            
            X_train, y_train = balancer.fit_resample(X_train, y_train)
            logger.info(f"Class distribution after balancing: {Counter(y_train)}")
            metadata['preprocessing_steps'].append('class_balancing')
        
        # Feature selection
        if num_features is not None and num_features < X_train.shape[1]:
            logger.info(f"Selecting top {num_features} features...")
            
            # Enhanced feature selection with multiple methods
            selector_f = SelectKBest(score_func=f_classif, k=num_features // 2)
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=num_features // 2)
            
            X_train_f = selector_f.fit_transform(X_train, y_train)
            X_train_mi = selector_mi.fit_transform(X_train, y_train)
            
            X_test_f = selector_f.transform(X_test)
            X_test_mi = selector_mi.transform(X_test)
            
            X_train = np.concatenate((X_train_f, X_train_mi), axis=1)
            X_test = np.concatenate((X_test_f, X_test_mi), axis=1)
            
            # Compute feature importance
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            metadata['feature_importance'] = analyze_feature_importance(X_train, y_train, feature_names)
            
            logger.info(f"After feature selection - Feature count: {X_train.shape[1]}")
            metadata['preprocessing_steps'].append('feature_selection')
        
        # Perform cross-validation
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train a simple model for validation
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_fold_train, y_fold_train)
            score = rf.score(X_fold_val, y_fold_val)
            cv_scores.append(score)
            
            logger.info(f"Fold {fold + 1}/{n_splits} - Validation score: {score:.4f}")
        
        metadata['cross_validation'] = {
            'n_splits': n_splits,
            'scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        }
        
        logger.info("Preprocessing complete!")
        return X_train, X_test, y_train, y_test, metadata
        
    except Exception as e:
        raise PreprocessingError(f"Error in preprocessing pipeline: {str(e)}")

def _process_row_with_artifacts(row_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Helper function for processing a single row with artifact cleaning"""
    for col in row_df.columns:
        if col != target_column:
            signal = np.array(row_df[col])
            if len(signal) > 10:
                cleaned_signal, _ = clean_eeg(signal.reshape(1, -1))
                cleaned_signal = apply_eeg_preprocessing(cleaned_signal[0])
                row_df[col] = cleaned_signal
    return extract_features(row_df)

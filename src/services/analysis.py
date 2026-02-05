import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import os

from src.preprocessing import (
    load_data,
    extract_features,
    preprocess_data
)
from src.preprocessing.labeling import label_eeg_states
from src.core.ml.model import load_calibrated_model
from src.core.processing.temporal import temporal_smoothing, calculate_state_durations
from src.services.recommendation import NLPRecommendationEngine
from src.config.settings import PROCESSING_CONFIG, THRESHOLDS

logger = logging.getLogger(__name__)

class MLProcessor:
    """
    ML Processor for EEG data analysis pipeline.
    Handles model loading, data preprocessing, predictions, and recommendations.
    """
    
    def __init__(self, default_model: Optional[str] = None):
        """
        Initialize ML Processor.
        
        Args:
            default_model: Name of the default architecture (optional)
        """
        self.default_model = default_model
        self.models = {}  # Model cache: {model_type: model_object}
        self.recommendation_engine = NLPRecommendationEngine()
        
        # Load all available models from the model directory
        self._load_available_models()
        
        # Ensure default model is loaded if specified (even if not on disk, it might be built on fly)
        if default_model and default_model not in self.models:
             self._get_or_load_model(default_model)
             
        logger.info(f"ML Processor initialized with models: {list(self.models.keys())}")

    def _load_available_models(self):
        """Scan model directory and load all .h5 files"""
        model_dir = "model"
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory '{model_dir}' does not exist")
            return

        for filename in os.listdir(model_dir):
            if filename.endswith(".h5") or filename.endswith(".keras"):
                model_name = os.path.splitext(filename)[0]
                try:
                    self._get_or_load_model(model_name)
                    logger.info(f"Pre-loaded available model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to pre-load model {model_name}: {e}")
    
    def _get_or_load_model(self, model_type: str):
        """Get model from cache or load it if not available"""
        if model_type not in self.models:
            model_path = f"model/{model_type}.h5"
            try:
                if os.path.exists(model_path):
                    model = load_calibrated_model(model_path)
                    if model is not None:
                        # Warm up the model
                        dummy_input = np.zeros((1, 5, 1))
                        _ = model.predict(dummy_input, verbose=0)
                        self.models[model_type] = model
                        logger.info(f"Model {model_type} loaded successfully from {model_path}")
                    else:
                        logger.warning(f"Model loading returned None for {model_type}")
                else:
                    logger.info(f"Model file not found at {model_path}. Creating base model for {model_type}.")
                    self.models[model_type] = load_calibrated_model(model_type)
            except Exception as e:
                logger.error(f"Error loading model {model_type}: {str(e)}")
                return None
        return self.models.get(model_type)

    def process_eeg_data(
        self, 
        data: Union[str, Dict, np.ndarray, pd.DataFrame], 
        subject_id: str = "anonymous", 
        session_id: str = "default_session",
        model_type: Optional[str] = None,
        overlap: float = 0.0,
        simple_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Process EEG data through the complete pipeline.
        
        Args:
            data: EEG data
            subject_id: Unique identifier for the subject
            session_id: Unique identifier for the session
            model_type: Architecture to use for inference
            
        Returns:
            Dict containing predictions, states, durations, and recommendations
        """
        try:
            model_type = model_type or self.default_model
            if not model_type:
                raise ValueError("model_type must be specified")
            logger.info(f"Processing EEG data for subject {subject_id}, session {session_id} using model {model_type}")
            
            # Step 1: Load and preprocess data
            processed_features = self._preprocess_input(data, overlap=overlap, simple_mode=simple_mode)
            
            # Step 2: Make predictions using the specified model
            model = self._get_or_load_model(model_type)
            predictions = self._make_predictions(processed_features, model=model)
            
            # Step 3: Apply temporal smoothing
            smoothed_states = temporal_smoothing(
                predictions['predicted_states'],
                window_size=PROCESSING_CONFIG['smoothing_window']
            )
            
            # Step 4: Calculate state durations
            state_durations = calculate_state_durations(smoothed_states)
            total_duration = len(smoothed_states)
            
            # Step 4.5: Calculate cognitive metrics
            cognitive_metrics = self._calculate_cognitive_metrics(processed_features)
            state_transitions = self._count_state_transitions(smoothed_states)
            
            # Step 5: Generate NLP-based recommendations
            recommendations = self.recommendation_engine.generate_recommendations(
                state_durations,
                total_duration,
                predictions['confidence'],
                cognitive_metrics=cognitive_metrics,
                state_transitions=state_transitions
            )
            
            # Step 6: Compile results
            result = {
                'predicted_state': predictions['predicted_states'].tolist() if isinstance(predictions['predicted_states'], np.ndarray) else predictions['predicted_states'],
                'smoothed_states': smoothed_states.tolist() if isinstance(smoothed_states, np.ndarray) else smoothed_states,
                'dominant_state': int(predictions['dominant_state']),
                'state_label': self._get_state_label(predictions['dominant_state']),
                'confidence': float(predictions['confidence']),
                'state_durations': {int(k): int(v) for k, v in state_durations.items()},
                'state_percentages': {
                    int(state): round(duration / total_duration * 100, 2)
                    for state, duration in state_durations.items()
                },
                'recommendations': recommendations,
                'temporal_analysis': {
                    'total_samples': int(total_duration),
                    'smoothing_window': PROCESSING_CONFIG['smoothing_window'],
                    'state_transitions': state_transitions
                },
                'cognitive_metrics': cognitive_metrics,
                'clinical_recommendations': recommendations,
                'metadata': {
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type
                }
            }
            
            logger.info(f"Processing complete. Dominant state: {result['state_label']}, Confidence: {result['confidence']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing EEG data: {str(e)}", exc_info=True)
            raise

    def _preprocess_input(self, data: Union[str, Dict, np.ndarray, pd.DataFrame], overlap: float = 0.0, simple_mode: bool = True) -> np.ndarray:
        """
        Preprocess input data into the format expected by the model.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Preprocessed numpy array of shape (n_samples, 5)
        """
        try:
            # Handle file path
            if isinstance(data, str):
                logger.debug(f"Loading data from file: {data}")
                raw_data = load_data(data)
                # Pass raw data directly to preprocess_data
                X_normalized, _, _, _, _ = preprocess_data(raw_data, overlap=overlap, simple_mode=simple_mode)
                return X_normalized
            
            # Handle dictionary (single sample or batch)
            elif isinstance(data, dict):
                logger.debug("Processing dictionary input")
                # Check if it's a single sample
                if all(isinstance(data.get(k), (int, float)) for k in ['alpha', 'beta', 'theta', 'delta', 'gamma']):
                    # Single sample
                    features_array = np.array([[
                        float(data.get('alpha', 0)),
                        float(data.get('beta', 0)),
                        float(data.get('theta', 0)),
                        float(data.get('delta', 0)),
                        float(data.get('gamma', 0))
                    ]])
                else:
                    # Batch of samples
                    features_array = np.array([
                        [float(data['alpha']), float(data['beta']), float(data['theta']), 
                         float(data['delta']), float(data['gamma'])]
                    ])
                
                # Normalize
                return self._normalize_features(features_array)
            
            # Handle numpy array
            elif isinstance(data, np.ndarray):
                logger.debug(f"Processing numpy array of shape {data.shape}")
                if data.shape[-1] != 5:
                    raise ValueError(f"Expected 5 features (alpha, beta, theta, delta, gamma), got {data.shape[-1]}")
                return self._normalize_features(data)
            
            # Handle pandas DataFrame
            elif isinstance(data, pd.DataFrame):
                logger.debug("Processing DataFrame input")
                required_cols = ['alpha', 'beta', 'theta', 'delta', 'gamma']
                if not all(col in data.columns for col in required_cols):
                    raise ValueError(f"DataFrame must contain columns: {required_cols}")
                features_array = data[required_cols].values
                return self._normalize_features(features_array)
            
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: Raw feature array
            
        Returns:
            Normalized feature array
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        normalized = (features - mean) / (std + 1e-10)
        return normalized

    def _make_predictions(self, features: np.ndarray, model=None) -> Dict[str, Any]:
        """
        Make predictions using the provided model.
        
        Args:
            features: Preprocessed feature array of shape (n_samples, 5)
            model: Model instance to use for inference
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        try:
            if model is None:
                logger.warning("No model provided, using rule-based classification")
                return self._rule_based_classification(features)
            
            # Reshape for model input (n_samples, 5, 1)
            features_reshaped = features.reshape(-1, 5, 1)
            
            # Make predictions
            predictions = model.predict(features_reshaped, verbose=0)
            
            # Get predicted classes
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate confidence (mean of max probabilities)
            confidences = np.max(predictions, axis=1)
            mean_confidence = np.mean(confidences)
            
            # Get dominant state (most common prediction)
            dominant_state = int(np.bincount(predicted_classes).argmax())
            
            return {
                'predicted_states': predicted_classes,
                'probabilities': predictions,
                'confidence': float(mean_confidence * 100),  # Convert to percentage
                'dominant_state': dominant_state
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            # Fallback to rule-based classification
            return self._rule_based_classification(features)
    
    def _rule_based_classification(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Fallback rule-based classification when model is not available.
        
        Args:
            features: Feature array of shape (n_samples, 5)
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        logger.info("Using rule-based classification")
        
        # Extract frequency bands (alpha, beta, theta, delta, gamma)
        alpha = features[:, 0]
        beta = features[:, 1]
        theta = features[:, 2]
        
        # Calculate ratios
        beta_alpha_ratio = beta / (alpha + 1e-10)
        theta_beta_ratio = theta / (beta + 1e-10)
        
        # Classify states
        states = np.zeros(len(features), dtype=int)
        
        # Relaxation: high alpha, low beta
        states[beta_alpha_ratio < 0.5] = 0
        
        # Attention: high beta, low theta
        states[(beta_alpha_ratio > 1.2) & (theta_beta_ratio < 0.5)] = 1
        
        # Stress: high beta, high theta
        states[(beta_alpha_ratio > 1.2) & (theta_beta_ratio > 0.8)] = 2
        
        # Calculate confidence based on ratio clarity
        confidence_scores = np.abs(beta_alpha_ratio - 1.0)  # Distance from neutral
        mean_confidence = np.mean(np.clip(confidence_scores * 50, 0, 100))
        
        # Get dominant state
        dominant_state = int(np.bincount(states).argmax())
        
        return {
            'predicted_states': states,
            'probabilities': None,
            'confidence': float(mean_confidence),
            'dominant_state': dominant_state
        }

    def _get_state_label(self, state: int) -> str:
        """
        Get human-readable label for state.
        
        Args:
            state: State index (0, 1, or 2)
            
        Returns:
            State label string
        """
        labels = {
            0: "relaxed",
            1: "focused",
            2: "stressed"
        }
        return labels.get(state, "unknown")
    
    def _count_state_transitions(self, states: np.ndarray) -> int:
        """
        Count the number of state transitions.
        
        Args:
            states: Array of state predictions
            
        Returns:
            Number of transitions
        """
        if len(states) < 2:
            return 0
        transitions = np.sum(states[:-1] != states[1:])
        return int(transitions)
    
    def _calculate_cognitive_metrics(self, features: np.ndarray) -> Dict[str, float]:
        """
        Calculate cognitive metrics from EEG features.
        
        Args:
            features: Feature array of shape (n_samples, 5)
            
        Returns:
            Dictionary of cognitive metrics
        """
        try:
            # Extract frequency bands
            alpha = features[:, 0]
            beta = features[:, 1]
            theta = features[:, 2]
            delta = features[:, 3]
            gamma = features[:, 4]
            
            # Calculate metrics
            metrics = {
                'attention_index': float(np.mean(beta / (theta + alpha + 1e-10))),
                'relaxation_index': float(np.mean(alpha / (beta + 1e-10))),
                'stress_index': float(np.mean((beta + theta) / (alpha + 1e-10))),
                'cognitive_load': float(np.mean((beta + gamma) / (alpha + theta + 1e-10))),
                'mental_fatigue': float(np.mean(theta / (alpha + beta + 1e-10))),
                'alertness': float(np.mean((beta + gamma) / (delta + theta + 1e-10))),
                'mean_alpha': float(np.mean(alpha)),
                'mean_beta': float(np.mean(beta)),
                'mean_theta': float(np.mean(theta)),
                'mean_delta': float(np.mean(delta)),
                'mean_gamma': float(np.mean(gamma))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating cognitive metrics: {str(e)}")
            return {}
    
    def reload_model(self, model_path: Optional[str] = None):
        """
        Reload the model from disk.
        
        Args:
            model_path: Optional new model path. If None, uses existing path.
        """
        if model_path:
            self.model_path = model_path
        
        logger.info(f"Reloading model from {self.model_path}")
        self._load_model()
    
    def generate_detailed_report(
        self,
        data: Union[str, Dict, np.ndarray, pd.DataFrame],
        subject_id: str = "anonymous",
        session_id: str = "default_session",
        save_report: bool = False,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed report with comprehensive recommendations.
        
        Args:
            data: EEG data in various formats
            subject_id: Subject identifier
            session_id: Session identifier
            save_report: Whether to save the report to a file
            model_type: Architecture to use for analysis
            
        Returns:
            Detailed report dictionary
        """
        try:
            # Process the data first
            result = self.process_eeg_data(data, subject_id, session_id, model_type=model_type)
            
            # Generate detailed report using NLP engine
            detailed_report = self.recommendation_engine.generate_detailed_report(
                state_durations=result['state_durations'],
                total_duration=result['temporal_analysis']['total_samples'],
                confidence=result['confidence'],
                cognitive_metrics=result['cognitive_metrics'],
                state_transitions=result['temporal_analysis']['state_transitions']
            )
            
            # Merge with existing result
            detailed_report['analysis_results'] = result
            
            # Save report if requested
            if save_report:
                filepath = self.recommendation_engine.save_report(detailed_report)
                detailed_report['report_saved_to'] = filepath
            
            return detailed_report
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the ML Processor.
        
        Returns:
            Dictionary containing status information
        """
        return {
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path),
            'model_type': type(self.model).__name__ if self.model else None,
            'recommendation_engine_loaded': self.recommendation_engine is not None
        }

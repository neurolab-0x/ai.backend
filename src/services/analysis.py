import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import os

from src.preprocessing import (
    load_data,
    extract_features,
)
from src.preprocessing.labeling import label_eeg_states
from src.core.processing.temporal import temporal_smoothing, calculate_state_durations
from src.services.recommendation import NLPRecommendationEngine
from src.services.database import db_service
from src.config.settings import PROCESSING_CONFIG, THRESHOLDS
from src.queue import safe_enqueue
from src.core.ml.model_types import sanitize_model_type
from src.services.model_manager import get_model_manager
from src.utils.validation import validate_safe_id

logger = logging.getLogger(__name__)
NON_DIAGNOSTIC_DISCLAIMER = (
    "This output reflects automated signal interpretation only and is not a diagnosis "
    "or a substitute for professional medical evaluation."
)

class MLProcessor:
    """
    ML Processor for EEG data analysis pipeline.
    Handles model loading, data preprocessing, predictions, and recommendations.
    """
    
    def __init__(self, default_model: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize ML Processor.
        
        Args:
            default_model: Name of the default architecture (optional)
        """
        if model_path and not default_model:
            candidate = os.path.basename(str(model_path))
            if candidate.endswith(".h5"):
                candidate = candidate[:-3]
            elif candidate.endswith(".keras"):
                candidate = candidate[:-6]
            try:
                default_model = sanitize_model_type(candidate)
            except ValueError:
                logger.warning(f"Ignoring unsupported model_path override: {model_path}")
        self.default_model = default_model
        self.recommendation_engine = NLPRecommendationEngine()
        self.model_manager = get_model_manager()

        if default_model:
            try:
                self.model_manager.get_model(sanitize_model_type(default_model), warmup=True)
            except Exception as e:
                logger.warning(f"Default model warmup skipped: {e}")

        logger.info("ML Processor initialized")

    async def process_eeg_data(
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
            model_type = sanitize_model_type(model_type or self.default_model)
            if not model_type:  # pragma: no cover (sanitize_model_type guards)
                raise ValueError("model_type must be specified")
            subject_id = validate_safe_id(subject_id, "subject_id")
            session_id = validate_safe_id(session_id, "session_id")
            logger.info(f"Processing EEG data for subject {subject_id}, session {session_id} using model {model_type}")
            
            # Step 1: Load and preprocess data
            processed_features = self._preprocess_input(
                data,
                model_type=model_type,
                overlap=overlap,
                simple_mode=simple_mode,
            )
            
            # Step 2: Make predictions using the specified model
            model = self.model_manager.get_model(model_type, warmup=False)
            if model is None:
                raise RuntimeError("Model unavailable (TensorFlow missing or model failed to load)")
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
            
            # Step 5: Generate NLP-based recommendations (RAG with Groq)
            recommendations = await self.recommendation_engine.generate_recommendations(
                state_durations,
                total_duration,
                predictions['confidence'],
                cognitive_metrics=cognitive_metrics,
                state_transitions=state_transitions,
                subject_id=subject_id,
                session_id=session_id
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
                'wellness_recommendations': recommendations,
                'medical_disclaimer': NON_DIAGNOSTIC_DISCLAIMER,
                'metadata': {
                    'subject_id': subject_id,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type,
                    'usage_notice': "For wellness tracking and research support only.",
                }
            }
            
            # Step 7: Persistence (Async triggers)
            try:
                safe_enqueue(
                    "persistence",
                    "src.jobs.persistence.store_eeg_data",
                    cognitive_metrics,
                    subject_id,
                    session_id,
                )
                safe_enqueue(
                    "persistence",
                    "src.jobs.persistence.store_session_summary",
                    {
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "dominant_state": result["state_label"],
                        "confidence": result["confidence"],
                        "state_percentages": result["state_percentages"],
                        "timestamp": datetime.now(),
                        "type": "eeg_analysis",
                    },
                )
            except Exception as pe:
                logger.warning(f"Non-critical persistence failure: {pe}")
            
            logger.info(f"Processing complete. Dominant state: {result['state_label']}, Confidence: {result['confidence']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing EEG data: {str(e)}", exc_info=True)
            raise

    def _preprocess_input(
        self,
        data: Union[str, Dict, np.ndarray, pd.DataFrame],
        model_type: str,
        overlap: float = 0.0,
        simple_mode: bool = True,
    ) -> np.ndarray:
        """
        Preprocess input data into the format expected by the model.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Preprocessed numpy array of shape (n_samples, 5)
        """
        try:
            scaler = self.model_manager.get_scaler(model_type)
            metadata = self.model_manager.get_metadata(model_type)
            if scaler is None or metadata is None:
                raise RuntimeError(
                    f"Model artifacts incomplete for model_type={model_type}. "
                    "Expected model, scaler, and metadata artifacts."
                )
            expected_features = metadata.get("input_features") or ['alpha', 'beta', 'theta', 'delta', 'gamma']
            if not isinstance(expected_features, list) or not expected_features:
                raise RuntimeError(f"Invalid metadata for model_type={model_type}: missing input_features")

            # Handle file path
            if isinstance(data, str):
                logger.debug(f"Loading data from file: {data}")
                raw_data = load_data(data)
                if not isinstance(raw_data, pd.DataFrame):
                    raise ValueError("Loaded EEG input must be a pandas DataFrame")
                feature_df = extract_features(raw_data, simple_mode=simple_mode, overlap=overlap)
                if not isinstance(feature_df, pd.DataFrame):
                    raise ValueError("Feature extraction did not return a DataFrame")
                missing_cols = [col for col in expected_features if col not in feature_df.columns]
                if missing_cols:
                    raise ValueError(f"Extracted features missing expected columns: {missing_cols}")
                return self._normalize_features(feature_df[expected_features].values, scaler=scaler)
            
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
                if len(expected_features) != features_array.shape[1]:
                    raise ValueError(
                        f"Model {model_type} expects {len(expected_features)} features {expected_features}, "
                        f"but dictionary input provides {features_array.shape[1]}"
                    )
                return self._normalize_features(features_array, scaler=scaler)
            
            # Handle numpy array
            elif isinstance(data, np.ndarray):
                logger.debug(f"Processing numpy array of shape {data.shape}")
                if data.shape[-1] != len(expected_features):
                    raise ValueError(f"Expected {len(expected_features)} features {expected_features}, got {data.shape[-1]}")
                return self._normalize_features(data, scaler=scaler)
            
            # Handle pandas DataFrame
            elif isinstance(data, pd.DataFrame):
                logger.debug("Processing DataFrame input")
                if not all(col in data.columns for col in expected_features):
                    raise ValueError(f"DataFrame must contain columns: {expected_features}")
                features_array = data[expected_features].values
                return self._normalize_features(features_array, scaler=scaler)
            
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise
    
    def _normalize_features(self, features: np.ndarray, scaler) -> np.ndarray:
        """
        Normalize features using the scaler fitted during training.
        
        Args:
            features: Raw feature array
            
        Returns:
            Normalized feature array
        """
        if scaler is None:
            raise RuntimeError("Scaler artifact is required for inference")
        return scaler.transform(features)

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
                raise RuntimeError("No model provided for inference")
            
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
            raise RuntimeError("Model inference failed") from e
    
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
            0: "calm",
            1: "engaged",
            2: "elevated_stress"
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
            candidate = os.path.basename(str(model_path))
            if candidate.endswith(".h5"):
                candidate = candidate[:-3]
            elif candidate.endswith(".keras"):
                candidate = candidate[:-6]
            self.default_model = sanitize_model_type(candidate)

        if not self.default_model:
            raise ValueError("default_model must be set before reloading")

        logger.info(f"Reloading model artifacts for {self.default_model}")
        self.model_manager.models.pop(self.default_model, None)
        self.model_manager.scalers.pop(self.default_model, None)
        self.model_manager.metadata.pop(self.default_model, None)
        self.model_manager.get_model(self.default_model, warmup=True)
    
    async def generate_detailed_report(
        self,
        data: Union[str, Dict, np.ndarray, pd.DataFrame],
        subject_id: str = "anonymous",
        session_id: str = "default_session",
        save_report: bool = False,
        model_type: Optional[str] = None,
        overlap: float = 0.0,
        simple_mode: bool = True
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
            result = await self.process_eeg_data(
                data,
                subject_id,
                session_id,
                model_type=model_type,
                overlap=overlap,
                simple_mode=simple_mode,
            )
            
            # Generate detailed report using NLP engine
            detailed_report = await self.recommendation_engine.generate_detailed_report(
                state_durations=result['state_durations'],
                total_duration=result['temporal_analysis']['total_samples'],
                confidence=result['confidence'],
                cognitive_metrics=result['cognitive_metrics'],
                state_transitions=result['temporal_analysis']['state_transitions'],
                subject_id=subject_id,
                session_id=session_id
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
        loaded_models = list(self.model_manager.models.keys())
        model_files = self.model_manager.list_model_files()

        return {
            'model_loaded': bool(loaded_models),
            'model_path': self.model_manager.model_dir,
            'model_exists': bool(model_files),
            'model_type': loaded_models[0] if loaded_models else None,
            'models_loaded': loaded_models,
            'tensorflow_available': self.model_manager.tensorflow_available,
            'recommendation_engine_loaded': self.recommendation_engine is not None
        }

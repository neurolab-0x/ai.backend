import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

def create_model(input_shape: Tuple[int, int] = (5, 1)) -> tf.keras.Model:
    """Create a new EEG state classification model"""
    try:
        # For small input (5 features), use a simpler architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 states: relaxation, attention, stress
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def load_calibrated_model(model_path: str) -> Optional[tf.keras.Model]:
    """Load a trained and calibrated model"""
    try:
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}. Creating new model.")
            return create_model()
            
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def calibrate_model(model: tf.keras.Model, calibration_data: np.ndarray) -> tf.keras.Model:
    """Calibrate model predictions using temperature scaling"""
    try:
        # Create calibration layer
        temperature = tf.Variable(1.0, trainable=True)
        
        def calibrate_predictions(x):
            return x / temperature
            
        # Add calibration layer
        calibrated_model = tf.keras.Sequential([
            model,
            tf.keras.layers.Lambda(calibrate_predictions)
        ])
        
        # Compile calibrated model
        calibrated_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train temperature parameter
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        for _ in range(100):  # 100 calibration steps
            with tf.GradientTape() as tape:
                predictions = calibrated_model(calibration_data)
                loss = tf.keras.losses.categorical_crossentropy(
                    tf.ones_like(predictions) / predictions.shape[-1],
                    predictions
                )
            
            gradients = tape.gradient(loss, [temperature])
            optimizer.apply_gradients(zip(gradients, [temperature]))
        
        logger.info(f"Model calibration complete. Final temperature: {temperature.numpy()}")
        return calibrated_model
    except Exception as e:
        logger.error(f"Error calibrating model: {str(e)}")
        raise

def save_model(model: tf.keras.Model, model_path: str) -> None:
    """Save model to disk"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logger.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {str(e)}")
        raise

def evaluate_model(model: tf.keras.Model, test_data: np.ndarray, test_labels: np.ndarray) -> dict:
    """Evaluate model performance"""
    try:
        # Get predictions
        predictions = model.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        confusion_matrix = tf.math.confusion_matrix(
            true_classes,
            predicted_classes,
            num_classes=3
        ).numpy()
        
        # Calculate per-class metrics
        precision = []
        recall = []
        for i in range(3):
            true_positives = confusion_matrix[i, i]
            false_positives = np.sum(confusion_matrix[:, i]) - true_positives
            false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
            
            p = true_positives / (true_positives + false_positives + 1e-10)
            r = true_positives / (true_positives + false_negatives + 1e-10)
            
            precision.append(p)
            recall.append(r)
        
        return {
            "accuracy": float(accuracy),
            "precision": [float(p) for p in precision],
            "recall": [float(r) for r in recall],
            "confusion_matrix": confusion_matrix.tolist()
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise 
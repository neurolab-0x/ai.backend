"""
Improved Model Training Script with Advanced Features
- Data augmentation
- Learning rate scheduling
- Early stopping with patience
- Model checkpointing
- Cross-validation
- Advanced metrics tracking
- Hyperparameter optimization support
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, Tuple, List, Any

# Add project root to path
# The script is in scripts/training/ - we need to go up two levels to reach the root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import preprocessing modules
from src.preprocessing.load_data import load_data
from src.preprocessing.features import extract_features
from src.preprocessing.preprocess import preprocess_data
from src.preprocessing.labeling import label_eeg_states

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImprovedModelTrainer:
    """Advanced model trainer with enhanced features"""
    
    def __init__(
        self,
        model_save_path: str = "model/trained_model.h5",
        checkpoint_dir: str = "./checkpoints",
        results_dir: str = "./training_results"
    ):
        """
        Initialize the improved trainer
        
        Args:
            model_save_path: Path to save the final model
            checkpoint_dir: Directory for model checkpoints
            results_dir: Directory for training results and plots
        """
        self.model_save_path = model_save_path
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        self.model = None
        self.history = None
        self.training_config = {}
        
        logger.info("Improved Model Trainer initialized")
    
    def build_advanced_model(
        self,
        input_shape: Tuple[int, int],
        num_classes: int = 3,
        architecture: str = "lstm_attention"
    ) -> keras.Model:
        """
        Build advanced model architecture
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            num_classes: Number of output classes
            architecture: Model architecture type
            
        Returns:
            Compiled Keras model
        """
        if architecture == "lstm_attention":
            model = self._build_lstm_attention_model(input_shape, num_classes)
        elif architecture == "bidirectional_lstm":
            model = self._build_bidirectional_lstm_model(input_shape, num_classes)
        elif architecture == "cnn_lstm":
            model = self._build_cnn_lstm_model(input_shape, num_classes)
        else:
            model = self._build_lstm_attention_model(input_shape, num_classes)
        
        logger.info(f"Built {architecture} model with input shape {input_shape}")
        return model
    
    def _build_lstm_attention_model(
        self,
        input_shape: Tuple[int, int],
        num_classes: int
    ) -> keras.Model:
        """Build LSTM model with attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(64)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_attention')
        
        # Compile with advanced optimizer
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def _build_bidirectional_lstm_model(
        self,
        input_shape: Tuple[int, int],
        num_classes: int
    ) -> keras.Model:
        """Build Bidirectional LSTM model"""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ], name='bidirectional_lstm')
        
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def _build_cnn_lstm_model(
        self,
        input_shape: Tuple[int, int],
        num_classes: int
    ) -> keras.Model:
        """Build CNN-LSTM hybrid model"""
        inputs = layers.Input(shape=input_shape)
        
        # CNN layers for feature extraction
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        
        # LSTM layers for temporal modeling
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_lstm')
        
        optimizer = optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def create_callbacks(self, patience: int = 15) -> List[callbacks.Callback]:
        """
        Create training callbacks
        
        Args:
            patience: Patience for early stopping
            
        Returns:
            List of Keras callbacks
        """
        callback_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=os.path.join(self.results_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                os.path.join(self.results_dir, 'training_log.csv'),
                append=True
            )
        ]
        
        return callback_list
    
    def augment_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data with noise and transformations
        
        Args:
            X: Input features
            y: Labels
            augmentation_factor: How many augmented samples per original
            
        Returns:
            Augmented X and y
        """
        logger.info(f"Augmenting data with factor {augmentation_factor}")
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augmentation_factor):
            # Add Gaussian noise
            noise = np.random.normal(0, 0.05, X.shape)
            X_noisy = X + noise
            X_augmented.append(X_noisy)
            y_augmented.append(y)
            
            # Time shifting (if applicable)
            if X.shape[1] > 1:
                shift = np.random.randint(-2, 3)
                X_shifted = np.roll(X, shift, axis=1)
                X_augmented.append(X_shifted)
                y_augmented.append(y)
        
        X_final = np.concatenate(X_augmented, axis=0)
        y_final = np.concatenate(y_augmented, axis=0)
        
        logger.info(f"Data augmented from {X.shape[0]} to {X_final.shape[0]} samples")
        return X_final, y_final
    
    def train_with_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        architecture: str = "lstm_attention"
    ) -> Dict[str, Any]:
        """
        Train model with k-fold cross-validation
        
        Args:
            X: Input features
            y: Labels
            n_splits: Number of CV folds
            epochs: Training epochs
            batch_size: Batch size
            architecture: Model architecture
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Starting {n_splits}-fold cross-validation")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        fold_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"Training fold {fold}/{n_splits}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Convert labels to categorical
            y_train_cat = to_categorical(y_train_fold, num_classes=3)
            y_val_cat = to_categorical(y_val_fold, num_classes=3)
            
            # Build model
            model = self.build_advanced_model(
                input_shape=(X_train_fold.shape[1], X_train_fold.shape[2]),
                num_classes=3,
                architecture=architecture
            )
            
            # Train
            history = model.fit(
                X_train_fold, y_train_cat,
                validation_data=(X_val_fold, y_val_cat),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.create_callbacks(patience=10),
                verbose=0
            )
            
            # Evaluate
            val_loss, val_acc, val_precision, val_recall = model.evaluate(
                X_val_fold, y_val_cat, verbose=0
            )
            
            cv_scores.append({
                'fold': fold,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'f1_score': 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)
            })
            
            fold_histories.append(history.history)
            
            logger.info(f"Fold {fold} - Val Acc: {val_acc:.4f}, F1: {cv_scores[-1]['f1_score']:.4f}")
        
        # Calculate mean scores
        mean_scores = {
            'mean_val_accuracy': np.mean([s['val_accuracy'] for s in cv_scores]),
            'std_val_accuracy': np.std([s['val_accuracy'] for s in cv_scores]),
            'mean_f1_score': np.mean([s['f1_score'] for s in cv_scores]),
            'std_f1_score': np.std([s['f1_score'] for s in cv_scores])
        }
        
        logger.info(f"CV Results - Mean Acc: {mean_scores['mean_val_accuracy']:.4f} ± {mean_scores['std_val_accuracy']:.4f}")
        logger.info(f"CV Results - Mean F1: {mean_scores['mean_f1_score']:.4f} ± {mean_scores['std_f1_score']:.4f}")
        
        return {
            'fold_scores': cv_scores,
            'mean_scores': mean_scores,
            'fold_histories': fold_histories
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        architecture: str = "lstm_attention",
        use_augmentation: bool = True
    ) -> keras.Model:
        """
        Train the model with advanced features
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            architecture: Model architecture
            use_augmentation: Whether to use data augmentation
            
        Returns:
            Trained model
        """
        logger.info("Starting model training")
        
        # Data augmentation
        if use_augmentation:
            X_train, y_train = self.augment_data(X_train, y_train, augmentation_factor=1)
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=3)
        y_val_cat = to_categorical(y_val, num_classes=3)
        
        # Build model
        self.model = self.build_advanced_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=3,
            architecture=architecture
        )
        
        # Print model summary
        self.model.summary(print_fn=logger.info)
        
        # Train
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.create_callbacks(),
            verbose=1
        )
        
        # Save training config
        self.training_config = {
            'architecture': architecture,
            'epochs': epochs,
            'batch_size': batch_size,
            'use_augmentation': use_augmentation,
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model
        self.model.save(self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")
        
        return self.model

    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model")
        
        # Convert labels
        y_test_cat = to_categorical(y_test, num_classes=3)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            X_test, y_test_cat, verbose=0
        )
        
        # Classification report
        class_names = ['Relaxed', 'Focused', 'Stressed']
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # F1 scores
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
        
        return results
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        class_names = ['Relaxed', 'Focused', 'Stressed']
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
        plt.close()
    
    def save_results(self, results: Dict[str, Any], filename: str = 'training_results.json'):
        """Save training results to JSON"""
        filepath = os.path.join(self.results_dir, filename)
        
        # Combine with training config
        full_results = {
            'training_config': self.training_config,
            'evaluation_results': results,
            'model_path': self.model_save_path
        }
        
        with open(filepath, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath


def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("Starting Improved Model Training Pipeline")
    logger.info("=" * 60)
    
    # Configuration
    DATA_PATH = "data/training_data/training.csv"
    ARCHITECTURE = "lstm_attention"  # Options: lstm_attention, bidirectional_lstm, cnn_lstm
    EPOCHS = 100
    BATCH_SIZE = 32
    USE_AUGMENTATION = True
    USE_CROSS_VALIDATION = False  # Set to True for CV
    
    try:
        # Load data
        logger.info(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        
        # Prepare features and labels
        feature_cols = ['alpha', 'beta', 'theta', 'delta', 'gamma']
        X = df[feature_cols].values
        y = df['state'].values
        
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        
        # Reshape for LSTM (samples, timesteps, features)
        X_reshaped = X_normalized.reshape(-1, 5, 1)
        
        # Initialize trainer
        trainer = ImprovedModelTrainer()
        
        if USE_CROSS_VALIDATION:
            # Cross-validation
            cv_results = trainer.train_with_cross_validation(
                X_reshaped, y,
                n_splits=5,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                architecture=ARCHITECTURE
            )
            
            # Save CV results
            trainer.save_results(cv_results, 'cv_results.json')
            
        else:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_reshaped, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            
            # Further split train into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=42,
                stratify=y_train
            )
            
            logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            
            # Train model
            model = trainer.train(
                X_train, y_train,
                X_val, y_val,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                architecture=ARCHITECTURE,
                use_augmentation=USE_AUGMENTATION
            )
            
            # Plot training history
            trainer.plot_training_history()
            
            # Evaluate
            results = trainer.evaluate(X_test, y_test)
            
            # Plot confusion matrix
            cm = np.array(results['confusion_matrix'])
            trainer.plot_confusion_matrix(cm)
            
            # Save results
            trainer.save_results(results)
            
            # Print summary
            logger.info("=" * 60)
            logger.info("Training Complete!")
            logger.info("=" * 60)
            logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
            logger.info(f"F1 Score (Macro): {results['f1_macro']:.4f}")
            logger.info(f"Model saved to: {trainer.model_save_path}")
            logger.info(f"Results saved to: {trainer.results_dir}")
            logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

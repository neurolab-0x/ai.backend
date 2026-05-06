import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import preprocessing modules
from src.preprocessing.load_data import load_data
from src.preprocessing.features import extract_features
from src.preprocessing.preprocess import preprocess_data
from src.preprocessing.labeling import label_eeg_states
from src.core.ml.model import (
    DEFAULT_FEATURE_NAMES,
    build_model,
    evaluate_model,
    get_model_artifact_paths,
    save_model_metadata,
    save_scaler_artifact,
)
from src.services.storage import MinioStorageService

def setup_logging(log_file: str = 'training_improved.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ImprovedModelTrainer:
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        results_dir: str = "./training_results"
    ):
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        self.model = None
        self.history = None
        self.training_config = {}
        logger.info("Improved Model Trainer initialized")

    def create_callbacks(self, patience: int = 15, model_type: str = "model") -> List[callbacks.Callback]:
        return [
            callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, f'{model_type}_best.keras'),
                monitor='val_accuracy', save_best_only=True, verbose=1
            ),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            callbacks.TensorBoard(log_dir=os.path.join(self.results_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))),
            callbacks.CSVLogger(os.path.join(self.results_dir, 'training_log.csv'), append=True)
        ]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        args: argparse.Namespace,
        preprocessing_metadata: Optional[Dict[str, Any]] = None,
    ) -> keras.Model:
        logger.info(f"Starting training for {args.model_type}")
        
        input_shape = (X_train.shape[1], X_train.shape[2]) if len(X_train.shape) > 2 else (X_train.shape[1], 1)
        num_classes = len(np.unique(y_train))
        
        artifact_paths = get_model_artifact_paths(args.model_type, base_dir="model")
        save_path = artifact_paths["model_path"]
        legacy_save_path = artifact_paths["legacy_model_path"]
        
        # Load existing model if resume is requested or if model exists
        resume_candidates = [save_path, legacy_save_path]
        existing_model_path = next((path for path in resume_candidates if os.path.exists(path)), None)
        if args.resume and existing_model_path:
            logger.info(f"Resuming training: Loading existing model from {existing_model_path}")
            try:
                self.model = keras.models.load_model(existing_model_path)
                # Recompile to ensure correct optimizer and loss (prevents potential issues)
                self.model.compile(
                    optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy']
                )
                logger.info("Successfully loaded and recompiled model for continuous training")
            except Exception as e:
                logger.error(f"Failed to load existing model: {str(e)}. Building fresh model instead.")
                self.model = build_model(
                    model_type=args.model_type,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    dropout_rate=args.dropout,
                    l1_reg=args.l1,
                    l2_reg=args.l2
                )
        else:
            if args.resume:
                logger.warning(f"Resume requested but model file not found at {save_path}. Building fresh model.")
            
            self.model = build_model(
                model_type=args.model_type,
                input_shape=input_shape,
                num_classes=num_classes,
                dropout_rate=args.dropout,
                l1_reg=args.l1,
                l2_reg=args.l2
            )
        
        self.model.summary(print_fn=logger.info)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=self.create_callbacks(patience=args.patience, model_type=args.model_type),
            verbose=1
        )
        
        os.makedirs(artifact_paths["artifact_dir"], exist_ok=True)
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

        feature_names = (
            (preprocessing_metadata or {}).get("feature_names")
            or DEFAULT_FEATURE_NAMES
        )
        scaler = (preprocessing_metadata or {}).get("scaler")
        if scaler is None:
            raise RuntimeError(
                "Preprocessing scaler missing from training metadata; cannot export inference artifacts"
            )

        save_scaler_artifact(args.model_type, scaler, base_dir="model")
        metadata_payload = {
            "input_features": feature_names,
            "scaler": scaler.__class__.__name__,
            "model_type": args.model_type,
            "trained_at": datetime.now().isoformat(),
            "source": "standalone_training_script",
            "dataset_path": args.data,
            "validation_split": args.val_split,
            "simple_mode": args.simple_mode,
            "overlap": args.overlap,
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "dropout": args.dropout,
                "l1": args.l1,
                "l2": args.l2,
                "seed": args.seed,
            },
        }
        if preprocessing_metadata:
            metadata_payload["dataset_metadata"] = {
                key: value
                for key, value in preprocessing_metadata.items()
                if key != "scaler"
            }
        save_model_metadata(args.model_type, metadata_payload, base_dir="model")
        logger.info("Saved scaler and metadata artifacts for %s", args.model_type)

        storage = MinioStorageService()
        if storage.enabled:
            uploads = {
                "model": storage.upload_file(
                    artifact_paths["model_path"],
                    "models",
                    f"active/{args.model_type}/model.keras",
                ),
                "scaler": storage.upload_file(
                    artifact_paths["scaler_path"],
                    "models",
                    f"active/{args.model_type}/scaler.joblib",
                ),
                "metadata": storage.upload_file(
                    artifact_paths["metadata_path"],
                    "models",
                    f"active/{args.model_type}/metadata.json",
                ),
            }
            if all(uploads.values()):
                logger.info("Uploaded active model artifacts to MinIO for %s", args.model_type)
            else:
                logger.warning(
                    "Failed to upload one or more active artifacts to MinIO for %s",
                    args.model_type,
                )
        
        self.training_config = vars(args)
        self.training_config['timestamp'] = datetime.now().isoformat()
        
        return self.model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        logger.info("Evaluating model")
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        eval_metrics = self.model.evaluate(X_test, y_test, verbose=0)
        
        class_names = ['Relaxed', 'Focused', 'Stressed']
        # Filter class names based on unique labels in y_test to avoid classification_report error
        unique_labels = np.unique(y_test)
        target_names = [class_names[i] for i in unique_labels]
        
        report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'test_loss': float(eval_metrics[0]),
            'test_accuracy': float(eval_metrics[1]),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Generate interpretability analysis
        try:
            from src.core.ml.interpretability import ModelInterpretability
            
            logger.info("Generating SHAP feature importance analysis...")
            interpreter = ModelInterpretability(self.model)
            interpreter.set_feature_names(['alpha', 'beta', 'theta', 'delta', 'gamma'])
            
            # Generate SHAP explanations
            shap_results = interpreter.explain_with_shap(X_test, n_samples=min(50, len(X_test)))
            
            # Extract feature importance (convert to serializable format)
            feature_importance = {}
            for class_idx, importance_array in shap_results['feature_importance'].items():
                if isinstance(importance_array, np.ndarray):
                    feature_importance[f'class_{class_idx}'] = {
                        'alpha': float(importance_array[0]) if len(importance_array) > 0 else 0.0,
                        'beta': float(importance_array[1]) if len(importance_array) > 1 else 0.0,
                        'theta': float(importance_array[2]) if len(importance_array) > 2 else 0.0,
                        'delta': float(importance_array[3]) if len(importance_array) > 3 else 0.0,
                        'gamma': float(importance_array[4]) if len(importance_array) > 4 else 0.0,
                    }
            
            results['interpretability'] = {
                'method': 'shap',
                'feature_importance_by_class': feature_importance,
                'n_samples_analyzed': min(50, len(X_test))
            }
            
            logger.info("Interpretability analysis complete")
            
        except Exception as e:
            logger.warning(f"Failed to generate interpretability analysis: {str(e)}")
            results['interpretability'] = {'error': str(e)}
        
        logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
        return results

    def plot_results(self, results: Dict[str, Any]):
        # History plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='train')
        plt.plot(self.history.history['val_accuracy'], label='val')
        plt.title('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='val')
        plt.title('Loss')
        plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, 'history.png'))
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.array(results['confusion_matrix']), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Neurolab AI Model Training Script")
    # Basic Config
    parser.add_argument("--data", type=str, default="data/training_data/training.csv", help="Path to training CSV")
    parser.add_argument("--model-type", type=str, default="enhanced_cnn_lstm", 
                        choices=['original', 'enhanced_cnn_lstm', 'resnet_lstm', 'transformer'], help="Architecture")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing model if available")
    
    # Training Params
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--l1", type=float, default=1e-5, help="L1 regularization")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization")
    
    # Preprocessing Params
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap (0.0 to 0.9)")
    parser.add_argument("--simple-mode", action="store_true", default=True, help="Use simple feature extraction")
    parser.add_argument("--no-simple-mode", action="store_false", dest="simple_mode", help="Use complex feature extraction")
    
    # Validation Config
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(f"Starting Training: {args.model_type} (Resume: {args.resume})")
    logger.info("=" * 60)
    
    try:
        df = pd.read_csv(args.data)
        df = label_eeg_states(df)
        
        # Pass raw data through unified preprocessing
        X_train, X_test, y_train, y_test, metadata = preprocess_data(
            df, 
            overlap=args.overlap, 
            simple_mode=args.simple_mode,
            test_size=args.val_split,
            random_state=args.seed
        )
        
        # Reshape for LSTM if needed (samples, features, 1)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
        # Split train further into train/val
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train
        )
        
        trainer = ImprovedModelTrainer()
        trainer.train(
            X_train_final,
            y_train_final,
            X_val,
            y_val,
            args,
            preprocessing_metadata=metadata,
        )
        
        results = trainer.evaluate(X_test, y_test)
        trainer.plot_results(results)
        
        with open(os.path.join(trainer.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info("✓ Training Complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

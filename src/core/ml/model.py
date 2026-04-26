from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import compute_class_weight
import numpy as np
import math
import time
import os
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any
from src.core.ml.model_types import sanitize_model_type

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Input, Dropout,
        BatchNormalization, Activation, Add, Concatenate, Attention,
        MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Bidirectional,
        SpatialDropout1D, GaussianNoise, TimeDistributed, SeparableConv1D, Lambda
    )
    from tensorflow.keras.callbacks import (
        ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,
        TensorBoard, LearningRateScheduler
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1_l2
    from tensorflow.keras import Sequential
    TF_AVAILABLE = True
except ImportError:
    tf = None
    Conv1D = MaxPooling1D = Flatten = Dense = LSTM = Input = Dropout = None
    BatchNormalization = Activation = Add = Concatenate = Attention = None
    MultiHeadAttention = LayerNormalization = GlobalAveragePooling1D = None
    Bidirectional = SpatialDropout1D = GaussianNoise = TimeDistributed = None
    SeparableConv1D = Lambda = None
    ReduceLROnPlateau = EarlyStopping = ModelCheckpoint = TensorBoard = LearningRateScheduler = None
    Model = Adam = l1_l2 = Sequential = None
    TF_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


def _require_tensorflow():
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed")

def cosine_annealing_schedule(epoch, lr):
    """Cosine annealing learning rate schedule."""
    initial_lr = 0.001
    min_lr = 1e-6
    max_epochs = 100
    return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / max_epochs)) / 2

def residual_block(x, filters, kernel_size=3, dropout_rate=0.3, use_separable=False):
    """Enhanced residual block with separable convolutions and better regularization."""
    shortcut = x
    
    # First conv layer
    if use_separable:
        x = SeparableConv1D(filters, kernel_size, padding='same')(x)
    else:
        x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)
    
    # Second conv layer
    if use_separable:
        x = SeparableConv1D(filters, kernel_size, padding='same')(x)
    else:
        x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Skip connection - ensure dimensions match
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add skip connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def transformer_block(x, embed_dim, num_heads, ff_dim, dropout=0.1, use_relative_pos=True):
    """Enhanced transformer block with relative positional encoding."""
    # Multi-head attention with relative positional encoding
    if use_relative_pos:
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads,
            attention_axes=(1,)  # Apply attention along sequence dimension
        )(x, x, x)
    else:
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim//num_heads
        )(x, x)
    
    attention_output = Dropout(dropout)(attention_output)
    
    # Add & norm (first residual connection)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed-forward network with GELU activation
    ffn_output = Dense(ff_dim, activation='gelu')(x)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    
    # Add & norm (second residual connection)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    return x

def attention_lstm_layer(x, units, dropout_rate=0.3):
    """Enhanced LSTM layer with multi-head attention and better regularization."""
    # Bidirectional LSTM with residual connection
    lstm_out = Bidirectional(LSTM(units, return_sequences=True))(x)
    lstm_out = SpatialDropout1D(dropout_rate)(lstm_out)
    
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=units//4
    )(lstm_out, lstm_out)
    attention_output = Dropout(dropout_rate)(attention_output)
    
    # Add residual connection
    lstm_out = Add()([lstm_out, attention_output])
    lstm_out = LayerNormalization(epsilon=1e-6)(lstm_out)
    
    # Global context attention
    context_vector = Dense(1, activation='tanh')(lstm_out)
    attention_weights = Activation('softmax')(context_vector)
    context = tf.multiply(lstm_out, attention_weights)
    context = tf.reduce_sum(context, axis=1)
    
    return context

def build_model(model_type='enhanced_cnn_lstm', input_shape=(5, 1), num_classes=3, **kwargs):
    """Build model architecture without training."""
    _require_tensorflow()
    dropout_rate = kwargs.get('dropout_rate', 0.3)
    use_separable = kwargs.get('use_separable', True)
    use_relative_pos = kwargs.get('use_relative_pos', True)
    l1_reg = kwargs.get('l1_reg', 1e-5)
    l2_reg = kwargs.get('l2_reg', 1e-4)

    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.01)(inputs)
    
    if model_type == 'original':
        x = Conv1D(64, kernel_size=3, padding='same', activation='relu', 
                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(dropout_rate)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'enhanced_cnn_lstm':
        x = SeparableConv1D(32, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = SpatialDropout1D(dropout_rate)(x)
        x = SeparableConv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = SpatialDropout1D(dropout_rate)(x)
        x = SeparableConv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(dropout_rate)(x)
        x = attention_lstm_layer(x, units=64, dropout_rate=dropout_rate)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'resnet_lstm':
        x = Conv1D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(dropout_rate)(x)
        x = residual_block(x, filters=64, dropout_rate=dropout_rate, use_separable=use_separable)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = residual_block(x, filters=128, dropout_rate=dropout_rate, use_separable=use_separable)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = attention_lstm_layer(x, units=64, dropout_rate=dropout_rate)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
    elif model_type == 'transformer':
        x = SeparableConv1D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = SpatialDropout1D(dropout_rate)(x)
        embed_dim = 64
        x = transformer_block(x, embed_dim=embed_dim, num_heads=4, ff_dim=128, dropout=dropout_rate, use_relative_pos=use_relative_pos)
        x = transformer_block(x, embed_dim=embed_dim, num_heads=4, ff_dim=128, dropout=dropout_rate, use_relative_pos=use_relative_pos)
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(x)
        x = Dropout(dropout_rate)(x)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Alias for backward compatibility
create_model = build_model

def train_hybrid_model(X_train, y_train, model_type='enhanced_cnn_lstm', **kwargs):
    """Enhanced hybrid model training with improved architectures."""
    _require_tensorflow()
    batch_size = kwargs.get('batch_size', 32)
    epochs = kwargs.get('epochs', 30)
    validation_data = kwargs.get('validation_data')
    
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(-1, X_train.shape[1], 1)
    if validation_data is not None:
        X_val, y_val = validation_data
        if len(X_val.shape) == 2:
            X_val = X_val.reshape(-1, X_val.shape[1], 1)
        validation_data = (X_val, y_val)
    
    input_shape = (X_train.shape[1], 1)
    num_classes = len(np.unique(y_train))
    
    model_path = os.path.join("model", f"{model_type}.h5")
    
    # Try to load existing model for continuous training
    if os.path.exists(model_path):
        logger.info(f"Loading existing {model_type} model for continuous training...")
        try:
            model = tf.keras.models.load_model(model_path)
            model.compile(optimizer=Adam(learning_rate=kwargs.get('learning_rate', 0.001)), 
                         loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}. Building new model.")
            model = build_model(model_type=model_type, input_shape=input_shape, num_classes=num_classes, **kwargs)
    else:
        model = build_model(model_type=model_type, input_shape=input_shape, num_classes=num_classes, **kwargs)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    callbacks = [
        LearningRateScheduler(cosine_annealing_schedule),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=f'./logs/{model_type}')
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.0 if validation_data is not None else 0.2,
        validation_data=validation_data,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    save_model(model, model_path)
    return model, history

def save_model(model, model_path: str) -> None:
    """Save model to disk with directory creation."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logger.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_calibrated_model(model_path: str) -> Optional[tf.keras.Model]:
    """Load a trained model, fallback to a base model if not found."""
    try:
        if not TF_AVAILABLE:
            logger.warning("TensorFlow is not installed; calibrated models are unavailable")
            return None

        base_dir = os.path.abspath("model")

        # Treat input as a model identifier/basename, never as a full path.
        model_name = os.path.basename(str(model_path))
        if model_name.endswith('.h5'):
            model_name = model_name[:-3]
        model_name = sanitize_model_type(model_name)

        normalized_model_path = os.path.join(base_dir, f"{model_name}.h5")

        if not os.path.exists(normalized_model_path):
            logger.warning(f"Model file not found at {normalized_model_path}. Creating base model.")
            return build_model(model_name)
        model = tf.keras.models.load_model(normalized_model_path)
        logger.info(f"Model loaded successfully from {normalized_model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def calibrate_model(model, calibration_data: np.ndarray, calibration_labels: np.ndarray):
    """Calibrate model predictions using temperature scaling."""
    try:
        _require_tensorflow()
        calibration_labels = np.asarray(calibration_labels)
        if calibration_labels.size == 0:
            raise ValueError("Calibration labels are required for model calibration")

        temperature = tf.Variable(1.0, trainable=True)
        calibrated_model = Sequential([model, Lambda(lambda x: x / temperature)])
        calibrated_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        for _ in range(100):
            with tf.GradientTape() as tape:
                predictions = calibrated_model(calibration_data)
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(calibration_labels, predictions)
                )
            gradients = tape.gradient(loss, [temperature])
            optimizer.apply_gradients(zip(gradients, [temperature]))
        
        logger.info(f"Model calibration complete. Temperature: {temperature.numpy()}")
        return calibrated_model
    except Exception as e:
        logger.error(f"Error calibrating model: {str(e)}")
        raise

def evaluate_model(model, test_data: np.ndarray, test_labels: np.ndarray, calibrate=True) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    _require_tensorflow()
    if len(test_data.shape) == 2:
        test_data = test_data.reshape(-1, test_data.shape[1], 1)
        
    y_pred_proba = model.predict(test_data)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(test_labels, y_pred)
    report = classification_report(test_labels, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report
    }
    
    try:
        y_test_oh = tf.keras.utils.to_categorical(test_labels)
        metrics['roc_auc'] = float(roc_auc_score(y_test_oh, y_pred_proba, multi_class='ovr'))
    except:
        metrics['roc_auc'] = None
        
    return metrics

def model_comparison(X_train, y_train, X_test, y_test, n_repeats=3):
    """Compare multiple model architectures."""
    _require_tensorflow()
    model_types = ['original', 'enhanced_cnn_lstm', 'resnet_lstm', 'transformer']
    results = {}
    
    # Determine input shape from data
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(-1, X_train.shape[1], 1)
        X_test = X_test.reshape(-1, X_test.shape[1], 1)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    
    for model_type in model_types:
        logger.info(f"Comparing {model_type} model...")
        accuracies = []
        for i in range(n_repeats):
            # Build fresh model for comparison (don't load existing)
            model = build_model(
                model_type=model_type,
                input_shape=input_shape,
                num_classes=num_classes
            )
            
            # Train the model
            model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            accuracies.append(metrics['accuracy'])
            
        results[model_type] = {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies))
        }
        logger.info(f"{model_type}: {results[model_type]['mean_accuracy']:.4f} ± {results[model_type]['std_accuracy']:.4f}")
        
    return results

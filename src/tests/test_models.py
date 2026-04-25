"""
Tests for model training and evaluation functions.
"""
import unittest
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.core.ml.model import (
        save_model as save_trained_model,
        cosine_annealing_schedule,
        residual_block,
        transformer_block,
        attention_lstm_layer,
        train_hybrid_model,
        evaluate_model
    )
    import tensorflow as tf
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Warning: Could not import models module: {e}")

@unittest.skipIf(not MODELS_AVAILABLE, "Models module not available")
class TestModelFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_trained_model(self):
        """Test model saving functionality"""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5, 1)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.h5")
        save_trained_model(model, model_path)
        
        # Verify model was saved
        self.assertTrue(os.path.exists(model_path))
        
        # Load and verify
        loaded_model = tf.keras.models.load_model(model_path)
        self.assertEqual(len(model.layers), len(loaded_model.layers))
    
    def test_cosine_annealing_schedule(self):
        """Test cosine annealing learning rate schedule"""
        initial_lr = 0.001
        
        # Test at different epochs
        lr_epoch_0 = cosine_annealing_schedule(0, initial_lr)
        lr_epoch_50 = cosine_annealing_schedule(50, initial_lr)
        lr_epoch_100 = cosine_annealing_schedule(100, initial_lr)
        
        # Learning rate should decrease over time
        self.assertGreater(lr_epoch_0, lr_epoch_50)
        self.assertGreater(lr_epoch_50, lr_epoch_100)
        
        # Learning rate should be positive
        self.assertGreater(lr_epoch_0, 0)
        self.assertGreater(lr_epoch_100, 0)
    
    def test_get_channel_config(self):
        """Test channel configuration retrieval"""
        # Test supported configurations
        config_8 = get_channel_config(8)
        config_16 = get_channel_config(16)
        config_32 = get_channel_config(32)
        config_64 = get_channel_config(64)
        
        # Verify configurations have required keys
        for config in [config_8, config_16, config_32, config_64]:
            self.assertIn('conv_filters', config)
            self.assertIn('lstm_units', config)
            self.assertIn('dense_units', config)
            self.assertIn('attention_heads', config)
        
        # Verify larger channel counts have larger model configurations
        self.assertLessEqual(config_8['lstm_units'], config_64['lstm_units'])
    
    def test_train_hybrid_model(self):
        """Test hybrid model training"""
        # Generate dummy training data
        X_train = np.random.randn(50, 5)
        y_train = np.random.randint(0, 3, 50)
        
        # Train model with minimal epochs
        model, history = train_hybrid_model(
            X_train, y_train,
            model_type='original',
            epochs=2,
            batch_size=16
        )
        
        # Verify model was created
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        
        # Verify model can make predictions
        X_test = np.random.randn(10, 5).reshape(-1, 5, 1)
        predictions = model.predict(X_test)
        
        self.assertEqual(predictions.shape[0], 10)
        self.assertEqual(predictions.shape[1], 3)  # 3 classes
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Generate dummy test data
        X_test = np.random.randn(30, 5)
        y_test = np.random.randint(0, 3, 30)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, calibrate=False)
        
        # Verify metrics were calculated
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        
        # Verify accuracy is between 0 and 1
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)

if __name__ == '__main__':
    unittest.main()

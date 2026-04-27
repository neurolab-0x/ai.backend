"""Tests for model training and evaluation functions."""

import os
import sys
import tempfile
import unittest

import pytest

np = pytest.importorskip("numpy")

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.core.ml.model import (
        build_model,
        cosine_annealing_schedule,
        evaluate_model,
        save_model as save_trained_model,
        train_hybrid_model,
    )
    tf = pytest.importorskip("tensorflow")

    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Warning: Could not import models module: {e}")


@unittest.skipIf(not MODELS_AVAILABLE, "Models module not available")
class TestModelFunctions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_trained_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(5, 1)),
                tf.keras.layers.Conv1D(32, 3, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )

        model_path = os.path.join(self.temp_dir, "test_model.h5")
        save_trained_model(model, model_path)

        self.assertTrue(os.path.exists(model_path))

        loaded_model = tf.keras.models.load_model(model_path)
        self.assertEqual(len(model.layers), len(loaded_model.layers))

    def test_cosine_annealing_schedule(self):
        initial_lr = 0.001

        lr_epoch_0 = cosine_annealing_schedule(0, initial_lr)
        lr_epoch_50 = cosine_annealing_schedule(50, initial_lr)
        lr_epoch_100 = cosine_annealing_schedule(100, initial_lr)

        self.assertGreater(lr_epoch_0, lr_epoch_50)
        self.assertGreater(lr_epoch_50, lr_epoch_100)
        self.assertGreater(lr_epoch_0, 0)
        self.assertGreater(lr_epoch_100, 0)

    def test_build_model_with_supported_inputs(self):
        model_small = build_model(input_shape=(8, 1), num_classes=3)
        model_large = build_model(input_shape=(64, 1), num_classes=3)

        self.assertEqual(model_small.output_shape[-1], 3)
        self.assertEqual(model_large.output_shape[-1], 3)

    def test_train_hybrid_model(self):
        X_train = np.random.randn(50, 5)
        y_train = np.random.randint(0, 3, 50)

        model, history = train_hybrid_model(
            X_train,
            y_train,
            model_type="original",
            epochs=2,
            batch_size=16,
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

        X_test = np.random.randn(10, 5).reshape(-1, 5, 1)
        predictions = model.predict(X_test)

        self.assertEqual(predictions.shape[0], 10)
        self.assertEqual(predictions.shape[1], 3)

    def test_evaluate_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(5, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        X_test = np.random.randn(30, 5)
        y_test = np.random.randint(0, 3, 30)

        metrics = evaluate_model(model, X_test, y_test, calibrate=False)

        self.assertIn("accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("classification_report", metrics)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()

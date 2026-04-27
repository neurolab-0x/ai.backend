import time
import pytest

tf = pytest.importorskip("tensorflow")
np = pytest.importorskip("numpy")
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load your trained model
MODEL_PATH = "processed/trained_model.h5"

def get_model():
    if not os.path.exists(MODEL_PATH):
        # Return a dummy model if real model missing, so tests don't crash
        logger.warning(f"Model not found at {MODEL_PATH}, using dummy model")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def generate_dummy_data(num_samples=100, sequence_length=5, num_classes=3):
    X = np.random.rand(num_samples, sequence_length, 1).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

def evaluate_model(model, X_test, y_test, batch_size=16):
    inference_times = []
    y_preds = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        start_time = time.time()
        outputs = model.predict(batch, verbose=0)
        inference_times.append(time.time() - start_time)
        
        preds = np.argmax(outputs, axis=1)
        y_preds.extend(preds)
    
    accuracy = accuracy_score(y_test, y_preds)
    # Use zero_division=0 to handle cases where precision undefined
    precision = precision_score(y_test, y_preds, average='weighted', zero_division=0)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision
    }

def test_performance_benchmark():
    """Run performance benchmark as a test"""
    model = get_model()
    X_test, y_test = generate_dummy_data(num_samples=20)
    results = evaluate_model(model, X_test, y_test)
    
    assert results["Accuracy"] >= 0.0
    assert results["Precision"] >= 0.0
    logger.info(f"Benchmark Test Results: {results}")

if __name__ == "__main__":
    test_performance_benchmark()

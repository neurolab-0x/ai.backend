"""
Pytest configuration and fixtures for test suite.
"""
import pytest
import os
import sys
import tempfile
import shutil
from datetime import datetime

# Add parent directory to path
# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data for testing"""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({
        'timestamp': [datetime.now().isoformat() for _ in range(100)],
        'alpha': np.random.randn(100),
        'beta': np.random.randn(100),
        'theta': np.random.randn(100),
        'delta': np.random.randn(100),
        'gamma': np.random.randn(100),
        'state': np.random.randint(0, 3, 100)
    })

@pytest.fixture
def sample_eeg_csv(test_data_dir, sample_eeg_data):
    """Create a sample EEG CSV file"""
    csv_path = os.path.join(test_data_dir, "sample_eeg.csv")
    sample_eeg_data.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def sample_eeg_json(test_data_dir, sample_eeg_data):
    """Create a sample EEG JSON file"""
    json_path = os.path.join(test_data_dir, "sample_eeg.json")
    sample_eeg_data.to_json(json_path, orient='records')
    return json_path

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    tf = pytest.importorskip("tensorflow")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5, 1)),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@pytest.fixture
def sample_features():
    """Generate sample feature dictionary"""
    return {
        'alpha': 0.5,
        'beta': 0.3,
        'theta': 0.2,
        'delta': 0.1,
        'gamma': 0.4
    }

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test"""
    # Store original environment
    original_env = os.environ.copy()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_influxdb_config():
    """Mock InfluxDB configuration"""
    return {
        'url': 'http://localhost:8086',
        'token': 'test-token',
        'org': 'test-org',
        'bucket': 'test-bucket'
    }

@pytest.fixture
def mock_mongodb_config():
    """Mock MongoDB configuration"""
    return {
        'url': 'mongodb://localhost:27017',
        'database': 'test_neurolab'
    }

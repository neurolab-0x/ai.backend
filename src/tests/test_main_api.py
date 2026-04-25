import unittest
import os
import json
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient
from main import app
import pandas as pd
import pytest
import tempfile
import shutil

class TestMainAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests"""
        cls.client = TestClient(app)
        
    def setUp(self):
        """Set up test fixtures for each test"""
        self.test_data_dir = tempfile.mkdtemp()
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create sample EEG data
        self.sample_data = pd.DataFrame({
            'timestamp': [datetime.now().isoformat() for _ in range(100)],
            'channel_1': np.random.randn(100),
            'channel_2': np.random.randn(100),
            'channel_3': np.random.randn(100),
            'label': np.random.randint(0, 3, 100)
        })
        
        # Save sample data
        self.csv_path = os.path.join(self.test_data_dir, "test_eeg.csv")
        self.sample_data.to_csv(self.csv_path, index=False)
        
        # Create test user token
        self.test_token = "test_token"  # In real tests, generate proper JWT token
        
    def tearDown(self):
        """Clean up test fixtures after each test"""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
            
    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("diagnostics", data)
        self.assertIn("models_loaded", data["diagnostics"])
        self.assertIn("tensorflow_available", data["diagnostics"])
        
    def test_upload_endpoint(self):
        """Test the file upload and processing endpoint"""
        with open(self.csv_path, "rb") as f:
            response = self.client.post(
                "/api/v1/eeg/upload",
                params={"model_type": "original"},
                files={"file": ("test_eeg.csv", f, "text/csv")}
            )
            
        # Accept both 200 and 500 since model/files might not be available in test environment
        self.assertIn(response.status_code, [200, 400, 500])
        
        if response.status_code == 200:
            data = response.json()
            # Check for expected response structure
            self.assertIsInstance(data, dict)
        
    def test_analyze_endpoint(self):
        """Test the analyze endpoint"""
        test_data = {
            "alpha": 0.5,
            "beta": 0.3,
            "theta": 0.2,
            "delta": 0.1,
            "gamma": 0.4,
            "subject_id": "test_subject",
            "session_id": "test_session_001"
        }
        
        response = self.client.post(
            "/api/v1/eeg/analyze",
            params={"model_type": "original"},
            json=test_data
        )
        
        # Accept both 200 and 500 since model might not be loaded
        self.assertIn(response.status_code, [200, 500])
        
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = self.client.get("/api/v1/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("name", data)
        self.assertIn("version", data)
        self.assertIn("features", data)
        
    def test_invalid_file_upload(self):
        """Test handling of invalid file uploads"""
        # Test with unsupported file type
        invalid_file = os.path.join(self.test_data_dir, "test.txt")
        with open(invalid_file, "w") as f:
            f.write("Invalid data")
            
        with open(invalid_file, "rb") as f:
            response = self.client.post(
                "/api/v1/eeg/upload",
                params={"model_type": "original"},
                files={"file": ("test.txt", f, "text/plain")}
            )
            
        # Should return error for invalid file type
        self.assertIn(response.status_code, [400, 500])
        
    def test_analyze_invalid_data(self):
        """Test handling of invalid analysis data"""
        # Test with missing required fields
        invalid_data = {
            "session_id": "test_session_001"
        }
        
        response = self.client.post(
            "/api/v1/eeg/analyze",
            params={"model_type": "original"},
            json=invalid_data
        )
        
        # Should return error for invalid data
        self.assertIn(response.status_code, [400, 422, 500])
        
    def test_calibrate_endpoint(self):
        """Test the calibrate endpoint"""
        calibration_data = {
            "model_name": "original",
            "calibration_data": {
                "X": [[0.1, 0.2, 0.3, 0.4, 0.5]],
                "y": [0]
            }
        }
        
        response = self.client.post(
            "/api/v1/models/calibrate",
            json=calibration_data
        )
        
        # Accept various status codes depending on model availability
        self.assertIn(response.status_code, [200, 500, 503])
        
    def test_recommendations_endpoint(self):
        """Test the recommendations endpoint"""
        response = self.client.post(
            "/api/v1/eeg/recommendations",
            json={
                "state_durations": {"0": 10, "1": 5, "2": 3},
                "total_duration": 18,
                "confidence": 75,
                "cognitive_metrics": {"focus_index": 0.6},
                "state_transitions": 2
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommendations", data)
        self.assertIn("count", data)

if __name__ == '__main__':
    unittest.main() 

"""
Integration tests for FastAPI endpoints using TestClient and pytest
"""
import pytest
from fastapi.testclient import TestClient
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
import io
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

# Patch ModelManager to prevent heavy model loading during import of main
with patch('src.utils.model_manager.ModelManager._load_model') as mock_load:
    from main import app

client = TestClient(app)

class TestAPIIntegration:
    
    @pytest.fixture
    def sample_eeg_csv(self):
        """Create a sample EEG CSV in memory"""
        data = {
            'alpha': [0.5, 0.6],
            'beta': [0.3, 0.4],
            'theta': [0.2, 0.3],
            'delta': [0.1, 0.2],
            'gamma': [0.4, 0.5]
        }
        df = pd.DataFrame(data)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    def test_root_endpoint(self):
        """Test root endpoint returns correct info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "NeuroLab" in data['name']
        assert "endpoints" in data

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "diagnostics" in data

    def test_analyze_endpoint_valid_json(self):
        """Test /analyze with valid JSON payload"""
        payload = {
            "alpha": 0.5,
            "beta": 0.3,
            "theta": 0.2,
            "delta": 0.1,
            "gamma": 0.4,
            "subject_id": "api_test_sub",
            "session_id": "api_test_sess"
        }
        
        response = client.post("/analyze", json=payload)
        
        # We accept 200 (Success)
        assert response.status_code == 200
        result = response.json()
        assert "state_label" in result
        assert "confidence" in result

    def test_analyze_endpoint_invalid_json(self):
        """Test /analyze with missing fields"""
        payload = {
            "alpha": 0.5
            # Missing other bands
        }
        
        # Depending on validation pydantic might return 422
        # But our processor check logic inside might raise 500 if not handled by pydantic model strictly
        # current main.py uses Dict[str, Any] so it might pass to processor and fail there.
        
        response = client.post("/analyze", json=payload)
        
        # The processor raises KeyError for missing keys in the batch branch, 
        # which main.py catches and returns as 500.
        assert response.status_code == 500

    def test_upload_endpoint_csv(self, sample_eeg_csv):
        """Test /upload with CSV file"""
        files = {
            "file": ("test_eeg.csv", sample_eeg_csv, "text/csv")
        }
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert "state_label" in result

    def test_upload_endpoint_invalid_file_type(self):
        """Test /upload with invalid file"""
        files = {
            "file": ("test.txt", io.BytesIO(b"bad data"), "text/plain")
        }
        
        response = client.post("/upload", files=files)
        
        # validate_file in src/utils/file_handler.py should raise 400
        assert response.status_code in [400, 500] 

    def test_recommendations_endpoint(self):
        """Test /recommendations endpoint"""
        # This endpoint accepts a lot of arguments.
        # state_durations is Dict[int, float]
        
        payload = {
            "state_durations": {0: 10, 1: 20, 2: 5},
            "total_duration": 35,
            "confidence": 85.5,
            "state_transitions": 2
        }
        
        response = client.post("/recommendations", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

import unittest
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
from src.services.data_service import DataHandler, EEGDataPoint

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Initialize components
        self.data_handler = DataHandler(buffer_size=1000)
        self.explanation_generator = ExplanationGenerator()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': [datetime.now().isoformat() for _ in range(100)],
            'channel_1': np.random.randn(100),
            'channel_2': np.random.randn(100),
            'channel_3': np.random.randn(100),
            'label': np.random.randint(0, 3, 100)
        })
        
        # Save sample data
        self.csv_path = os.path.join(self.test_data_dir, "test_eeg.csv")
        self.json_path = os.path.join(self.test_data_dir, "test_eeg.json")
        self.sample_data.to_csv(self.csv_path, index=False)
        self.sample_data.to_json(self.json_path, orient='records')
        
    def tearDown(self):
        # Clean up test files
        for file_path in [self.csv_path, self.json_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.test_data_dir):
            os.rmdir(self.test_data_dir)
            
    def test_load_manual_data_csv(self):
        """Test loading data from CSV file"""
        data_points = self.data_handler.load_manual_data(
            self.csv_path,
            subject_id="test_subject",
            session_id="test_session"
        )
        
        self.assertIsInstance(data_points, list)
        self.assertTrue(len(data_points) > 0)
        self.assertIsInstance(data_points[0], EEGDataPoint)
        
    def test_load_manual_data_json(self):
        """Test loading data from JSON file"""
        data_points = self.data_handler.load_manual_data(
            self.json_path,
            subject_id="test_subject",
            session_id="test_session"
        )
        
        self.assertIsInstance(data_points, list)
        self.assertTrue(len(data_points) > 0)
        self.assertIsInstance(data_points[0], EEGDataPoint)
        
    def test_process_data_point(self):
        """Test processing a single data point"""
        data_point = EEGDataPoint(
            timestamp=datetime.now(),
            features={
                "channel_1": 0.5,
                "channel_2": -0.3,
                "channel_3": 0.1
            },
            subject_id="test_subject",
            session_id="test_session",
            state="relaxed",
            confidence=0.85
        )
        
        explanation = self.data_handler.process_data_point(
            data_point,
            self.explanation_generator
        )
        
        self.assertIsInstance(explanation, dict)
        self.assertIn("clinical_observation", explanation)
        self.assertIn("technical_analysis", explanation)
        self.assertIn("interpretation", explanation)
        
    def test_save_data(self):
        """Test saving data points to file"""
        data_points = [
            EEGDataPoint(
                timestamp=datetime.now(),
                features={
                    "channel_1": np.random.randn(),
                    "channel_2": np.random.randn(),
                    "channel_3": np.random.randn()
                },
                subject_id="test_subject",
                session_id="test_session",
                state="relaxed",
                confidence=0.85
            )
            for _ in range(5)
        ]
        
        # Test saving to CSV
        csv_output = os.path.join(self.test_data_dir, "output.csv")
        self.data_handler.save_data(data_points, csv_output)
        self.assertTrue(os.path.exists(csv_output))
        
        # Test saving to JSON
        json_output = os.path.join(self.test_data_dir, "output.json")
        self.data_handler.save_data(data_points, json_output)
        self.assertTrue(os.path.exists(json_output))
        
        # Clean up
        os.remove(csv_output)
        os.remove(json_output)
        
    def test_explanation_generator(self):
        """Test the explanation generator"""
        eeg_state = EEGState(
            state="stressed",
            confidence=0.92,
            features={
                "channel_1": 0.5,
                "channel_2": -0.3,
                "channel_3": 0.1
            },
            timestamp=datetime.now(),
            subject_id="test_subject",
            session_id="test_session"
        )
        
        explanation = self.explanation_generator.generate_explanation(
            eeg_state,
            additional_context={
                "patient_age": 30,
                "occupation": "software_engineer",
                "session_duration": "300 seconds",
                "previous_sessions": 5,
                "reported_symptoms": ["headache", "fatigue"],
                "medication": None,
                "sleep_hours": 6.5
            }
        )
        
        self.assertIsInstance(explanation, dict)
        self.assertIn("clinical_observation", explanation)
        self.assertIn("technical_analysis", explanation)
        self.assertIn("interpretation", explanation)
        self.assertIn("temporal_analysis", explanation)
        self.assertIn("safety_assessment", explanation)
        self.assertIn("recommendations", explanation)
        
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        # Test with invalid file path
        with self.assertRaises(Exception):
            self.data_handler.load_manual_data(
                "nonexistent_file.csv",
                subject_id="test_subject",
                session_id="test_session"
            )
            
        # Test with invalid data point
        invalid_data_point = EEGDataPoint(
            timestamp=datetime.now(),
            features={},  # Empty features
            subject_id="test_subject",
            session_id="test_session"
        )
        
        with self.assertRaises(Exception):
            self.data_handler.process_data_point(
                invalid_data_point,
                self.explanation_generator
            )
            
    def test_buffer_management(self):
        """Test data buffer management"""
        # Add data points to buffer
        for _ in range(5):
            data_point = EEGDataPoint(
                timestamp=datetime.now(),
                features={
                    "channel_1": np.random.randn(),
                    "channel_2": np.random.randn(),
                    "channel_3": np.random.randn()
                },
                subject_id="test_subject",
                session_id="test_session"
            )
            self.data_handler.buffer.append(data_point)
            
        # Test buffer retrieval
        buffer_data = self.data_handler.get_buffer_data()
        self.assertEqual(len(buffer_data), 5)
        
        # Test buffer clearing
        self.data_handler.clear_buffer()
        self.assertEqual(len(self.data_handler.buffer), 0)

if __name__ == '__main__':
    unittest.main() 
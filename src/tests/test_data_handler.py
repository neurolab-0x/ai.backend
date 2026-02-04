import unittest
import asyncio
from datetime import datetime
import pandas as pd
import json
import os
from src.services.data_service import DataHandler, EEGDataPoint

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.data_handler = DataHandler(buffer_size=100)
        self.explanation_generator = ExplanationGenerator()
        
        # Create test data directory
        os.makedirs('test_data', exist_ok=True)
        
    def test_manual_data_handling(self):
        """Test handling of manual data input."""
        # Create sample CSV data
        csv_data = pd.DataFrame({
            'timestamp': [datetime.now().isoformat()],
            'beta': [25.5],
            'alpha': [7.2],
            'theta': [3.8],
            'beta_amplitude': [35.0],
            'alpha_amplitude': [45.0],
            'theta_amplitude': [30.0]
        })
        csv_path = 'test_data/test_eeg.csv'
        csv_data.to_csv(csv_path, index=False)
        
        # Load and process manual data
        data_points = self.data_handler.load_manual_data(
            csv_path,
            subject_id="SUBJ001",
            session_id="SESS001"
        )
        
        # Verify data loading
        self.assertEqual(len(data_points), 1)
        self.assertEqual(data_points[0].subject_id, "SUBJ001")
        self.assertEqual(data_points[0].features['beta'], 25.5)
        
        # Clean up
        os.remove(csv_path)
        
    def test_streaming_data_handling(self):
        """Test handling of streaming data."""
        # Create a sample data generator
        def sample_data_generator():
            for _ in range(5):
                yield {
                    'beta': 25.5,
                    'alpha': 7.2,
                    'theta': 3.8,
                    'beta_amplitude': 35.0,
                    'alpha_amplitude': 45.0,
                    'theta_amplitude': 30.0
                }
                time.sleep(0.1)  # Simulate real-time data
                
        # Start streaming
        asyncio.run(self.data_handler.start_streaming(
            sample_data_generator(),
            subject_id="SUBJ001",
            session_id="SESS002"
        ))
        
        # Wait for some data to be collected
        time.sleep(0.6)
        
        # Stop streaming
        self.data_handler.stop_streaming()
        
        # Get buffered data
        data_points = self.data_handler.get_buffer_data()
        
        # Verify streaming data
        self.assertGreater(len(data_points), 0)
        self.assertEqual(data_points[0].subject_id, "SUBJ001")
        self.assertEqual(data_points[0].features['beta'], 25.5)
        
    async def test_data_processing(self):
        """Test processing of data points with explanation generation."""
        # Create a sample data point
        data_point = EEGDataPoint(
            timestamp=datetime.now(),
            features={
                'beta': 25.5,
                'alpha': 7.2,
                'theta': 3.8,
                'beta_amplitude': 35.0,
                'alpha_amplitude': 45.0,
                'theta_amplitude': 30.0
            },
            subject_id="SUBJ001",
            session_id="SESS003",
            state="stressed",
            confidence=0.92
        )
        
        # Process data point
        explanation = await self.data_handler.process_data_point(
            data_point,
            self.explanation_generator
        )
        
        # Verify explanation generation
        self.assertIn('clinical_observation', explanation)
        self.assertIn('technical_analysis', explanation)
        self.assertIn('interpretation', explanation)
        self.assertIn('recommendations', explanation)
        
    def test_data_saving(self):
        """Test saving data to different formats."""
        # Create sample data points
        data_points = [
            EEGDataPoint(
                timestamp=datetime.now(),
                features={
                    'beta': 25.5,
                    'alpha': 7.2,
                    'theta': 3.8
                },
                subject_id="SUBJ001",
                session_id="SESS001"
            )
        ]
        
        # Save as CSV
        csv_path = 'test_data/saved_data.csv'
        self.data_handler.save_data(data_points, csv_path)
        self.assertTrue(os.path.exists(csv_path))
        
        # Save as JSON
        json_path = 'test_data/saved_data.json'
        self.data_handler.save_data(data_points, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # Clean up
        os.remove(csv_path)
        os.remove(json_path)
        
    def tearDown(self):
        # Clean up test data directory
        if os.path.exists('test_data'):
            os.rmdir('test_data')

if __name__ == '__main__':
    unittest.main()
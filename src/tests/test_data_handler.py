import asyncio
import os
import time
import unittest
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.services.data_service import DataHandler, EEGDataPoint


@dataclass
class _Context:
    state_label: str = "relaxed"


class ExplanationGenerator:
    """Minimal stub used by legacy unit tests."""

    def _build_context(self, **kwargs):
        return _Context(state_label=kwargs.get("state_label", "relaxed"))

    async def generate_medical_explanation(self, context):
        return {
            "clinical_observation": "ok",
            "technical_analysis": "ok",
            "interpretation": context.state_label,
            "recommendations": [],
        }


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.data_handler = DataHandler(buffer_size=100)
        self.explanation_generator = ExplanationGenerator()

        # Create test data directory
        os.makedirs("test_data", exist_ok=True)

    def test_manual_data_handling(self):
        """Test handling of manual data input."""
        csv_data = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "beta": [25.5],
                "alpha": [7.2],
                "theta": [3.8],
                "beta_amplitude": [35.0],
                "alpha_amplitude": [45.0],
                "theta_amplitude": [30.0],
            }
        )
        csv_path = "test_data/test_eeg.csv"
        csv_data.to_csv(csv_path, index=False)

        data_points = self.data_handler.load_manual_data(
            csv_path,
            subject_id="SUBJ001",
            session_id="SESS001",
        )

        self.assertEqual(len(data_points), 1)
        self.assertEqual(data_points[0].subject_id, "SUBJ001")
        self.assertEqual(data_points[0].features["beta"], 25.5)

        os.remove(csv_path)

    def test_streaming_data_handling(self):
        """Test handling of streaming data."""

        def sample_data_generator():
            for _ in range(5):
                yield {
                    "beta": 25.5,
                    "alpha": 7.2,
                    "theta": 3.8,
                    "beta_amplitude": 35.0,
                    "alpha_amplitude": 45.0,
                    "theta_amplitude": 30.0,
                }
                time.sleep(0.1)

        asyncio.run(
            self.data_handler.start_streaming(
                sample_data_generator(),
                subject_id="SUBJ001",
                session_id="SESS002",
            )
        )

        time.sleep(0.6)
        self.data_handler.stop_streaming()

        data_points = self.data_handler.get_buffer_data()

        self.assertGreater(len(data_points), 0)
        self.assertEqual(data_points[0].subject_id, "SUBJ001")
        self.assertEqual(data_points[0].features["beta"], 25.5)

    def test_data_processing(self):
        """Test processing of data points with explanation generation."""
        data_point = EEGDataPoint(
            timestamp=datetime.now(),
            features={
                "beta": 25.5,
                "alpha": 7.2,
                "theta": 3.8,
                "beta_amplitude": 35.0,
                "alpha_amplitude": 45.0,
                "theta_amplitude": 30.0,
            },
            subject_id="SUBJ001",
            session_id="SESS003",
            state="stressed",
            confidence=0.92,
        )

        explanation = asyncio.run(
            self.data_handler.process_data_point(
                data_point,
                self.explanation_generator,
            )
        )

        self.assertIn("clinical_observation", explanation)
        self.assertIn("technical_analysis", explanation)
        self.assertIn("interpretation", explanation)
        self.assertIn("recommendations", explanation)

    def test_data_saving(self):
        """Test saving data to different formats."""
        data_points = [
            EEGDataPoint(
                timestamp=datetime.now(),
                features={"beta": 25.5, "alpha": 7.2, "theta": 3.8},
                subject_id="SUBJ001",
                session_id="SESS001",
            )
        ]

        csv_path = "test_data/saved_data.csv"
        self.data_handler.save_data(data_points, csv_path)
        self.assertTrue(os.path.exists(csv_path))

        json_path = "test_data/saved_data.json"
        self.data_handler.save_data(data_points, json_path)
        self.assertTrue(os.path.exists(json_path))

        os.remove(csv_path)
        os.remove(json_path)

    def tearDown(self):
        if os.path.exists("test_data"):
            os.rmdir("test_data")


if __name__ == "__main__":
    unittest.main()

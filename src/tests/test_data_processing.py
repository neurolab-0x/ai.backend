import asyncio
import os
import unittest
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.services.data_service import DataHandler, EEGDataPoint


@dataclass
class EEGState:
    state: str
    confidence: float
    features: dict
    timestamp: datetime
    subject_id: str
    session_id: str


class ExplanationGenerator:
    def generate_explanation(self, eeg_state, additional_context=None):
        _ = additional_context
        return {
            "clinical_observation": eeg_state.state,
            "technical_analysis": "ok",
            "interpretation": "ok",
            "temporal_analysis": {},
            "safety_assessment": "ok",
            "recommendations": [],
        }

    def _build_context(self, **kwargs):
        return type("Context", (), {"state_label": kwargs.get("state_label", "relaxed")})()

    async def generate_medical_explanation(self, context):
        return {
            "clinical_observation": context.state_label,
            "technical_analysis": "ok",
            "interpretation": "ok",
            "recommendations": [],
        }


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)

        self.data_handler = DataHandler(buffer_size=1000)
        self.explanation_generator = ExplanationGenerator()

        self.sample_data = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat() for _ in range(100)],
                "channel_1": np.random.randn(100),
                "channel_2": np.random.randn(100),
                "channel_3": np.random.randn(100),
                "label": np.random.randint(0, 3, 100),
            }
        )

        self.csv_path = os.path.join(self.test_data_dir, "test_eeg.csv")
        self.json_path = os.path.join(self.test_data_dir, "test_eeg.json")
        self.sample_data.to_csv(self.csv_path, index=False)
        self.sample_data.to_json(self.json_path, orient="records")

    def tearDown(self):
        for file_path in [self.csv_path, self.json_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.test_data_dir):
            os.rmdir(self.test_data_dir)

    def test_load_manual_data_csv(self):
        data_points = self.data_handler.load_manual_data(
            self.csv_path,
            subject_id="test_subject",
            session_id="test_session",
        )
        self.assertIsInstance(data_points, list)
        self.assertTrue(len(data_points) > 0)
        self.assertIsInstance(data_points[0], EEGDataPoint)

    def test_load_manual_data_json(self):
        data_points = self.data_handler.load_manual_data(
            self.json_path,
            subject_id="test_subject",
            session_id="test_session",
        )

        self.assertIsInstance(data_points, list)
        self.assertTrue(len(data_points) > 0)
        self.assertIsInstance(data_points[0], EEGDataPoint)

    def test_process_data_point(self):
        data_point = EEGDataPoint(
            timestamp=datetime.now(),
            features={"channel_1": 0.5, "channel_2": -0.3, "channel_3": 0.1},
            subject_id="test_subject",
            session_id="test_session",
            state="relaxed",
            confidence=0.85,
        )

        explanation = asyncio.run(
            self.data_handler.process_data_point(data_point, self.explanation_generator)
        )

        self.assertIsInstance(explanation, dict)
        self.assertIn("clinical_observation", explanation)
        self.assertIn("technical_analysis", explanation)
        self.assertIn("interpretation", explanation)

    def test_save_data(self):
        data_points = [
            EEGDataPoint(
                timestamp=datetime.now(),
                features={
                    "channel_1": np.random.randn(),
                    "channel_2": np.random.randn(),
                    "channel_3": np.random.randn(),
                },
                subject_id="test_subject",
                session_id="test_session",
                state="relaxed",
                confidence=0.85,
            )
            for _ in range(5)
        ]

        csv_output = os.path.join(self.test_data_dir, "output.csv")
        self.data_handler.save_data(data_points, csv_output)
        self.assertTrue(os.path.exists(csv_output))

        json_output = os.path.join(self.test_data_dir, "output.json")
        self.data_handler.save_data(data_points, json_output)
        self.assertTrue(os.path.exists(json_output))

        os.remove(csv_output)
        os.remove(json_output)

    def test_explanation_generator(self):
        eeg_state = EEGState(
            state="stressed",
            confidence=0.92,
            features={"channel_1": 0.5, "channel_2": -0.3, "channel_3": 0.1},
            timestamp=datetime.now(),
            subject_id="test_subject",
            session_id="test_session",
        )

        explanation = self.explanation_generator.generate_explanation(
            eeg_state,
            additional_context={
                "patient_age": 30,
                "occupation": "software_engineer",
            },
        )

        self.assertIsInstance(explanation, dict)
        self.assertIn("clinical_observation", explanation)
        self.assertIn("technical_analysis", explanation)
        self.assertIn("interpretation", explanation)
        self.assertIn("temporal_analysis", explanation)
        self.assertIn("safety_assessment", explanation)
        self.assertIn("recommendations", explanation)

    def test_invalid_data_handling(self):
        with self.assertRaises(Exception):
            self.data_handler.load_manual_data(
                "nonexistent_file.csv",
                subject_id="test_subject",
                session_id="test_session",
            )

    def test_buffer_management(self):
        for _ in range(5):
            self.data_handler.data_buffer.put(
                EEGDataPoint(
                    timestamp=datetime.now(),
                    features={
                        "channel_1": np.random.randn(),
                        "channel_2": np.random.randn(),
                        "channel_3": np.random.randn(),
                    },
                    subject_id="test_subject",
                    session_id="test_session",
                )
            )

        buffer_data = self.data_handler.get_buffer_data()
        self.assertEqual(len(buffer_data), 5)
        self.data_handler.clear_buffer()
        self.assertEqual(self.data_handler.data_buffer.qsize(), 0)


if __name__ == "__main__":
    unittest.main()

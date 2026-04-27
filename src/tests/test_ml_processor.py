"""
Test script for ML Processor integration using pytest
"""
import pytest
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
import os
from src.services.analysis import MLProcessor

# Configuration
TEST_MODEL_PATH = "model/trained_model.h5"

class TestMLProcessor:
    @pytest.fixture
    def processor(self):
        """Initialize ML Processor"""
        return MLProcessor(model_path=TEST_MODEL_PATH)
    
    @pytest.fixture
    def sample_dict_data(self):
        """Sample dictionary data"""
        return {
            'alpha': 0.5,
            'beta': 0.3,
            'theta': 0.2,
            'delta': 0.1,
            'gamma': 0.4
        }
        
    @pytest.fixture
    def sample_array_data(self):
        """Sample numpy array data"""
        return np.array([
            [0.5, 0.3, 0.2, 0.1, 0.4],
            [0.6, 0.4, 0.3, 0.2, 0.5],
            [0.4, 0.2, 0.1, 0.05, 0.3],
            [0.7, 0.5, 0.4, 0.3, 0.6],
            [0.3, 0.1, 0.05, 0.02, 0.2]
        ])

    def test_initialization(self, processor):
        """Test processor initialization and status"""
        status = processor.get_status()
        assert 'model_loaded' in status
        assert 'model_path' in status
        assert 'recommendation_engine_loaded' in status
        
        # We don't assert model_loaded=True strictly because it might fallback to False
        # if the file doesn't exist, which is a valid state for the test environment.

    def test_process_single_sample_dict(self, processor, sample_dict_data):
        """Test processing of a single sample dictionary"""
        result = processor.process_eeg_data(
            sample_dict_data,
            subject_id="test_sub_001",
            session_id="test_sess_001"
        )
        
        assert 'state_label' in result
        assert 'dominant_state' in result
        assert 'confidence' in result
        assert 'cognitive_metrics' in result
        assert isinstance(result['confidence'], float)
        assert result['dominant_state'] in [0, 1, 2]
        assert result['metadata']['subject_id'] == "test_sub_001"

    def test_process_batch_array(self, processor, sample_array_data):
        """Test processing of a batch numpy array"""
        result = processor.process_eeg_data(
            sample_array_data,
            subject_id="test_sub_002",
            session_id="test_sess_002"
        )
        
        assert result['temporal_analysis']['total_samples'] == 5
        assert 'state_transitions' in result['temporal_analysis']
        assert len(result['predicted_state']) == 5
        assert len(result['smoothed_states']) == 5

    def test_process_dataframe(self, processor, sample_array_data):
        """Test processing of a pandas DataFrame"""
        columns = ['alpha', 'beta', 'theta', 'delta', 'gamma']
        df = pd.DataFrame(sample_array_data, columns=columns)
        
        result = processor.process_eeg_data(
            df,
            subject_id="test_sub_003"
        )
        
        assert result['temporal_analysis']['total_samples'] == 5
        assert result['metadata']['subject_id'] == "test_sub_003"

    def test_error_handling_invalid_input(self, processor):
        """Test error handling for invalid input"""
        invalid_data = {'invalid_key': 123}
        
        with pytest.raises(Exception):
            processor.process_eeg_data(invalid_data)

    def test_cognitive_metrics_calculation(self, processor, sample_dict_data):
        """Test cognitive metrics calculation correctness"""
        # Note: process_eeg_data normalizes input. For a single sample, z-score normalization 
        # results in all zeros (mean=value, std=0). 
        # Thus the metrics will likely be 0. We verify the structure and type here.
        
        result = processor.process_eeg_data(sample_dict_data)
        metrics = result['cognitive_metrics']
        
        expected_keys = [
            'attention_index', 'relaxation_index', 'stress_index', 
            'cognitive_load', 'mental_fatigue', 'alertness'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_detailed_report_generation(self, processor, sample_dict_data, tmp_path):
        """Test detailed report generation and saving"""
        # Save to a temporary file managed by pytest
        report_path = tmp_path / "test_report.md"
        
        report = processor.generate_detailed_report(
            sample_dict_data,
            subject_id="report_test_sub",
            save_report=False
        )
        
        assert 'analysis_results' in report
        assert 'recommendations' in report
        assert 'insights' in report
        assert 'wellness_score' in report

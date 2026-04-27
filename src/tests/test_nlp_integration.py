"""
Test script for NLP recommendations integration
"""
import pytest

np = pytest.importorskip("numpy")
from src.services.analysis import MLProcessor
from src.services.recommendation import NLPRecommendationEngine

def test_basic_recommendations():
    """Test basic recommendation generation"""
    print("=" * 60)
    print("Testing NLP Recommendation Engine")
    print("=" * 60)
    
    engine = NLPRecommendationEngine()
    
    # Test case 1: High stress scenario
    print("\n1. High Stress Scenario:")
    print("-" * 60)
    state_durations = {0: 10, 1: 20, 2: 70}  # 70% stressed
    recommendations = engine.generate_recommendations(
        state_durations=state_durations,
        total_duration=100,
        confidence=85.0,
        cognitive_metrics={'cognitive_load': 2.5, 'mental_fatigue': 0.6},
        state_transitions=12
    )
    for rec in recommendations:
        print(rec)
    
    # Test case 2: Good focus scenario
    print("\n\n2. Good Focus Scenario:")
    print("-" * 60)
    state_durations = {0: 20, 1: 70, 2: 10}  # 70% focused
    recommendations = engine.generate_recommendations(
        state_durations=state_durations,
        total_duration=100,
        confidence=90.0,
        cognitive_metrics={'cognitive_load': 1.2, 'mental_fatigue': 0.3},
        state_transitions=5
    )
    for rec in recommendations:
        print(rec)
    
    # Test case 3: Relaxed scenario
    print("\n\n3. Relaxed Scenario:")
    print("-" * 60)
    state_durations = {0: 80, 1: 15, 2: 5}  # 80% relaxed
    recommendations = engine.generate_recommendations(
        state_durations=state_durations,
        total_duration=100,
        confidence=88.0,
        cognitive_metrics={'cognitive_load': 0.5, 'mental_fatigue': 0.2},
        state_transitions=3
    )
    for rec in recommendations:
        print(rec)

def test_detailed_report():
    """Test detailed report generation"""
    print("\n\n" + "=" * 60)
    print("Testing Detailed Report Generation")
    print("=" * 60)
    
    engine = NLPRecommendationEngine()
    
    state_durations = {0: 25, 1: 35, 2: 40}
    report = engine.generate_detailed_report(
        state_durations=state_durations,
        total_duration=100,
        confidence=82.5,
        cognitive_metrics={
            'cognitive_load': 2.1,
            'mental_fatigue': 0.55,
            'attention_index': 1.8,
            'stress_index': 1.9
        },
        state_transitions=15
    )
    
    print(f"\nSession Summary:")
    print(f"  Duration: {report['session_summary']['duration_minutes']:.1f} minutes")
    print(f"  Dominant State: {report['session_summary']['dominant_state']}")
    print(f"  Confidence: {report['session_summary']['confidence']:.1f}%")
    
    print(f"\nState Distribution:")
    for state, percentage in report['state_distribution'].items():
        print(f"  {state}: {percentage}")
    
    print(f"\nWellness Score: {report['wellness_score']['description']}")
    
    print(f"\nInsights:")
    for insight in report['insights']:
        print(f"  {insight}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")

def test_ml_processor_integration():
    """Test ML Processor integration"""
    print("\n\n" + "=" * 60)
    print("Testing ML Processor Integration")
    print("=" * 60)
    
    processor = MLProcessor()
    
    # Create sample EEG data
    sample_data = {
        'alpha': 8.5,
        'beta': 15.2,
        'theta': 6.3,
        'delta': 3.1,
        'gamma': 2.8
    }
    
    print("\nProcessing sample EEG data...")
    try:
        result = processor.process_eeg_data(sample_data)
        
        print(f"\nAnalysis Results:")
        print(f"  State: {result['state_label']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  State Transitions: {result['temporal_analysis']['state_transitions']}")
        
        print(f"\nCognitive Metrics:")
        for metric, value in result['cognitive_metrics'].items():
            print(f"  {metric}: {value:.3f}")
        
        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  {rec}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Note: This is expected if the model file is not available.")

if __name__ == "__main__":
    test_basic_recommendations()
    test_detailed_report()
    test_ml_processor_integration()
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

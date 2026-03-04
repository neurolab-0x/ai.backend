from groq import Groq
from src.config.settings import LLM_CONFIG
from src.services.database import db_service
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class RecommendationContext:
    """Context information for generating recommendations and explanations"""
    state_label: str
    confidence: float
    stress_ratio: float
    relaxation_ratio: float
    focus_ratio: float
    cognitive_metrics: Dict[str, float]
    state_transitions: int
    session_duration: float
    timestamp: datetime = datetime.now()
    subject_id: str = "unknown"
    session_id: str = "unknown"
    features: Dict[str, float] = None

class NLPRecommendationEngine:
    """
    RAG-based recommendation engine using Groq LLM
    Generates personalized insights grounded in historical EEG data.
    """
    
    def __init__(self):
        self.client = None
        if LLM_CONFIG.get('api_key'):
            try:
                self.client = Groq(api_key=LLM_CONFIG['api_key'])
                logger.info(f"Groq client initialized with model: {LLM_CONFIG['model']}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("GROQ_API_KEY not found in environment. Falling back to rule-based logic.")
            
        logger.info("NLP Recommendation Engine initialized")

    def _build_context(
        self,
        state_durations: Dict[int, float],
        total_duration: float,
        confidence: float,
        cognitive_metrics: Dict[str, float],
        state_transitions: int,
        timestamp: datetime = None,
        subject_id: str = "unknown",
        session_id: str = "unknown",
        features: Dict[str, float] = None
    ) -> RecommendationContext:
        """Build a recommendation context object from raw metrics"""
        # Map state indices: 0: relaxed, 1: focused, 2: stressed
        relaxation_ratio = state_durations.get(0, 0) / total_duration if total_duration > 0 else 0
        focus_ratio = state_durations.get(1, 0) / total_duration if total_duration > 0 else 0
        stress_ratio = state_durations.get(2, 0) / total_duration if total_duration > 0 else 0
        
        # Determine current state label based on dominant duration
        states = {0: "relaxed", 1: "focused", 2: "stressed"}
        dominant_state_idx = max(state_durations, key=state_durations.get) if state_durations else 0
        state_label = states.get(dominant_state_idx, "unknown")
        
        return RecommendationContext(
            state_label=state_label,
            confidence=confidence,
            stress_ratio=stress_ratio,
            relaxation_ratio=relaxation_ratio,
            focus_ratio=focus_ratio,
            cognitive_metrics=cognitive_metrics,
            state_transitions=state_transitions,
            session_duration=total_duration,
            timestamp=timestamp or datetime.now(),
            subject_id=subject_id,
            session_id=session_id,
            features=features
        )
    
    async def generate_recommendations(
        self,
        state_durations: Dict[int, float],
        total_duration: float,
        confidence: float,
        cognitive_metrics: Dict[str, float] = None,
        state_transitions: int = 0,
        subject_id: str = "unknown",
        session_id: str = "unknown",
        max_recommendations: int = 5
    ) -> List[str]:
        """Generate personalized recommendations using RAG with Groq LLM"""
        try:
            context = self._build_context(
                state_durations, total_duration, confidence,
                cognitive_metrics or {}, state_transitions,
                subject_id=subject_id, session_id=session_id
            )
            
            history = await db_service.get_user_history(subject_id, limit=3)
            
            if self.client:
                return await self._generate_llm_recommendations(context, history, max_recommendations)
            
            return self._get_fallback_recommendations()
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._get_fallback_recommendations()
 
    async def _generate_llm_recommendations(
        self, 
        context: RecommendationContext, 
        history: List[Dict[str, Any]],
        max_recommendations: int
    ) -> List[str]:
        """Build prompt and call Groq LLM"""
        history_str = "\n".join([
            f"- {h['time']}: ID {h['run_id']}, Accuracy: {h['accuracy']:.2f}, Loss: {h['loss']:.2f}"
            for h in history
        ]) if history else "No previous history available."
 
        prompt = f"""
You are the NeuroLab AI Neural Health Expert. Your task is to provide personalized, actionable recommendations based on a user's EEG session data and their historical trends.
 
### Current EEG Session Context:
- State: {context.state_label} (Confidence: {context.confidence:.1f}%)
- Session Distribution:
    - Focused: {context.focus_ratio*100:.1f}%
    - Relaxed: {context.relaxation_ratio*100:.1f}%
    - Stressed: {context.stress_ratio*100:.1f}%
- Cognitive Metrics: {json.dumps(context.cognitive_metrics)}
- Stability: {context.state_transitions} transitions over {context.session_duration/60:.1f} minutes.
 
### User Historical Trends (Last 3 Sessions):
{history_str}
 
### Instructions:
1. Analyze if current metrics indicate an improvement or regression from history.
2. Provide exactly {max_recommendations} concise, bulleted recommendations.
3. Focus on mental performance, stress reduction, or maintaining focus based on the dominant state.
4. Keep the tone professional, encouraging, and medical/scientific.
 
Recommendations:
"""
        try:
            completion = self.client.chat.completions.create(
                model=LLM_CONFIG['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_CONFIG['temperature'],
                max_tokens=LLM_CONFIG['max_tokens'],
            )
            
            response = completion.choices[0].message.content
            lines = [line.strip() for line in response.split('\n') if line.strip() and (line.strip().startswith('•') or line.strip().startswith('-') or line.strip().startswith('*') or (len(line) > 2 and line[1] == '.'))]
            
            if not lines:
                return [l.strip() for l in response.split('\n') if l.strip()][:max_recommendations]
                
            return lines[:max_recommendations]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_fallback_recommendations()

    def _get_fallback_recommendations(self) -> List[str]:
        """Provide fallback recommendations if generation fails"""
        return [
            "• Take regular breaks every 50-90 minutes",
            "• Practice deep breathing exercises",
            "• Stay hydrated throughout the day",
            "• maintain good posture while working",
            "• Get adequate sleep (7-9 hours per night)"
        ]

    async def generate_medical_explanation(
        self,
        context: RecommendationContext,
        occupation: str = "default"
    ) -> Dict[str, Any]:
        """Generate a professional medical explanation using Groq LLM"""
        if not self.client:
            return {"error": "LLM client not initialized", "explanation": "Rule-based logic removed per user request."}

        prompt = f"""
You are a Clinical Neuroscientist. Provide a detailed medical explanation for this EEG session.
Context:
- State: {context.state_label} (Confidence: {context.confidence:.1f}%)
- Metrics: {json.dumps(context.cognitive_metrics)}
- Features: {json.dumps(context.features or {{}})}
 
Format as JSON:
{{
  "clinical_observation": "...",
  "technical_analysis": "...",
  "interpretation": "...",
  "safety_assessment": {{ "alert_level": "low|medium|high", "immediate_actions": [] }}
}}
"""
        try:
            completion = self.client.chat.completions.create(
                model=LLM_CONFIG['model'],
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            explanation = json.loads(completion.choices[0].message.content)
            explanation["metadata"] = {
                "timestamp": context.timestamp.isoformat(),
                "subject_id": context.subject_id,
                "session_id": context.session_id,
                "confidence": float(context.confidence)
            }
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate medical explanation: {e}")
            return {{"error": str(e)}}

    async def generate_detailed_report(
        self,
        state_durations: Dict[int, float],
        total_duration: float,
        confidence: float,
        cognitive_metrics: Dict[str, float] = None,
        state_transitions: int = 0,
        subject_id: str = "unknown",
        session_id: str = "unknown"
    ) -> Dict[str, Any]:
        """Generate a complete session report with LLM insights"""
        try:
            context = self._build_context(
                state_durations, total_duration, confidence,
                cognitive_metrics or {}, state_transitions,
                subject_id=subject_id, session_id=session_id
            )
            
            recs = await self.generate_recommendations(
                state_durations, total_duration, confidence,
                cognitive_metrics, state_transitions,
                subject_id=subject_id, session_id=session_id
            )
            
            explanation = await self.generate_medical_explanation(context)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "subject_id": subject_id,
                "dominant_state": context.state_label,
                "wellness_rating": explanation.get("safety_assessment", {}).get("alert_level", "unknown"),
                "clinical_observation": explanation.get("clinical_observation"),
                "technical_analysis": explanation.get("technical_analysis"),
                "recommendations": recs,
                "metrics": {
                    "confidence": confidence,
                    "transitions": state_transitions,
                    "cognitive": cognitive_metrics
                }
            }
        except Exception as e:
            logger.error(f"Error in detailed report: {e}")
            return {"error": str(e)}

    def save_report(self, report: Dict[str, Any], filepath: str = None) -> str:
        """Save report to JSON file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"reports/eeg_report_{timestamp}.json"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            return filepath
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return None

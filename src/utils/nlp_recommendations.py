"""
NLP-based Recommendation System with RAG (Retrieval-Augmented Generation)
Generates personalized recommendations based on EEG analysis using semantic search
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

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


class RecommendationKnowledgeBase:
    """Knowledge base for EEG-based recommendations"""
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Build comprehensive knowledge base for recommendations and medical analysis"""
        return {
            "medical_terminology": {
                "relaxed": {
                    "terms": ["alpha waves", "relaxed state", "mental calmness"],
                    "normal_range": {"alpha": (8, 13), "theta": (4, 8)},
                    "clinical_significance": "Indicates a relaxed, alert state of mind",
                    "occupation_specific": {
                        "software_engineer": "Optimal state for complex problem-solving",
                        "healthcare_worker": "Good state for patient care and decision-making",
                        "student": "Ideal state for learning and information retention"
                    }
                },
                "focused": {
                    "terms": ["beta waves", "concentration", "mental focus"],
                    "normal_range": {"beta": (13, 30)},
                    "clinical_significance": "Suggests active mental engagement and concentration",
                    "occupation_specific": {
                        "software_engineer": "Good for coding and debugging tasks",
                        "healthcare_worker": "Appropriate for medical procedures",
                        "student": "Suitable for studying and exam preparation"
                    }
                },
                "drowsy": {
                    "terms": ["theta waves", "drowsiness", "light sleep"],
                    "normal_range": {"theta": (4, 8)},
                    "clinical_significance": "May indicate transition to sleep or reduced alertness",
                    "occupation_specific": {
                        "software_engineer": "Risk for code errors and reduced productivity",
                        "healthcare_worker": "Safety concern for patient care",
                        "student": "May affect learning and retention"
                    }
                },
                "stressed": {
                    "terms": ["high beta", "stress response", "mental tension"],
                    "normal_range": {"beta": (20, 30)},
                    "clinical_significance": "Suggests heightened mental activity and potential stress",
                    "occupation_specific": {
                        "software_engineer": "May impact code quality and team collaboration",
                        "healthcare_worker": "Could affect patient care quality",
                        "student": "May hinder learning and performance"
                    }
                }
            },
            "safety_thresholds": {
                "alpha_amplitude": {"default": 50, "software_engineer": 45, "healthcare_worker": 40, "student": 55},
                "beta_amplitude": {"default": 30, "software_engineer": 35, "healthcare_worker": 25, "student": 32},
                "theta_amplitude": {"default": 40, "software_engineer": 35, "healthcare_worker": 30, "student": 45}
            },
            "stress_management": [
                {
                    "condition": "high_stress",
                    "severity": "severe",
                    "techniques": [
                        "Practice 4-7-8 breathing: Inhale for 4 seconds, hold for 7, exhale for 8",
                        "Try progressive muscle relaxation starting from your toes to your head",
                        "Listen to binaural beats at 10 Hz (alpha frequency) for 15-20 minutes",
                        "Consider a 5-minute guided meditation focusing on body scan",
                        "Take a short walk in nature or practice grounding techniques"
                    ],
                    "urgency": "high",
                    "duration": "15-30 minutes"
                },
                {
                    "condition": "moderate_stress",
                    "severity": "moderate",
                    "techniques": [
                        "Practice box breathing: 4 counts in, 4 hold, 4 out, 4 hold",
                        "Try a 10-minute mindfulness meditation",
                        "Engage in light physical activity like stretching or yoga",
                        "Listen to calming music or nature sounds",
                        "Practice gratitude journaling for 5 minutes"
                    ],
                    "urgency": "medium",
                    "duration": "10-15 minutes"
                },
                {
                    "condition": "mild_stress",
                    "severity": "mild",
                    "techniques": [
                        "Take 5 deep diaphragmatic breaths",
                        "Practice brief mindfulness: focus on your senses for 2 minutes",
                        "Step away from your current task for a short break",
                        "Drink water and do gentle neck stretches",
                        "Practice positive self-talk or affirmations"
                    ],
                    "urgency": "low",
                    "duration": "5-10 minutes"
                }
            ],
            "focus_enhancement": [
                {
                    "condition": "low_focus",
                    "techniques": [
                        "Try the Pomodoro Technique: 25 minutes focused work, 5 minutes break",
                        "Eliminate distractions: close unnecessary tabs and silence notifications",
                        "Practice single-tasking: focus on one task at a time",
                        "Use background music designed for concentration (40 Hz gamma waves)",
                        "Take a brief walk to increase blood flow and alertness"
                    ],
                    "duration": "25-30 minutes"
                },
                {
                    "condition": "optimal_focus",
                    "techniques": [
                        "Maintain your current workflow - you're in an optimal state",
                        "Consider tackling your most challenging tasks now",
                        "Stay hydrated and maintain good posture",
                        "Set a timer to remind yourself to take breaks every 50 minutes",
                        "Document your current environment/conditions for future reference"
                    ],
                    "duration": "ongoing"
                }
            ],
            "relaxation_optimization": [
                {
                    "condition": "optimal_relaxation",
                    "techniques": [
                        "Continue your current relaxation practice",
                        "This is an ideal time for creative thinking or reflection",
                        "Consider practicing visualization or guided imagery",
                        "Engage in activities you find personally fulfilling",
                        "Maintain this state with gentle music or nature sounds"
                    ],
                    "duration": "ongoing"
                },
                {
                    "condition": "excessive_relaxation",
                    "techniques": [
                        "Engage in light physical activity to increase alertness",
                        "Try cold water on your face or wrists to boost alertness",
                        "Practice energizing breathing: quick inhales and exhales",
                        "Stand up and do some light stretching or jumping jacks",
                        "Expose yourself to bright light or step outside"
                    ],
                    "duration": "5-10 minutes"
                }
            ],
            "cognitive_load": [
                {
                    "condition": "high_cognitive_load",
                    "techniques": [
                        "Break complex tasks into smaller, manageable chunks",
                        "Use external memory aids: write things down or use task lists",
                        "Take a 10-minute break to allow mental recovery",
                        "Practice the 20-20-20 rule: every 20 min, look 20 feet away for 20 sec",
                        "Consider delegating or postponing non-critical tasks"
                    ],
                    "duration": "10-15 minutes"
                }
            ],
            "mental_fatigue": [
                {
                    "condition": "high_fatigue",
                    "techniques": [
                        "Take a power nap (10-20 minutes) if possible",
                        "Practice yoga nidra or body scan meditation",
                        "Ensure adequate hydration and consider a healthy snack",
                        "Take a longer break (30+ minutes) from cognitive tasks",
                        "Consider ending your work session if possible - rest is crucial"
                    ],
                    "duration": "20-30 minutes"
                },
                {
                    "condition": "moderate_fatigue",
                    "techniques": [
                        "Take a 5-10 minute break from screen time",
                        "Do light physical movement or stretching",
                        "Practice alternate nostril breathing for energy balance",
                        "Switch to less demanding tasks temporarily",
                        "Ensure you're in a well-ventilated space with good lighting"
                    ],
                    "duration": "10-15 minutes"
                }
            ],
            "general_wellness": [
                {
                    "condition": "baseline",
                    "techniques": [
                        "Maintain regular sleep schedule (7-9 hours per night)",
                        "Practice daily mindfulness or meditation (10-20 minutes)",
                        "Engage in regular physical exercise (30 minutes, 5x per week)",
                        "Stay hydrated throughout the day",
                        "Limit caffeine intake, especially in the afternoon",
                        "Take regular breaks during work (every 50-90 minutes)",
                        "Practice good sleep hygiene: dark room, cool temperature, no screens before bed"
                    ],
                    "duration": "ongoing"
                }
            ]
        }
    
    def get_relevant_recommendations(
        self, 
        category: str, 
        condition: str
    ) -> List[str]:
        """Retrieve relevant recommendations from knowledge base"""
        if category not in self.knowledge_base:
            return []
        
        for entry in self.knowledge_base[category]:
            if entry["condition"] == condition:
                return entry["techniques"]
        
        return []


class NLPRecommendationEngine:
    """
    NLP-based recommendation engine using semantic understanding
    and context-aware generation
    """
    
    def __init__(self):
        self.knowledge_base = RecommendationKnowledgeBase()
        logger.info("NLP Recommendation Engine initialized")
    
    def generate_recommendations(
        self,
        state_durations: Dict[int, float],
        total_duration: float,
        confidence: float,
        cognitive_metrics: Dict[str, float] = None,
        state_transitions: int = 0,
        max_recommendations: int = 5
    ) -> List[str]:
        """
        Generate personalized recommendations using NLP and RAG
        
        Args:
            state_durations: Dictionary mapping state indices to durations
            total_duration: Total duration of the session
            confidence: Confidence score of the prediction
            cognitive_metrics: Dictionary of cognitive metrics
            state_transitions: Number of state transitions
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of personalized recommendation strings
        """
        try:
            # Build context
            context = self._build_context(
                state_durations,
                total_duration,
                confidence,
                cognitive_metrics or {},
                state_transitions
            )
            
            # Generate recommendations based on context
            recommendations = self._generate_contextual_recommendations(
                context,
                max_recommendations
            )
            
            # Add personalization and formatting
            formatted_recommendations = self._format_recommendations(
                recommendations,
                context
            )
            
            logger.info(f"Generated {len(formatted_recommendations)} recommendations")
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return self._get_fallback_recommendations()
    
    def _build_context(
        self,
        state_durations: Dict[int, float],
        total_duration: float,
        confidence: float,
        cognitive_metrics: Dict[str, float],
        state_transitions: int,
        timestamp: Optional[datetime] = None,
        subject_id: str = "unknown",
        session_id: str = "unknown",
        features: Optional[Dict[str, float]] = None
    ) -> RecommendationContext:
        """Build comprehensive context for recommendation generation and explanation"""
        
        # Calculate state ratios
        stress_ratio = state_durations.get(2, 0) / total_duration if total_duration > 0 else 0
        relaxation_ratio = state_durations.get(0, 0) / total_duration if total_duration > 0 else 0
        focus_ratio = state_durations.get(1, 0) / total_duration if total_duration > 0 else 0
        
        # Determine dominant state
        if state_durations:
            dominant_state_idx = max(state_durations.items(), key=lambda x: x[1])[0]
        else:
            dominant_state_idx = 0
            
        state_labels = {0: "relaxed", 1: "focused", 2: "stressed"}
        state_label = state_labels.get(dominant_state_idx, "unknown")
        
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
            features=features or {}
        )
    
    def _generate_contextual_recommendations(
        self,
        context: RecommendationContext,
        max_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on context using RAG approach"""
        
        recommendations = []
        
        # Priority 1: Address stress if present
        if context.stress_ratio > 0.4 and context.confidence > 70:
            severity = "severe"
            condition = "high_stress"
        elif context.stress_ratio > 0.25 and context.confidence > 60:
            severity = "moderate"
            condition = "moderate_stress"
        elif context.stress_ratio > 0.15:
            severity = "mild"
            condition = "mild_stress"
        else:
            severity = None
            condition = None
        
        if condition:
            stress_recs = self.knowledge_base.get_relevant_recommendations(
                "stress_management",
                condition
            )
            recommendations.extend([
                {
                    "text": rec,
                    "category": "stress_management",
                    "priority": "high" if severity == "severe" else "medium",
                    "icon": "🧠"
                }
                for rec in stress_recs[:2]
            ])
        
        # Priority 2: Cognitive load and mental fatigue
        if context.cognitive_metrics:
            cognitive_load = context.cognitive_metrics.get('cognitive_load', 0)
            mental_fatigue = context.cognitive_metrics.get('mental_fatigue', 0)
            
            if cognitive_load > 2.0:
                load_recs = self.knowledge_base.get_relevant_recommendations(
                    "cognitive_load",
                    "high_cognitive_load"
                )
                recommendations.extend([
                    {
                        "text": rec,
                        "category": "cognitive_load",
                        "priority": "high",
                        "icon": "🎯"
                    }
                    for rec in load_recs[:1]
                ])
            
            if mental_fatigue > 0.5:
                fatigue_condition = "high_fatigue" if mental_fatigue > 0.7 else "moderate_fatigue"
                fatigue_recs = self.knowledge_base.get_relevant_recommendations(
                    "mental_fatigue",
                    fatigue_condition
                )
                recommendations.extend([
                    {
                        "text": rec,
                        "category": "mental_fatigue",
                        "priority": "high" if mental_fatigue > 0.7 else "medium",
                        "icon": "💤"
                    }
                    for rec in fatigue_recs[:1]
                ])
        
        # Priority 3: Focus optimization
        if context.focus_ratio < 0.3 and context.stress_ratio < 0.3:
            focus_recs = self.knowledge_base.get_relevant_recommendations(
                "focus_enhancement",
                "low_focus"
            )
            recommendations.extend([
                {
                    "text": rec,
                    "category": "focus_enhancement",
                    "priority": "medium",
                    "icon": "🎯"
                }
                for rec in focus_recs[:2]
            ])
        elif context.focus_ratio > 0.5:
            focus_recs = self.knowledge_base.get_relevant_recommendations(
                "focus_enhancement",
                "optimal_focus"
            )
            recommendations.extend([
                {
                    "text": rec,
                    "category": "focus_enhancement",
                    "priority": "low",
                    "icon": "✨"
                }
                for rec in focus_recs[:1]
            ])
        
        # Priority 4: Relaxation optimization
        if context.relaxation_ratio > 0.6:
            relax_recs = self.knowledge_base.get_relevant_recommendations(
                "relaxation_optimization",
                "optimal_relaxation"
            )
            recommendations.extend([
                {
                    "text": rec,
                    "category": "relaxation_optimization",
                    "priority": "low",
                    "icon": "🌿"
                }
                for rec in relax_recs[:1]
            ])
        elif context.relaxation_ratio > 0.8:
            relax_recs = self.knowledge_base.get_relevant_recommendations(
                "relaxation_optimization",
                "excessive_relaxation"
            )
            recommendations.extend([
                {
                    "text": rec,
                    "category": "relaxation_optimization",
                    "priority": "medium",
                    "icon": "⚡"
                }
                for rec in relax_recs[:1]
            ])
        
        # Priority 5: State transitions (instability indicator)
        if context.state_transitions > 10 and context.session_duration > 300:
            recommendations.append({
                "text": "Your mental state has been fluctuating frequently. Consider taking a longer break to stabilize.",
                "category": "stability",
                "priority": "medium",
                "icon": "⚖️"
            })
        
        # Always include general wellness tip if space available
        if len(recommendations) < max_recommendations:
            wellness_recs = self.knowledge_base.get_relevant_recommendations(
                "general_wellness",
                "baseline"
            )
            if wellness_recs:
                recommendations.append({
                    "text": np.random.choice(wellness_recs),
                    "category": "general_wellness",
                    "priority": "low",
                    "icon": "💚"
                })
        
        # Sort by priority and limit
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations[:max_recommendations]
    
    def _format_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        context: RecommendationContext
    ) -> List[str]:
        """Format recommendations with personalization and context"""
        
        formatted = []
        
        # Add contextual header based on dominant state
        if context.stress_ratio > 0.3:
            header = f"Based on your elevated stress levels ({context.stress_ratio*100:.1f}% of session):"
        elif context.focus_ratio > 0.5:
            header = f"You've maintained good focus ({context.focus_ratio*100:.1f}% of session):"
        elif context.relaxation_ratio > 0.6:
            header = f"You're in a relaxed state ({context.relaxation_ratio*100:.1f}% of session):"
        else:
            header = "Based on your current mental state:"
        
        formatted.append(header)
        formatted.append("")  # Empty line for readability
        
        # Format each recommendation
        for i, rec in enumerate(recommendations, 1):
            icon = rec.get("icon", "•")
            text = rec["text"]
            formatted.append(f"{icon} {text}")
        
        # Add confidence note if low
        if context.confidence < 70:
            formatted.append("")
            formatted.append(f"⚠️ Note: These recommendations are based on {context.confidence:.1f}% confidence. Consider a longer session for more accurate insights.")
        
        return formatted
    
    def _get_fallback_recommendations(self) -> List[str]:
        """Provide fallback recommendations if generation fails"""
        return [
            "Unable to generate personalized recommendations at this time.",
            "",
            "General wellness tips:",
            "• Take regular breaks every 50-90 minutes",
            "• Practice deep breathing exercises",
            "• Stay hydrated throughout the day",
            "• Maintain good posture while working",
            "• Get adequate sleep (7-9 hours per night)"
        ]
    
    def generate_detailed_report(
        self,
        state_durations: Dict[int, float],
        total_duration: float,
        confidence: float,
        cognitive_metrics: Dict[str, float] = None,
        state_transitions: int = 0
    ) -> Dict[str, Any]:
        """
        Generate a detailed report with recommendations and insights
        
        Returns:
            Dictionary containing recommendations, metrics, and insights
        """
        try:
            context = self._build_context(
                state_durations,
                total_duration,
                confidence,
                cognitive_metrics or {},
                state_transitions
            )
            
            recommendations = self.generate_recommendations(
                state_durations,
                total_duration,
                confidence,
                cognitive_metrics,
                state_transitions
            )
            
            # Generate insights
            insights = self._generate_insights(context)
            
            # Calculate wellness score
            wellness_score = self._calculate_wellness_score(context)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "session_summary": {
                    "duration_minutes": total_duration / 60,
                    "dominant_state": context.state_label,
                    "confidence": context.confidence,
                    "state_transitions": state_transitions
                },
                "state_distribution": {
                    "relaxed": f"{context.relaxation_ratio * 100:.1f}%",
                    "focused": f"{context.focus_ratio * 100:.1f}%",
                    "stressed": f"{context.stress_ratio * 100:.1f}%"
                },
                "wellness_score": wellness_score,
                "insights": insights,
                "recommendations": recommendations,
                "cognitive_metrics": cognitive_metrics or {}
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {str(e)}")
            return {
                "error": "Failed to generate report",
                "recommendations": self._get_fallback_recommendations()
            }
    
    def _generate_insights(self, context: RecommendationContext) -> List[str]:
        """Generate insights based on context analysis"""
        insights = []
        
        # Stress insights
        if context.stress_ratio > 0.4:
            insights.append(f"⚠️ High stress detected: You spent {context.stress_ratio*100:.1f}% of your session in a stressed state. This may impact your productivity and well-being.")
        elif context.stress_ratio > 0.25:
            insights.append(f"⚡ Moderate stress: {context.stress_ratio*100:.1f}% of your session showed stress indicators. Consider implementing stress management techniques.")
        
        # Focus insights
        if context.focus_ratio > 0.6:
            insights.append(f"✨ Excellent focus: You maintained high focus for {context.focus_ratio*100:.1f}% of the session. Great work!")
        elif context.focus_ratio < 0.2:
            insights.append(f"🎯 Low focus detected: Only {context.focus_ratio*100:.1f}% of your session showed focused attention. Try focus-enhancing techniques.")
        
        # Relaxation insights
        if context.relaxation_ratio > 0.7:
            insights.append(f"🌿 Highly relaxed: {context.relaxation_ratio*100:.1f}% of your session was in a relaxed state. This is great for recovery but may indicate low engagement.")
        
        # State stability insights
        if context.state_transitions > 15 and context.session_duration > 300:
            transitions_per_min = context.state_transitions / (context.session_duration / 60)
            insights.append(f"⚖️ Frequent state changes: {context.state_transitions} transitions detected ({transitions_per_min:.1f} per minute). This may indicate mental instability or environmental distractions.")
        
        # Cognitive load insights
        if context.cognitive_metrics:
            cognitive_load = context.cognitive_metrics.get('cognitive_load', 0)
            if cognitive_load > 2.0:
                insights.append(f"🧠 High cognitive load: Your mental workload appears elevated. Consider breaking tasks into smaller chunks.")
        
        # Confidence insights
        if context.confidence < 70:
            insights.append(f"📊 Low confidence: The analysis confidence is {context.confidence:.1f}%. Longer sessions provide more accurate insights.")
        
        return insights if insights else ["📈 Your session data looks normal. Keep up the good work!"]
    
    def _calculate_wellness_score(self, context: RecommendationContext) -> Dict[str, Any]:
        """Calculate overall wellness score (0-100)"""
        
        # Base score starts at 100
        score = 100.0
        
        # Deduct for stress
        score -= context.stress_ratio * 40  # Max -40 points
        
        # Add for good focus
        score += context.focus_ratio * 20  # Max +20 points
        
        # Slight deduction for excessive relaxation
        if context.relaxation_ratio > 0.8:
            score -= (context.relaxation_ratio - 0.8) * 30
        
        # Deduct for state instability
        if context.state_transitions > 10 and context.session_duration > 300:
            instability_factor = min((context.state_transitions - 10) / 20, 1.0)
            score -= instability_factor * 15
        
        # Deduct for cognitive overload
        if context.cognitive_metrics:
            cognitive_load = context.cognitive_metrics.get('cognitive_load', 0)
            if cognitive_load > 2.0:
                score -= min((cognitive_load - 2.0) * 10, 20)
            
            mental_fatigue = context.cognitive_metrics.get('mental_fatigue', 0)
            if mental_fatigue > 0.5:
                score -= mental_fatigue * 20
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        # Determine rating
        if score >= 80:
            rating = "Excellent"
            emoji = "🌟"
        elif score >= 60:
            rating = "Good"
            emoji = "👍"
        elif score >= 40:
            rating = "Fair"
            emoji = "⚠️"
        else:
            rating = "Needs Attention"
            emoji = "🔴"
        
        return {
            "score": round(score, 1),
            "rating": rating,
            "emoji": emoji,
            "description": f"{emoji} {rating} - {score:.1f}/100"
        }
    
    def generate_medical_explanation(
        self,
        context: RecommendationContext,
        occupation: str = "default"
    ) -> Dict[str, Any]:
        """
        Generate a professional medical explanation based on EEG analysis.
        Rule-based engine replacing heavy NLP models for efficiency.
        """
        try:
            state_info = self.knowledge_base.knowledge_base["medical_terminology"].get(
                context.state_label, {}
            )
            
            explanation = {
                "clinical_observation": self._generate_clinical_observation(context, state_info),
                "technical_analysis": self._generate_technical_analysis(context, state_info, occupation),
                "interpretation": self._generate_interpretation(context, state_info),
                "recommendations": self.generate_recommendations(
                    {0: context.relaxation_ratio, 1: context.focus_ratio, 2: context.stress_ratio},
                    1.0, # normalized
                    context.confidence,
                    context.cognitive_metrics,
                    context.state_transitions
                ),
                "safety_assessment": self._generate_safety_assessment(context, occupation),
                "metadata": {
                    "timestamp": context.timestamp.isoformat(),
                    "subject_id": context.subject_id,
                    "session_id": context.session_id,
                    "confidence": float(context.confidence)
                }
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating medical explanation: {str(e)}")
            raise

    def _generate_clinical_observation(self, context: RecommendationContext, state_info: Dict) -> str:
        """Generate clinical observation section"""
        terms = state_info.get("terms", [])
        observation = f"Clinical observation indicates a state of {context.state_label} "
        observation += f"with key characteristics: {', '.join(terms)}. "
        observation += f"The confidence level of this assessment is {context.confidence:.2%}. "
        observation += state_info.get("clinical_significance", "")
        return observation

    def _generate_technical_analysis(self, context: RecommendationContext, state_info: Dict, occupation: str) -> str:
        """Generate technical analysis section"""
        analysis = "Technical Analysis:\n"
        
        # Power band analysis if features provided
        if context.features:
            analysis += "Wave Patterns:\n"
            for feature, value in context.features.items():
                if feature in state_info.get("normal_range", {}):
                    n_range = state_info["normal_range"][feature]
                    analysis += f"- {feature}: {value:.2f} "
                    if n_range[0] <= value <= n_range[1]:
                        analysis += "(within normal range)\n"
                    else:
                        analysis += "(outside normal range)\n"
        
        # Amplitude/Safety analysis
        analysis += "\nSafety Assessment:\n"
        thresholds = self.knowledge_base.knowledge_base["safety_thresholds"]
        for metric, val in context.features.items():
            if "amplitude" in metric.lower():
                thresh_data = thresholds.get(metric, thresholds.get("beta_amplitude", {}))
                thresh = thresh_data.get(occupation, thresh_data.get("default", 50))
                analysis += f"- {metric}: {val:.2f} μV "
                analysis += "(within safety limits)\n" if val <= thresh else "(exceeds safety limits)\n"
        
        return analysis

    def _generate_interpretation(self, context: RecommendationContext, state_info: Dict) -> str:
        """Generate clinical interpretation"""
        interpretation = f"The EEG analysis suggests a {context.state_label} state. "
        interpretation += state_info.get("clinical_significance", "")
        if context.stress_ratio > 0.3:
            interpretation += " Prolonged stress indicators may require intervention."
        return interpretation

    def _generate_safety_assessment(self, context: RecommendationContext, occupation: str) -> Dict[str, Any]:
        """Generate structured safety assessment"""
        assessment = {"alert_level": "mild", "concerns": [], "immediate_actions": []}
        thresholds = self.knowledge_base.knowledge_base["safety_thresholds"]
        
        for metric, val in context.features.items():
            if "amplitude" in metric.lower():
                thresh_data = thresholds.get(metric, thresholds.get("beta_amplitude", {}))
                thresh = thresh_data.get(occupation, thresh_data.get("default", 50))
                
                if val > thresh:
                    assessment["concerns"].append(f"Elevated {metric} detected")
                    if val > thresh * 1.5:
                        assessment["alert_level"] = "severe"
                        assessment["immediate_actions"].append("Urgent: Seek rest or professional consultation.")
                    elif val > thresh * 1.2:
                        assessment["alert_level"] = "moderate"
                        assessment["immediate_actions"].append("Warning: Consider taking a break.")
        
        return assessment

    def format_medical_report(self, explanation: Dict) -> str:
        """Format the explanation into a professional report string"""
        report = "EEG Clinical Analysis Report\n" + "="*40 + "\n"
        report += f"Subject: {explanation['metadata']['subject_id']}\n"
        report += f"Session: {explanation['metadata']['session_id']}\n\n"
        
        if explanation["safety_assessment"]["alert_level"] != "mild":
            report += f"⚠️ ALERT: {explanation['safety_assessment']['alert_level'].upper()}\n"
            for action in explanation["safety_assessment"]["immediate_actions"]:
                report += f"• {action}\n"
            report += "\n"
            
        report += "1. Clinical Observation\n" + explanation["clinical_observation"] + "\n\n"
        report += "2. Interpretation\n" + explanation["interpretation"] + "\n\n"
        report += "3. Recommendations\n"
        for i, rec in enumerate(explanation["recommendations"][:3], 1):
            if isinstance(rec, dict):
                 report += f"{i}. {rec.get('text', '')}\n"
            else:
                 report += f"{i}. {str(rec).split(':', 1)[-1].strip() if ':' in str(rec) else str(rec)}\n"
        
        return report

    def save_report(self, report: Dict[str, Any], filepath: str = None) -> str:
        """Save report to JSON file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"reports/eeg_report_{timestamp}.json"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Report saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return None


# Convenience function for quick recommendations
def get_recommendations(
    state_durations: Dict[int, float],
    total_duration: float,
    confidence: float,
    **kwargs
) -> List[str]:
    """
    Quick function to get recommendations
    
    Args:
        state_durations: Dictionary mapping state indices to durations
        total_duration: Total duration of the session
        confidence: Confidence score
        **kwargs: Additional parameters (cognitive_metrics, state_transitions, etc.)
    
    Returns:
        List of recommendation strings
    """
    engine = NLPRecommendationEngine()
    return engine.generate_recommendations(
        state_durations,
        total_duration,
        confidence,
        **kwargs
    )
import numpy as np
import shap
import lime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

class ModelInterpretability:
    """Handles model interpretability using SHAP and LIME"""
    
    def __init__(self, model):
        self.model = model
        self.feature_names = None
        
    def set_feature_names(self, names: List[str]):
        """Set feature names for explanations"""
        self.feature_names = names
        
    def explain_with_shap(self, X: np.ndarray, n_samples: int = 20) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        explainer = shap.DeepExplainer(self.model, X[:n_samples])
        shap_values = explainer.shap_values(X[:n_samples])
        
        return {
            "feature_importance": {
                str(i): np.abs(shap_values[i]).mean(axis=0)
                for i in range(len(shap_values))
            },
            "explainer": explainer
        }
        
    def explain_with_lime(self, X: np.ndarray, sample_idx: int = 0, num_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanations"""
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            class_names=[str(i) for i in range(self.model.output_shape[-1])]
        )
        explanation = explainer.explain_instance(
            X[sample_idx],
            self.model.predict,
            num_features=num_features
        )
        
        return {
            "feature_importance": dict(explanation.as_list()),
            "explanation": explanation
        }
        
    def calibrate_confidence(self, X_val: np.ndarray, y_val: np.ndarray, method: str = "temperature_scaling") -> Dict[str, Any]:
        """Calibrate model confidence scores"""
        if method == "temperature_scaling":
            from sklearn.calibration import CalibratedClassifierCV
            calibrator = CalibratedClassifierCV(self.model, cv=5, method='sigmoid')
            calibrator.fit(X_val, y_val)
            return {
                "temperature": calibrator.calibrators_[0].T_,
                "expected_calibration_error": self._compute_ece(calibrator.predict_proba(X_val), y_val)
            }
        return {}
        
    def _compute_ece(self, preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(preds >= bin_lower, preds < bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin] == np.argmax(preds[in_bin], axis=1))
                avg_confidence_in_bin = np.mean(preds[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
        
    def plot_calibration_curve(self, preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> plt.Figure:
        """Plot reliability diagram"""
        fig, ax = plt.subplots(figsize=(8, 6))
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(preds >= bin_lower, preds < bin_upper)
            if np.any(in_bin):
                accuracy = np.mean(labels[in_bin] == np.argmax(preds[in_bin], axis=1))
                confidence = np.mean(preds[in_bin])
                accuracies.append(accuracy)
                confidences.append(confidence)
                
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(confidences, accuracies, 's-', label='Model calibration')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability Diagram')
        ax.legend()
        
        return fig
"""
Fast Explainable AI (XAI) module using linear approximations and Tree-SHAP.
Provides real-time feature importance without the computational overhead of standard SHAP.
Optimized for low-latency trading decisions.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class LinearApproximationLIME:
    """
    Lightweight LIME-like explainer using local linear approximation.
    Much faster than kernel SHAP by using simple linear regression locally.
    """
    
    def __init__(
        self,
        n_samples: int = 50,
        perturbation_scale: float = 0.1,
        kernel_width: float = 1.0,
    ):
        self.n_samples = n_samples
        self.perturbation_scale = perturbation_scale
        self.kernel_width = kernel_width
        
        # Cache for recent explanations
        self.explanation_cache: deque = deque(maxlen=100)
    
    def explain(
        self,
        model_predict_fn,
        x_instance: np.ndarray,
        X_background: np.ndarray,
        n_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate local explanation for a single prediction.
        
        Args:
            model_predict_fn: Function that takes X and returns predictions
            x_instance: Instance to explain (1D array)
            X_background: Background data for computing statistics
            n_features: Number of top features to return
        
        Returns:
            Dictionary with feature weights and importance scores
        """
        if len(x_instance.shape) > 1:
            x_instance = x_instance.flatten()
        
        n_features_total = len(x_instance)
        n_features_return = n_features or min(10, n_features_total)
        
        # Compute background statistics
        bg_mean = np.mean(X_background, axis=0)
        bg_std = np.std(X_background, axis=0) + 1e-8
        
        # Generate perturbed samples
        perturbed_X = np.zeros((self.n_samples, n_features_total))
        distances = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            # Randomly perturb features
            mask = np.random.random(n_features_total) < 0.5
            
            perturbed = x_instance.copy()
            perturbed[mask] = bg_mean[mask] + np.random.randn(mask.sum()) * bg_std[mask] * self.perturbation_scale
            
            perturbed_X[i] = perturbed
            
            # Compute distance from original
            distances[i] = np.sqrt(np.sum((perturbed - x_instance) ** 2))
        
        # Get predictions
        try:
            predictions = model_predict_fn(perturbed_X)
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
        
        # Compute kernel weights (exponential decay with distance)
        kernel_weights = np.exp(-distances ** 2 / self.kernel_width ** 2)
        
        # Fit weighted linear regression
        # y = w @ X + b
        X_centered = perturbed_X - x_instance
        
        # Weighted least squares: (X^T W X)^-1 X^T W y
        W = np.diag(kernel_weights)
        
        XtW = X_centered.T @ W
        XtWX = XtW @ X_centered
        
        # Add regularization for stability
        XtWX += np.eye(n_features_total) * 1e-6
        
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
        
        XtWy = XtW @ predictions
        weights = XtWX_inv @ XtWy
        
        # Sort by absolute importance
        importance_indices = np.argsort(np.abs(weights))[::-1][:n_features_return]
        
        # Build explanation
        explanation = {
            'feature_importance': {},
            'weights': weights,
            'top_features': [],
            'prediction_at_instance': model_predict_fn(x_instance.reshape(1, -1))[0],
        }
        
        for idx in importance_indices:
            feature_name = f'feature_{idx}'
            importance = float(weights[idx])
            
            explanation['feature_importance'][feature_name] = importance
            explanation['top_features'].append({
                'feature': feature_name,
                'importance': importance,
                'abs_importance': abs(importance),
                'value': float(x_instance[idx]),
            })
        
        # Cache explanation
        self.explanation_cache.append(explanation)
        
        return explanation
    
    def get_aggregate_importance(self, n_recent: int = 50) -> Dict[str, float]:
        """Get aggregate feature importance from recent explanations."""
        if len(self.explanation_cache) == 0:
            return {}
        
        # Take last N explanations
        recent = list(self.explanation_cache)[-n_recent:]
        
        # Aggregate absolute weights
        aggregate = {}
        count = 0
        
        for exp in recent:
            if 'weights' not in exp:
                continue
            
            for i, w in enumerate(exp['weights']):
                key = f'feature_{i}'
                if key not in aggregate:
                    aggregate[key] = []
                aggregate[key].append(abs(w))
            
            count += 1
        
        # Average
        result = {k: np.mean(v) for k, v in aggregate.items()}
        
        # Sort
        sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_result


class FastTreeSHAP:
    """
    Simplified Tree-SHAP implementation for tree-based models.
    Optimized for speed over perfect accuracy.
    """
    
    def __init__(self, max_depth: int = 10, n_subsets: int = 100):
        self.max_depth = max_depth
        self.n_subsets = n_subsets
        
        # Cached Shapley values
        self.shap_cache: deque = deque(maxlen=100)
    
    def approximate_shap_values(
        self,
        model,
        x_instance: np.ndarray,
        X_reference: np.ndarray,
        n_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Approximate SHAP values using sampling approach.
        
        For tree models, uses path-dependent attribution.
        For other models, falls back to sampling-based approximation.
        """
        if len(x_instance.shape) > 1:
            x_instance = x_instance.flatten()
        
        n_features_total = len(x_instance)
        n_features_return = n_features or n_features_total
        
        # Check if model has tree structure
        if hasattr(model, 'get_booster') or hasattr(model, 'trees_'):
            # Use tree-specific approximation
            shap_values = self._tree_shap_approx(model, x_instance, X_reference)
        else:
            # Use sampling-based approximation
            shap_values = self._sampling_shap(model, x_instance, X_reference)
        
        # Sort by absolute value
        importance_order = np.argsort(np.abs(shap_values))[::-1][:n_features_return]
        
        explanation = {
            'shap_values': shap_values,
            'feature_importance': {},
            'top_features': [],
            'base_value': float(np.mean([model.predict(np.zeros_like(x_instance).reshape(1, -1))[0]])),
        }
        
        for idx in importance_order:
            feature_name = f'feature_{idx}'
            shap_val = float(shap_values[idx])
            
            explanation['feature_importance'][feature_name] = shap_val
            explanation['top_features'].append({
                'feature': feature_name,
                'shap_value': shap_val,
                'abs_shap_value': abs(shap_val),
                'contribution_direction': 'positive' if shap_val > 0 else 'negative',
            })
        
        # Cache
        self.shap_cache.append(explanation)
        
        return explanation
    
    def _tree_shap_approx(
        self,
        model,
        x_instance: np.ndarray,
        X_reference: np.ndarray,
    ) -> np.ndarray:
        """Approximate tree SHAP using feature frequencies in splits."""
        n_features = len(x_instance)
        shap_values = np.zeros(n_features)
        
        # Simplified: use feature usage frequency as proxy
        try:
            if hasattr(model, 'feature_importances_'):
                # Normalize and scale by deviation from mean
                importance = model.feature_importances_
                
                # Scale by how much instance differs from reference mean
                ref_mean = np.mean(X_reference, axis=0)
                deviation = x_instance - ref_mean
                
                shap_values = importance * np.sign(deviation) * np.abs(deviation)
            else:
                # Fallback to uniform
                shap_values = np.ones(n_features) / n_features
        except Exception as e:
            logger.warning(f"Tree SHAP approximation failed: {e}")
            shap_values = np.zeros(n_features)
        
        return shap_values
    
    def _sampling_shap(
        self,
        model,
        x_instance: np.ndarray,
        X_reference: np.ndarray,
    ) -> np.ndarray:
        """Sampling-based SHAP approximation."""
        n_features = len(x_instance)
        shap_values = np.zeros(n_features)
        
        base_pred = model.predict(x_instance.reshape(1, -1))[0]
        
        # Sample random feature subsets
        for _ in range(self.n_subsets):
            # Random subset
            mask = np.random.random(n_features) < 0.5
            if mask.sum() == 0 or mask.sum() == n_features:
                continue
            
            # Create hybrid sample
            hybrid = x_instance.copy()
            hybrid[~mask] = np.random.choice(X_reference[:, ~mask].flatten())
            
            # Predict
            try:
                pred = model.predict(hybrid.reshape(1, -1))[0]
                
                # Attribute difference to active features
                diff = pred - base_pred
                shap_values[mask] += diff / mask.sum()
            except Exception:
                pass
        
        # Normalize
        shap_values /= self.n_subsets
        
        return shap_values


class FastXAIExplainer:
    """
    Combined fast XAI explainer using both LIME and Tree-SHAP.
    Provides consensus explanations for robustness.
    """
    
    def __init__(
        self,
        model=None,
        use_lime: bool = True,
        use_shap: bool = True,
        lime_weight: float = 0.5,
        shap_weight: float = 0.5,
    ):
        self.model = model
        self.use_lime = use_lime
        self.use_shap = use_shap
        self.lime_weight = lime_weight
        self.shap_weight = shap_weight
        
        self.lime_explainer = LinearApproximationLIME() if use_lime else None
        self.shap_explainer = FastTreeSHAP() if use_shap else None
        
        # Background data
        self.X_background: Optional[np.ndarray] = None
    
    def set_background_data(self, X: np.ndarray, max_samples: int = 1000):
        """Set background/reference data for explanations."""
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        self.X_background = X
        logger.info(f"Background data set with {len(X)} samples")
    
    def explain(
        self,
        x_instance: np.ndarray,
        n_top_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single instance.
        
        Returns:
            Dictionary with combined feature importance
        """
        if self.X_background is None:
            return {"error": "Background data not set"}
        
        results = {
            'lime': None,
            'shap': None,
            'consensus': {},
            'top_features': [],
        }
        
        # LIME explanation
        if self.use_lime and self.lime_explainer:
            results['lime'] = self.lime_explainer.explain(
                self.model.predict,
                x_instance,
                self.X_background,
                n_features=n_top_features,
            )
        
        # SHAP explanation
        if self.use_shap and self.shap_explainer and self.model:
            results['shap'] = self.shap_explainer.approximate_shap_values(
                self.model,
                x_instance,
                self.X_background,
                n_features=n_top_features,
            )
        
        # Combine into consensus
        consensus_importance = {}
        
        if results['lime'] and 'weights' in results['lime']:
            lime_weights = np.abs(results['lime']['weights'])
            for i, w in enumerate(lime_weights):
                key = f'feature_{i}'
                consensus_importance[key] = self.lime_weight * w
        
        if results['shap'] and 'shap_values' in results['shap']:
            shap_values = np.abs(results['shap']['shap_values'])
            for i, s in enumerate(shap_values):
                key = f'feature_{i}'
                if key in consensus_importance:
                    consensus_importance[key] += self.shap_weight * s
                else:
                    consensus_importance[key] = self.shap_weight * s
        
        # Sort and get top features
        sorted_features = sorted(consensus_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feat, imp in sorted_features[:n_top_features]:
            results['top_features'].append({
                'feature': feat,
                'importance': float(imp),
            })
        
        results['consensus'] = consensus_importance
        
        return results
    
    def get_global_importance(self) -> Dict[str, float]:
        """Get global feature importance from aggregated explanations."""
        lime_agg = self.lime_explainer.get_aggregate_importance() if self.lime_explainer else {}
        
        # Combine with model's built-in importance if available
        if self.model and hasattr(self.model, 'feature_importances_'):
            model_imp = self.model.feature_importances_
            for i, imp in enumerate(model_imp):
                key = f'feature_{i}'
                if key in lime_agg:
                    lime_agg[key] = (lime_agg[key] + imp) / 2
                else:
                    lime_agg[key] = imp
        
        return lime_agg


def main():
    """Test XAI module."""
    import psutil
    
    print(f"Initial RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Create synthetic model (simple linear for testing)
    class SimpleModel:
        def __init__(self):
            self.coef = np.array([1.0, -0.5, 0.3, -0.2, 0.1, 0.05, -0.03, 0.02, -0.01, 0.005])
            self.feature_importances_ = np.abs(self.coef) / np.sum(np.abs(self.coef))
        
        def predict(self, X):
            return X @ self.coef
    
    model = SimpleModel()
    
    # Generate background data
    np.random.seed(42)
    X_bg = np.random.randn(500, 10)
    
    # Instance to explain
    x_instance = np.random.randn(10)
    
    # Create explainer
    explainer = FastXAIExplainer(
        model=model,
        use_lime=True,
        use_shap=True,
    )
    
    explainer.set_background_data(X_bg)
    
    # Explain
    print("\n--- Generating Explanation ---")
    explanation = explainer.explain(x_instance, n_top_features=5)
    
    print(f"\nTop 5 Features:")
    for feat in explanation['top_features']:
        print(f"  {feat['feature']}: {feat['importance']:.4f}")
    
    # Global importance
    print("\n--- Global Feature Importance ---")
    global_imp = explainer.get_global_importance()
    for feat, imp in list(global_imp.items())[:5]:
        print(f"  {feat}: {imp:.4f}")
    
    print(f"\nFinal RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()

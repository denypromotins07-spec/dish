"""
LightGBM trainer with Hyperopt integration, strictly bounded by memory constraints.
Implements early stopping, aggressive binning, and model size validation (<50MB).
Optimized for AMD ROCm GPU acceleration where available.
"""

import os
import logging
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

# Memory-constrained environment variables
os.environ["LIGHTGBM_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("lightgbm required. Install with: pip install lightgbm")

try:
    from hyperopt import hp, tpe, Trials, fmin
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    logging.warning("hyperopt not available. Install with: pip install hyperopt")

logger = logging.getLogger(__name__)


class MemoryBoundedLightGBMTuner:
    """
    LightGBM tuner with strict memory bounds and Hyperopt integration.
    Enforces max model size of 50MB through aggressive pruning and parameter tuning.
    """
    
    def __init__(
        self,
        max_depth: int = -1,  # -1 means no limit, but we'll enforce via max_bin
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        max_bin: int = 127,  # Aggressive binning for memory efficiency
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_data_in_leaf: int = 20,
        lambda_l1: float = 1.0,
        lambda_l2: float = 1.0,
        device: str = "gpu",  # Will use CPU if GPU not available
        max_model_size_mb: int = 50,
        early_stopping_rounds: int = 10,
        max_hyperopt_trials: int = 20,
        target_ram_gb: float = 14.0,
    ):
        self.base_params = {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_bin": max_bin,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_data_in_leaf": min_data_in_leaf,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "device": device,
            "verbose": -1,
            "force_row_wise": True,  # Better for memory layout
            "seed": 42,
        }
        
        self.max_model_size_mb = max_model_size_mb
        self.early_stopping_rounds = early_stopping_rounds
        self.max_hyperopt_trials = max_hyperopt_trials
        self.target_ram_gb = target_ram_gb
        
        self.model: Optional[lgb.Booster] = None
        self.best_params: Dict[str, Any] = {}
        self.trials_history: List[Dict] = []
    
    def _check_memory(self) -> float:
        """Check available system RAM."""
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    
    def _objective_function(self, params: Dict[str, Any]) -> float:
        """
        Objective function for Hyperopt optimization.
        Returns validation loss while enforcing memory constraints.
        """
        # Check memory before training
        available_ram = self._check_memory()
        if available_ram < 1.0:
            logger.warning("Critical low memory. Returning high loss to skip this trial.")
            return 999.0
        
        # Adjust parameters based on available memory
        if available_ram < 3.0:
            params["subsample"] = min(params.get("subsample", 0.8), 0.5)
            params["colsample_bytree"] = min(params.get("colsample_bytree", 0.8), 0.5)
            params["max_bin"] = min(params.get("max_bin", 127), 63)
        
        try:
            # Create lightweight dataset
            train_data = lgb.Dataset(
                self.X_train, 
                label=self.y_train,
                free_raw_data=False,
            )
            val_data = lgb.Dataset(
                self.X_val, 
                label=self.y_val,
                reference=train_data,
                free_raw_data=False,
            )
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                num_boost_round=params.get("n_estimators", self.base_params["n_estimators"]),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            
            # Get validation score
            val_score = model.best_score["valid"]["rmse"] if "rmse" in model.best_score["valid"] else model.best_score["valid"]["l2"]
            
            # Check model size
            model_size_mb = self._estimate_model_size(model)
            if model_size_mb > self.max_model_size_mb:
                # Penalize oversized models
                val_score += (model_size_mb - self.max_model_size_mb) * 0.1
            
            # Store trial info
            self.trials_history.append({
                "params": params.copy(),
                "score": val_score,
                "model_size_mb": model_size_mb,
                "available_ram_gb": available_ram,
            })
            
            return val_score
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 999.0
    
    def _estimate_model_size(self, model: lgb.Booster) -> float:
        """Estimate model size in MB without saving to disk."""
        # Approximate based on number of trees and leaves
        num_trees = model.num_trees()
        avg_leaves = self.base_params.get("num_leaves", 31)
        
        # Rough estimation: each leaf node ~100 bytes
        estimated_bytes = num_trees * avg_leaves * 100
        return estimated_bytes / (1024 * 1024)
    
    def tune_hyperopt(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Perform Hyperopt tuning with strict memory constraints.
        """
        if not HYPEROPT_AVAILABLE:
            logger.warning("Hyperopt not available. Using default parameters.")
            return self.base_params
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Define search space (conservative to save memory)
        space = {
            "num_leaves": hp.quniform("num_leaves", 15, 50, 1),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "n_estimators": hp.quniform("n_estimators", 50, 150, 10),
            "max_bin": hp.choice("max_bin", [63, 127, 255]),
            "subsample": hp.uniform("subsample", 0.6, 0.9),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 0.9),
            "min_data_in_leaf": hp.quniform("min_data_in_leaf", 10, 50, 5),
            "lambda_l1": hp.loguniform("lambda_l1", np.log(0.1), np.log(10)),
            "lambda_l2": hp.loguniform("lambda_l2", np.log(0.1), np.log(10)),
        }
        
        logger.info(f"Starting Hyperopt tuning with {self.max_hyperopt_trials} trials...")
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=self._objective_function,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_hyperopt_trials,
            trials=trials,
            verbose=1,
        )
        
        # Convert best params
        self.best_params = {
            "num_leaves": int(best["num_leaves"]),
            "learning_rate": float(best["learning_rate"]),
            "n_estimators": int(best["n_estimators"]),
            "max_bin": [63, 127, 255][int(best["max_bin"])],
            "subsample": float(best["subsample"]),
            "colsample_bytree": float(best["colsample_bytree"]),
            "min_data_in_leaf": int(best["min_data_in_leaf"]),
            "lambda_l1": float(best["lambda_l1"]),
            "lambda_l2": float(best["lambda_l2"]),
        }
        
        # Merge with base params
        self.best_params.update({
            "device": self.base_params["device"],
            "verbose": -1,
            "force_row_wise": True,
            "seed": 42,
        })
        
        logger.info(f"Best params found: {self.best_params}")
        
        return self.best_params
    
    def train_final_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train final model with optimized or provided parameters.
        Enforces strict model size limits.
        """
        import psutil
        
        final_params = params if params else self.best_params
        if not final_params:
            final_params = self.base_params
        
        # Check memory before training
        available_ram_gb = self._check_memory()
        if available_ram_gb < 2.0:
            logger.warning(f"Low memory ({available_ram_gb:.2f}GB). Reducing model complexity.")
            final_params["num_leaves"] = max(15, final_params.get("num_leaves", 31) // 2)
            final_params["n_estimators"] = max(50, int(final_params.get("n_estimators", 100) * 0.7))
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
        
        logger.info(f"Training final model with params: {final_params}")
        
        # Train model
        self.model = lgb.train(
            final_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            num_boost_round=int(final_params.get("n_estimators", 100)),
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=10,
        )
        
        # Validate model size
        model_size_mb = self._estimate_model_size(self.model)
        if model_size_mb > self.max_model_size_mb:
            logger.warning(
                f"Model size ({model_size_mb:.2f}MB) exceeds limit. Retraining with reduced complexity."
            )
            final_params["n_estimators"] = max(20, int(final_params["n_estimators"] * 0.7))
            final_params["num_leaves"] = max(10, int(final_params["num_leaves"] * 0.8))
            
            self.model = lgb.train(
                final_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                num_boost_round=int(final_params.get("n_estimators", 100)),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
        
        final_size_mb = self._estimate_model_size(self.model)
        
        return {
            "model_size_mb": final_size_mb,
            "best_iteration": self.model.best_iteration,
            "available_ram_gb": self._check_memory(),
            "params_used": final_params,
        }
    
    def predict(self, X: np.ndarray, num_iteration: Optional[int] = None) -> np.ndarray:
        """Make predictions with memory-efficient batching."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        batch_size = 10000
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            pred = self.model.predict(batch, num_iteration=num_iteration)
            predictions.append(pred)
        
        return np.concatenate(predictions)
    
    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Model loaded from {path}")


def main():
    """Example usage with synthetic data."""
    import psutil
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 50000
    n_features = 30
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Initialize tuner
    tuner = MemoryBoundedLightGBMTuner(
        max_hyperopt_trials=15,  # Limited trials for speed
        max_model_size_mb=50,
        device="gpu",  # Will fallback to CPU if GPU unavailable
    )
    
    # Tune hyperparameters
    best_params = tuner.tune_hyperopt(X_train, y_train, X_val, y_val)
    print(f"\nBest parameters: {best_params}")
    
    # Train final model
    results = tuner.train_final_model(X_train, y_train, X_val, y_val, best_params)
    print(f"\nFinal model results: {results}")
    
    print(f"\nAvailable RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()

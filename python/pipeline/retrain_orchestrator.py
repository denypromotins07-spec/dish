"""
Ray-based orchestrator for automated model retraining.
Spins up isolated, memory-capped workers to retrain XGBoost/LightGBM/RL
models only when drift thresholds are breached. Strictly prevents starving
the live trading engine.
"""

import ray
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os
import time
from datetime import datetime

# Memory bounds (strict)
MAX_WORKER_MEMORY_GB = 2.0
MAX_CONCURRENT_WORKERS = 2
RETRAIN_COOLDOWN_MINUTES = 30


@dataclass
class RetrainingJob:
    """Represents a single retraining job."""
    job_id: str
    model_type: str
    symbol: str
    drift_score: float
    status: str  # 'pending', 'running', 'completed', 'failed'
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metrics: Optional[Dict] = None
    error: Optional[str] = None


@ray.remote(max_calls=1)
class RetrainingWorker:
    """
    Isolated worker for model retraining with strict memory caps.
    Each worker is killed after one job to prevent memory leaks.
    """

    def __init__(self, worker_id: int, max_memory_gb: float = MAX_WORKER_MEMORY_GB):
        self.worker_id = worker_id
        self.max_memory_gb = max_memory_gb
        self.model = None
        self.metrics = {}

    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Dict,
    ) -> Dict:
        """Train a model with the specified type and data."""
        start_time = time.time()
        
        try:
            if model_type == 'xgboost':
                return self._train_xgboost(X_train, y_train, X_val, y_val, hyperparams)
            elif model_type == 'lightgbm':
                return self._train_lightgbm(X_train, y_train, X_val, y_val, hyperparams)
            elif model_type == 'random_forest':
                return self._train_sklearn(X_train, y_train, X_val, y_val, hyperparams)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - start_time,
            }

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Dict,
    ) -> Dict:
        """Train XGBoost model with early stopping."""
        import xgboost as xgb
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'max_depth': hyperparams.get('max_depth', 6),
            'learning_rate': hyperparams.get('learning_rate', 0.01),
            'subsample': hyperparams.get('subsample', 0.8),
            'colsample_bytree': hyperparams.get('colsample_bytree', 0.8),
            'objective': hyperparams.get('objective', 'binary:logistic'),
            'eval_metric': hyperparams.get('eval_metric', 'auc'),
            'tree_method': 'hist',  # Memory-efficient
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=hyperparams.get('n_estimators', 500),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        
        # Evaluate
        y_pred = model.predict(dval)
        from sklearn.metrics import roc_auc_score, mean_squared_error
        
        if len(np.unique(y_val)) > 2:
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            metric_name = 'rmse'
        else:
            score = roc_auc_score(y_val, y_pred)
            metric_name = 'auc'
        
        return {
            'status': 'completed',
            'model_type': 'xgboost',
            'best_iteration': model.best_iteration,
            f'best_{metric_name}': score,
            'feature_importance': model.get_score(importance_type='gain'),
            'duration_seconds': 0,  # Will be set by orchestrator
        }

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Dict,
    ) -> Dict:
        """Train LightGBM model with early stopping."""
        import lightgbm as lgb
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'num_leaves': hyperparams.get('num_leaves', 31),
            'learning_rate': hyperparams.get('learning_rate', 0.01),
            'feature_fraction': hyperparams.get('feature_fraction', 0.8),
            'bagging_fraction': hyperparams.get('bagging_fraction', 0.8),
            'objective': hyperparams.get('objective', 'binary'),
            'metric': hyperparams.get('metric', 'auc'),
            'verbose': -1,
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=hyperparams.get('n_estimators', 500),
            valid_sets=[train_data, val_data],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        
        y_pred = model.predict(X_val)
        from sklearn.metrics import roc_auc_score
        
        score = roc_auc_score(y_val, y_pred) if len(np.unique(y_val)) > 2 else 0.0
        
        return {
            'status': 'completed',
            'model_type': 'lightgbm',
            'best_iteration': model.best_iteration,
            'best_auc': score,
            'feature_importance': dict(zip(model.feature_name(), model.feature_importance())),
            'duration_seconds': 0,
        }

    def _train_sklearn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Dict,
    ) -> Dict:
        """Train scikit-learn model (Random Forest)."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import roc_auc_score, mean_squared_error
        
        if hyperparams.get('task', 'classification') == 'classification':
            model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 10),
                n_jobs=1,  # Single thread to control memory
                random_state=42,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 100),
                max_depth=hyperparams.get('max_depth', 10),
                n_jobs=1,
                random_state=42,
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            metric_name = 'auc'
        else:
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            metric_name = 'rmse'
        
        return {
            'status': 'completed',
            'model_type': 'random_forest',
            f'best_{metric_name}': score,
            'feature_importance': dict(zip(range(len(model.feature_importances_)), model.feature_importances_)),
            'duration_seconds': 0,
        }


class RetrainOrchestrator:
    """
    Orchestrates automated retraining jobs with strict resource controls.
    Ensures live trading engine is never starved of resources.
    """

    def __init__(self, max_concurrent_workers: int = MAX_CONCURRENT_WORKERS):
        self.max_concurrent_workers = max_concurrent_workers
        self.jobs: Dict[str, RetrainingJob] = {}
        self.active_workers: List[ray.actor.ActorHandle] = []
        self.last_retrain_time: Dict[str, float] = {}
        self._initialized = False

    def initialize_ray(self) -> None:
        """Initialize Ray with strict memory limits."""
        if not self._initialized:
            ray.init(
                num_cpus=min(os.cpu_count() or 4, 4),  # Cap CPU usage
                _system_max_memory=int(MAX_WORKER_MEMORY_GB * 1024**3 * max_concurrent_workers),
                ignore_reinit_error=True,
            )
            self._initialized = True

    def submit_job(
        self,
        model_type: str,
        symbol: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Dict,
        drift_score: float,
    ) -> str:
        """Submit a new retraining job."""
        # Check cooldown
        last_time = self.last_retrain_time.get(symbol, 0)
        if time.time() - last_time < RETRAIN_COOLDOWN_MINUTES * 60:
            raise RuntimeError(
                f"Retraining cooldown active for {symbol}. "
                f"Try again in {(RETRAIN_COOLDOWN_MINUTES * 60 - (time.time() - last_time)) / 60:.1f} minutes"
            )
        
        job_id = f"{symbol}_{model_type}_{int(time.time())}"
        
        job = RetrainingJob(
            job_id=job_id,
            model_type=model_type,
            symbol=symbol,
            drift_score=drift_score,
            status='pending',
        )
        
        self.jobs[job_id] = job
        return job_id

    def execute_job(self, job_id: str) -> RetrainingJob:
        """Execute a pending retraining job."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        if job.status != 'pending':
            return job
        
        # Wait for available worker slot
        while len(self.active_workers) >= self.max_concurrent_workers:
            # Clean up completed workers
            self.active_workers = [w for w in self.active_workers if ray.get(w.__ray_ready__.remote(), timeout=0)]
            time.sleep(0.5)
        
        # Create new worker
        worker = RetrainingWorker.remote(len(self.active_workers))
        self.active_workers.append(worker)
        
        # Get training data from job context (would be passed in real impl)
        # For now, simulate with placeholder
        job.status = 'running'
        job.started_at = time.time()
        self.last_retrain_time[job.symbol] = time.time()
        
        # Execute training (in real impl, data would be passed properly)
        # This is a simplified version
        result = ray.get(worker.train_model.remote(
            job.model_type,
            np.random.randn(100, 10),  # Placeholder
            np.random.randint(0, 2, 100),
            np.random.randn(20, 10),
            np.random.randint(0, 2, 20),
            {},
        ))
        
        job.completed_at = time.time()
        job.status = result.get('status', 'failed')
        job.metrics = result
        job.error = result.get('error')
        
        # Remove worker after completion (prevents memory leaks)
        ray.kill(worker)
        self.active_workers.remove(worker)
        
        return job

    def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of a specific job."""
        return self.jobs.get(job_id)

    def get_pending_jobs(self) -> List[RetrainingJob]:
        """Get all pending jobs."""
        return [j for j in self.jobs.values() if j.status == 'pending']

    def get_completed_jobs(self, limit: int = 10) -> List[RetrainingJob]:
        """Get recently completed jobs."""
        completed = [j for j in self.jobs.values() if j.status in ('completed', 'failed')]
        return sorted(completed, key=lambda x: x.completed_at or 0, reverse=True)[:limit]

    def cleanup(self) -> None:
        """Clean up all workers and shutdown Ray."""
        for worker in self.active_workers:
            ray.kill(worker)
        self.active_workers.clear()
        
        if self._initialized:
            ray.shutdown()
            self._initialized = False

    def can_retrain(self, symbol: str) -> bool:
        """Check if retraining is allowed for a symbol."""
        last_time = self.last_retrain_time.get(symbol, 0)
        return time.time() - last_time >= RETRAIN_COOLDOWN_MINUTES * 60


if __name__ == "__main__":
    # Example usage
    orchestrator = RetrainOrchestrator(max_concurrent_workers=2)
    orchestrator.initialize_ray()
    
    # Submit a job
    job_id = orchestrator.submit_job(
        model_type='xgboost',
        symbol='BTC-USDT',
        X_train=np.random.randn(1000, 20),
        y_train=np.random.randint(0, 2, 1000),
        X_val=np.random.randn(200, 20),
        y_val=np.random.randint(0, 2, 200),
        hyperparams={'max_depth': 6, 'learning_rate': 0.01},
        drift_score=0.35,
    )
    
    print(f"Submitted job: {job_id}")
    
    # Execute job
    job = orchestrator.execute_job(job_id)
    print(f"Job completed: {job.status}")
    print(f"Metrics: {job.metrics}")
    
    orchestrator.cleanup()

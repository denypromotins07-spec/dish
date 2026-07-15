"""
Automated rollback manager that instantly reverts to previous model version
via LMDB snapshots if new model's live performance drops below safety threshold.
"""

import lmdb
import pickle
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import json
from pathlib import Path

# Safety thresholds
MAX_DRAWDOWN_THRESHOLD = 0.05
MIN_HIT_RATE_THRESHOLD = 0.45
MAX_CONSECUTIVE_LOSSES = 10


@dataclass
class ModelSnapshot:
    """Snapshot of model state for rollback."""
    version: str
    timestamp_ns: int
    weights_path: str
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    is_active: bool = False


class RollbackManager:
    """
    Manages model versioning and automatic rollback based on
    real-time performance monitoring.
    """

    def __init__(
        self,
        db_path: str = './model_snapshots',
        max_snapshots: int = 10,
        map_size_mb: int = 1024,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.max_snapshots = max_snapshots
        self.max_drawdown_threshold = MAX_DRAWDOWN_THRESHOLD
        self.min_hit_rate_threshold = MIN_HIT_RATE_THRESHOLD
        self.max_consecutive_losses = MAX_CONSECUTIVE_LOSSES
        
        # Initialize LMDB
        self._env = lmdb.open(
            str(self.db_path),
            map_size=map_size_mb * 1024 * 1024,
            max_dbs=10,
        )
        
        self._metadata_db = self._env.open_db(b'metadata')
        self._weights_db = self._env.open_db(b'weights')
        self._performance_db = self._env.open_db(b'performance')
        
        self._current_version: Optional[str] = None
        self._rollback_count = 0
        self._lock = threading.RLock()
        
        # Performance tracking
        self._recent_pnls: List[float] = []
        self._consecutive_losses = 0
        self._peak_equity = 0.0
        self._current_equity = 0.0

    def save_snapshot(
        self,
        version: str,
        weights: np.ndarray,
        metadata: Dict[str, Any],
        initial_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Save a model snapshot for potential rollback."""
        with self._lock:
            try:
                timestamp_ns = int(datetime.now().timestamp() * 1e9)
                
                with self._env.begin(write=True) as txn:
                    # Save metadata
                    snapshot_data = {
                        'version': version,
                        'timestamp_ns': timestamp_ns,
                        'metadata': metadata,
                        'performance_metrics': initial_metrics or {},
                        'is_active': True,
                    }
                    txn.put(
                        f'meta:{version}'.encode(),
                        pickle.dumps(snapshot_data),
                        db=self._metadata_db,
                    )
                    
                    # Save weights (compressed)
                    weights_compressed = np.savez_compressed(None, weights).getbuffer()
                    txn.put(
                        f'weights:{version}'.encode(),
                        weights_compressed.tobytes(),
                        db=self._weights_db,
                    )
                
                # Deactivate other snapshots
                self._set_active_version(version)
                
                return True
                
            except Exception as e:
                print(f"Failed to save snapshot: {e}")
                return False

    def load_snapshot(self, version: str) -> Optional[Dict]:
        """Load a model snapshot by version."""
        with self._lock:
            try:
                with self._env.begin() as txn:
                    # Get metadata
                    meta_bytes = txn.get(f'meta:{version}'.encode(), db=self._metadata_db)
                    if meta_bytes is None:
                        return None
                    
                    metadata = pickle.loads(meta_bytes)
                    
                    # Get weights
                    weights_bytes = txn.get(f'weights:{version}'.encode(), db=self._weights_db)
                    if weights_bytes is None:
                        return None
                    
                    import io
                    weights = np.load(io.BytesIO(weights_bytes))['arr_0']
                    
                    return {
                        'version': version,
                        'weights': weights,
                        'metadata': metadata['metadata'],
                        'performance_metrics': metadata.get('performance_metrics', {}),
                    }
                    
            except Exception as e:
                print(f"Failed to load snapshot: {e}")
                return None

    def _set_active_version(self, version: str) -> None:
        """Mark a version as active and deactivate others."""
        with self._env.begin(write=True) as txn:
            # Get all versions
            cursor = txn.cursor(db=self._metadata_db)
            for key, value in cursor:
                if key.startswith(b'meta:'):
                    try:
                        data = pickle.loads(value)
                        data['is_active'] = (key.decode() == f'meta:{version}')
                        txn.put(key, pickle.dumps(data), db=self._metadata_db)
                    except Exception:
                        pass
            
            self._current_version = version

    def record_trade_result(self, pnl: float) -> Optional[str]:
        """
        Record a trade result and check if rollback is needed.
        Returns rollback version if triggered, None otherwise.
        """
        with self._lock:
            self._recent_pnls.append(pnl)
            self._current_equity += pnl
            
            # Track peak equity for drawdown calculation
            if self._current_equity > self._peak_equity:
                self._peak_equity = self._current_equity
            
            # Update consecutive losses
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0
            
            # Check rollback triggers
            if self._should_rollback():
                return self._execute_rollback()
            
            # Update performance metrics for current version
            if self._current_version:
                self._update_performance_metrics()
            
            return None

    def _should_rollback(self) -> bool:
        """Check if rollback conditions are met."""
        if len(self._recent_pnls) < 20:
            return False
        
        # Check consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            print(f"ROLLBACK TRIGGERED: {self._consecutive_losses} consecutive losses")
            return True
        
        # Check drawdown
        if self._peak_equity > 0:
            current_drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            if current_drawdown >= self.max_drawdown_threshold:
                print(f"ROLLBACK TRIGGERED: Drawdown {current_drawdown:.2%} >= threshold")
                return True
        
        # Check recent hit rate
        recent_trades = self._recent_pnls[-50:]
        if len(recent_trades) >= 20:
            hit_rate = sum(1 for p in recent_trades if p > 0) / len(recent_trades)
            if hit_rate < self.min_hit_rate_threshold:
                print(f"ROLLBACK TRIGGERED: Hit rate {hit_rate:.2%} < threshold")
                return True
        
        return False

    def _execute_rollback(self) -> Optional[str]:
        """Execute rollback to previous stable version."""
        with self._lock:
            # Find previous version
            previous_version = self._find_previous_version()
            
            if previous_version is None:
                print("No previous version available for rollback")
                return None
            
            print(f"Executing rollback from {self._current_version} to {previous_version}")
            
            # Update rollback count
            self._rollback_count += 1
            
            # Reset performance tracking
            self._reset_performance_tracking()
            
            return previous_version

    def _find_previous_version(self) -> Optional[str]:
        """Find the most recent stable version before current."""
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._metadata_db)
            
            versions = []
            for key, value in cursor:
                if key.startswith(b'meta:'):
                    try:
                        data = pickle.loads(value)
                        if not data.get('is_active', False):
                            versions.append({
                                'version': key.decode().replace('meta:', ''),
                                'timestamp': data.get('timestamp_ns', 0),
                                'metrics': data.get('performance_metrics', {}),
                            })
                    except Exception:
                        pass
            
            if not versions:
                return None
            
            # Sort by timestamp descending
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Return most recent non-current version
            for v in versions:
                if v['version'] != self._current_version:
                    return v['version']
            
            return None

    def _update_performance_metrics(self) -> None:
        """Update performance metrics for current version in database."""
        if not self._current_version:
            return
        
        metrics = self._calculate_current_metrics()
        
        with self._env.begin(write=True) as txn:
            try:
                meta_bytes = txn.get(f'meta:{self._current_version}'.encode(), db=self._metadata_db)
                if meta_bytes:
                    data = pickle.loads(meta_bytes)
                    data['performance_metrics'] = metrics
                    txn.put(f'meta:{self._current_version}'.encode(), pickle.dumps(data), db=self._metadata_db)
            except Exception:
                pass

    def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics."""
        if not self._recent_pnls:
            return {}
        
        recent = self._recent_pnls[-100:]
        winners = [p for p in recent if p > 0]
        
        return {
            'total_trades': len(recent),
            'winning_trades': len(winners),
            'hit_rate': len(winners) / len(recent) if recent else 0,
            'avg_pnl': np.mean(recent),
            'total_pnl': sum(recent),
            'sharpe': (np.mean(recent) / (np.std(recent) + 1e-8)) * np.sqrt(252 * 24),
            'max_drawdown': self._calculate_running_drawdown(recent),
            'consecutive_losses': self._consecutive_losses,
        }

    def _calculate_running_drawdown(self, pnls: List[float]) -> float:
        """Calculate max drawdown from PnL series."""
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / (np.maximum(running_max, 1e-8))
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _reset_performance_tracking(self) -> None:
        """Reset performance tracking after rollback."""
        self._recent_pnls = []
        self._consecutive_losses = 0
        self._peak_equity = 0.0
        self._current_equity = 0.0

    def get_available_versions(self) -> List[Dict]:
        """Get list of all available model versions."""
        versions = []
        
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._metadata_db)
            for key, value in cursor:
                if key.startswith(b'meta:'):
                    try:
                        data = pickle.loads(value)
                        versions.append({
                            'version': key.decode().replace('meta:', ''),
                            'timestamp_ns': data.get('timestamp_ns', 0),
                            'is_active': data.get('is_active', False),
                            'metrics': data.get('performance_metrics', {}),
                        })
                    except Exception:
                        pass
        
        return sorted(versions, key=lambda x: x['timestamp_ns'], reverse=True)

    def cleanup_old_snapshots(self, keep_last_n: int = 5) -> int:
        """Remove old snapshots to free space."""
        versions = self.get_available_versions()
        
        if len(versions) <= keep_last_n:
            return 0
        
        removed = 0
        with self._env.begin(write=True) as txn:
            for v in versions[keep_last_n:]:
                version = v['version']
                txn.delete(f'meta:{version}'.encode(), db=self._metadata_db)
                txn.delete(f'weights:{version}'.encode(), db=self._weights_db)
                txn.delete(f'performance:{version}'.encode(), db=self._performance_db)
                removed += 1
        
        return removed

    def get_rollback_stats(self) -> Dict:
        """Get rollback statistics."""
        return {
            'current_version': self._current_version,
            'rollback_count': self._rollback_count,
            'available_versions': len(self.get_available_versions()),
            'recent_pnls': len(self._recent_pnls),
            'consecutive_losses': self._consecutive_losses,
            'current_drawdown': (self._peak_equity - self._current_equity) / (self._peak_equity + 1e-8),
        }


if __name__ == "__main__":
    # Example usage
    manager = RollbackManager(db_path='./test_rollbacks')
    
    # Save initial model
    manager.save_snapshot(
        version='v1.0.0',
        weights=np.random.randn(100, 10),
        metadata={'model_type': 'xgboost', 'symbols': ['BTC-USDT']},
        initial_metrics={'sharpe': 2.5, 'hit_rate': 0.55},
    )
    
    # Save second model
    manager.save_snapshot(
        version='v2.0.0',
        weights=np.random.randn(100, 10),
        metadata={'model_type': 'lightgbm'},
    )
    
    # Simulate losing trades
    for i in range(15):
        pnl = -100 if i < 12 else 50  # 12 losses, then wins
        rollback_version = manager.record_trade_result(pnl)
        
        if rollback_version:
            print(f"ROLLBACK TRIGGERED! Reverting to: {rollback_version}")
            break
    
    # Get stats
    stats = manager.get_rollback_stats()
    print(f"\nRollback Stats: {json.dumps(stats, indent=2)}")
    
    # List versions
    versions = manager.get_available_versions()
    print(f"\nAvailable Versions:")
    for v in versions:
        print(f"  {v['version']} (active={v['is_active']})")

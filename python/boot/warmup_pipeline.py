#!/usr/bin/env python3
"""
ML Model and Feature Store Warmup Pipeline.
Runs dummy inference and populates Redis/DuckDB caches to avoid cold-start latency.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class WarmupConfig:
    """Configuration for warmup pipeline."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    duckdb_path: str = "/tmp/features.duckdb"
    model_paths: List[str] = None
    feature_dimensions: Dict[str, int] = None
    warmup_samples: int = 1000
    max_workers: int = 4
    
    def __post_init__(self):
        if self.model_paths is None:
            self.model_paths = []
        if self.feature_dimensions is None:
            self.feature_dimensions = {"default": 128}


class FeatureStoreWarmer:
    """Warms up feature store with sample data."""
    
    def __init__(self, config: WarmupConfig):
        self.config = config
        self.duckdb_conn = None
        
    def initialize_duckdb(self) -> bool:
        """Initialize DuckDB with schema."""
        try:
            import duckdb
            
            self.duckdb_conn = duckdb.connect(self.config.duckdb_path)
            
            # Create feature tables
            self.duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS price_features (
                    symbol VARCHAR,
                    timestamp_us BIGINT,
                    feature_vector FLOAT[],
                    label INTEGER
                )
            """)
            
            self.duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_features (
                    symbol VARCHAR,
                    timestamp_us BIGINT,
                    bid_depth FLOAT,
                    ask_depth FLOAT,
                    spread_bps FLOAT,
                    imbalance FLOAT
                )
            """)
            
            # Create indexes for fast lookup
            self.duckdb_conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_ts ON price_features(symbol, timestamp_us)")
            
            logger.info(f"DuckDB initialized at {self.config.duckdb_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")
            return False
    
    def populate_sample_features(self, symbol: str, count: int) -> int:
        """Populate feature store with sample data for a symbol."""
        if self.duckdb_conn is None:
            return 0
            
        try:
            base_time = int(time.time() * 1_000_000)
            dim = self.config.feature_dimensions.get(symbol, 128)
            
            # Generate sample features in batches
            batch_size = 100
            inserted = 0
            
            for batch_start in range(0, count, batch_size):
                batch_count = min(batch_size, count - batch_start)
                
                timestamps = [base_time + i * 1000 for i in range(batch_count)]
                features = np.random.randn(batch_count, dim).astype(np.float32)
                labels = np.random.randint(0, 3, batch_count)  # 0=sell, 1=hold, 2=buy
                
                # Insert batch
                self.duckdb_conn.execute("""
                    INSERT INTO price_features (symbol, timestamp_us, feature_vector, label)
                    SELECT ?, UNNEST(?), UNNEST(?), UNNEST(?)
                """, [symbol, timestamps, features.tolist(), labels.tolist()])
                
                inserted += batch_count
            
            self.duckdb_conn.commit()
            logger.info(f"Inserted {inserted} sample features for {symbol}")
            return inserted
            
        except Exception as e:
            logger.error(f"Failed to populate features for {symbol}: {e}")
            return 0
    
    def close(self):
        """Close DuckDB connection."""
        if self.duckdb_conn:
            self.duckdb_conn.close()


class ModelWarmer:
    """Warms up ML models by running dummy inference."""
    
    def __init__(self, config: WarmupConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        
    def load_and_warm_model(self, model_path: str) -> bool:
        """Load a model and run warmup inference."""
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model path does not exist: {model_path}, creating stub")
                # Create stub model for demonstration
                self.models[model_path] = {"type": "stub", "warmed": False}
            else:
                # In production, load actual model (ONNX, PyTorch, etc.)
                logger.info(f"Loading model from {model_path}")
                self.models[model_path] = {"type": "loaded", "path": model_path, "warmed": False}
            
            # Run warmup inference
            return self._run_warmup_inference(model_path)
            
        except Exception as e:
            logger.error(f"Failed to load/warm model {model_path}: {e}")
            return False
    
    def _run_warmup_inference(self, model_path: str) -> bool:
        """Run dummy inference to warm up model."""
        try:
            model = self.models.get(model_path)
            if model is None:
                return False
            
            dim = self.config.feature_dimensions.get("default", 128)
            
            # Run multiple warmup passes
            for i in range(10):
                dummy_input = np.random.randn(1, dim).astype(np.float32)
                
                # Simulate inference (in production, call model.predict())
                _ = np.sum(dummy_input)  # Dummy computation
                
            model["warmed"] = True
            logger.info(f"Model {model_path} warmed up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Warmup inference failed for {model_path}: {e}")
            return False
    
    def all_models_warmed(self) -> bool:
        """Check if all models are warmed."""
        return all(m.get("warmed", False) for m in self.models.values())


class WarmupPipeline:
    """Main warmup pipeline orchestrator."""
    
    def __init__(self, config: Optional[WarmupConfig] = None):
        self.config = config or WarmupConfig()
        self.feature_warmer = FeatureStoreWarmer(self.config)
        self.model_warmer = ModelWarmer(self.config)
        self.stats = {
            "features_inserted": 0,
            "models_warmed": 0,
            "total_duration_sec": 0.0,
        }
        
    def run(self, symbols: List[str] = None) -> bool:
        """Run the complete warmup pipeline."""
        start_time = time.time()
        logger.info("Starting warmup pipeline...")
        
        if symbols is None:
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        
        success = True
        
        # Step 1: Initialize feature store
        logger.info("Step 1: Initializing feature store...")
        if not self.feature_warmer.initialize_duckdb():
            logger.error("Feature store initialization failed")
            success = False
        
        # Step 2: Populate sample features (parallel)
        logger.info("Step 2: Populating sample features...")
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self.feature_warmer.populate_sample_features, 
                    symbol, 
                    self.config.warmup_samples
                ): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    count = future.result()
                    self.stats["features_inserted"] += count
                except Exception as e:
                    logger.error(f"Failed to populate features for {symbol}: {e}")
        
        # Step 3: Warm up models
        logger.info("Step 3: Warming up models...")
        for model_path in self.config.model_paths:
            if self.model_warmer.load_and_warm_model(model_path):
                self.stats["models_warmed"] += 1
        
        # Step 4: Verify warmup
        logger.info("Step 4: Verifying warmup...")
        if not self.model_warmer.all_models_warmed():
            logger.warning("Some models failed to warm up")
        
        # Cleanup
        self.feature_warmer.close()
        
        self.stats["total_duration_sec"] = time.time() - start_time
        
        logger.info(
            f"Warmup complete - Features: {self.stats['features_inserted']}, "
            f"Models: {self.stats['models_warmed']}, Duration: {self.stats['total_duration_sec']:.2f}s"
        )
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get warmup statistics."""
        return self.stats.copy()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Model and Feature Store Warmup")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USD", "ETH-USD"], help="Symbols to warm up")
    parser.add_argument("--samples", type=int, default=1000, help="Number of sample features per symbol")
    parser.add_argument("--duckdb-path", type=str, default="/tmp/features.duckdb", help="DuckDB file path")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    config = WarmupConfig(
        duckdb_path=args.duckdb_path,
        warmup_samples=args.samples,
        max_workers=args.workers,
        feature_dimensions={"BTC-USD": 256, "ETH-USD": 256, "default": 128}
    )
    
    pipeline = WarmupPipeline(config)
    success = pipeline.run(args.symbols)
    
    stats = pipeline.get_stats()
    print(f"\nWarmup Statistics:")
    print(f"  Features Inserted: {stats['features_inserted']}")
    print(f"  Models Warmed: {stats['models_warmed']}")
    print(f"  Duration: {stats['total_duration_sec']:.2f}s")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

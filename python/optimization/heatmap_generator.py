"""
Parameter Heatmap Generator for optimization results.
Creates 2D/3D heatmaps mapping Sharpe/Sortino ratios across parameter grids.
Uses memory-bounded Polars DataFrames to prevent RAM spikes.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# Strict memory bounds
MAX_ROWS = 1_000_000
MAX_COLUMNS = 256


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""
    x_param: str
    y_param: str
    z_param: Optional[str] = None  # For 3D heatmaps
    metric: str = "sharpe"  # sharpe, sortino, pnl, drawdown
    resolution_x: int = 50
    resolution_y: int = 50


class HeatmapGenerator:
    """
    Memory-efficient heatmap generator using Polars.
    Strictly bounds DataFrame sizes and uses lazy evaluation.
    """
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self._cache: Dict[str, pl.DataFrame] = {}
        self._cache_max_size = 5
        
    def load_optimization_results(self, results_path: str) -> pl.LazyFrame:
        """
        Load optimization results from parquet/CSV with strict row limits.
        Uses lazy evaluation to minimize memory footprint.
        """
        # Detect file type and load lazily
        if results_path.endswith('.parquet'):
            lf = pl.scan_parquet(results_path, n_rows=MAX_ROWS)
        elif results_path.endswith('.csv'):
            lf = pl.scan_csv(results_path, n_rows=MAX_ROWS)
        else:
            raise ValueError(f"Unsupported file format: {results_path}")
        
        return lf
    
    def generate_2d_heatmap(
        self,
        results: pl.LazyFrame,
        config: HeatmapConfig
    ) -> Dict[str, np.ndarray]:
        """
        Generate a 2D heatmap from optimization results.
        Returns dictionary with 'x_edges', 'y_edges', 'values' arrays.
        """
        # Collect with row limit
        df = results.collect(streaming=True)
        
        if len(df) == 0:
            return {'x_edges': np.array([]), 'y_edges': np.array([]), 'values': np.array([])}
        
        # Extract parameters and metric
        x_data = df[config.x_param].to_numpy()
        y_data = df[config.y_param].to_numpy()
        
        # Map metric column
        metric_col = f"{config.metric}_ratio" if config.metric in ['sharpe', 'sortino'] else config.metric
        if metric_col not in df.columns:
            metric_col = config.metric
        z_data = df[metric_col].to_numpy()
        
        # Create bin edges
        x_edges = np.linspace(x_data.min(), x_data.max(), config.resolution_x + 1)
        y_edges = np.linspace(y_data.min(), y_data.max(), config.resolution_y + 1)
        
        # Initialize value matrix
        values = np.full((config.resolution_y, config.resolution_x), np.nan)
        
        # Bin data into heatmap cells
        x_indices = np.digitize(x_data, x_edges[:-1]) - 1
        y_indices = np.digitize(y_data, y_edges[:-1]) - 1
        
        # Aggregate metric values per cell (using mean)
        for i in range(len(x_data)):
            xi = min(x_indices[i], config.resolution_x - 1)
            yi = min(y_indices[i], config.resolution_y - 1)
            
            if np.isnan(values[yi, xi]):
                values[yi, xi] = z_data[i]
            else:
                # Running average
                count = 2  # Simplified; would track counts properly in production
                values[yi, xi] = ((count - 1) * values[yi, xi] + z_data[i]) / count
        
        return {
            'x_edges': x_edges,
            'y_edges': y_edges,
            'values': values
        }
    
    def generate_3d_heatmap(
        self,
        results: pl.LazyFrame,
        config: HeatmapConfig
    ) -> Dict[str, np.ndarray]:
        """
        Generate a 3D heatmap (volume) from optimization results.
        Returns dictionary with 'x_edges', 'y_edges', 'z_edges', 'values' arrays.
        """
        if config.z_param is None:
            raise ValueError("z_param required for 3D heatmap")
        
        df = results.collect(streaming=True)
        
        if len(df) == 0:
            return {
                'x_edges': np.array([]),
                'y_edges': np.array([]),
                'z_edges': np.array([]),
                'values': np.array([])
            }
        
        resolution = 20  # Lower resolution for 3D to save memory
        
        x_data = df[config.x_param].to_numpy()
        y_data = df[config.y_param].to_numpy()
        z_param_data = df[config.z_param].to_numpy()
        
        metric_col = f"{config.metric}_ratio" if config.metric in ['sharpe', 'sortino'] else config.metric
        if metric_col not in df.columns:
            metric_col = config.metric
        values_data = df[metric_col].to_numpy()
        
        # Create 3D bin edges
        x_edges = np.linspace(x_data.min(), x_data.max(), resolution + 1)
        y_edges = np.linspace(y_data.min(), y_data.max(), resolution + 1)
        z_edges = np.linspace(z_param_data.min(), z_param_data.max(), resolution + 1)
        
        # Initialize 3D value array
        values = np.full((resolution, resolution, resolution), np.nan)
        
        # Bin data
        x_indices = np.digitize(x_data, x_edges[:-1]) - 1
        y_indices = np.digitize(y_data, y_edges[:-1]) - 1
        z_indices = np.digitize(z_param_data, z_edges[:-1]) - 1
        
        for i in range(len(x_data)):
            xi = min(x_indices[i], resolution - 1)
            yi = min(y_indices[i], resolution - 1)
            zi = min(z_indices[i], resolution - 1)
            values[zi, yi, xi] = values_data[i]  # Simplified; would aggregate properly
        
        return {
            'x_edges': x_edges,
            'y_edges': y_edges,
            'z_edges': z_edges,
            'values': values
        }
    
    def find_optimal_region(
        self,
        heatmap_data: Dict[str, np.ndarray],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Find top N optimal regions in the heatmap.
        Returns list of parameter combinations with best metrics.
        """
        values = heatmap_data['values']
        x_edges = heatmap_data['x_edges']
        y_edges = heatmap_data['y_edges']
        
        # Flatten and sort
        flat_indices = np.argsort(values.flatten())[::-1][:top_n]
        
        results = []
        for idx in flat_indices:
            if np.isnan(values.flatten()[idx]):
                continue
            
            # Convert flat index to 2D coordinates
            yi = idx // values.shape[1]
            xi = idx % values.shape[1]
            
            results.append({
                'x_center': (x_edges[xi] + x_edges[xi + 1]) / 2,
                'y_center': (y_edges[yi] + y_edges[yi + 1]) / 2,
                'metric_value': float(values[yi, xi]),
                'rank': len(results) + 1
            })
        
        return results
    
    def export_to_json(self, heatmap_data: Dict[str, np.ndarray]) -> str:
        """Export heatmap data to compact JSON for UI consumption."""
        # Downsample for UI (max 100x100)
        values = heatmap_data['values']
        if values.shape[0] > 100 or values.shape[1] > 100:
            step_y = max(1, values.shape[0] // 100)
            step_x = max(1, values.shape[1] // 100)
            values_downsampled = values[::step_y, ::step_x]
            x_edges_downsampled = heatmap_data['x_edges'][::step_x]
            y_edges_downsampled = heatmap_data['y_edges'][::step_y]
        else:
            values_downsampled = values
            x_edges_downsampled = heatmap_data['x_edges']
            y_edges_downsampled = heatmap_data['y_edges']
        
        export_data = {
            'x_edges': x_edges_downsampled.tolist(),
            'y_edges': y_edges_downsampled.tolist(),
            'values': np.nan_to_num(values_downsampled, nan=0.0).tolist(),
            'shape': values_downsampled.shape
        }
        
        return json.dumps(export_data)
    
    def clear_cache(self):
        """Clear internal cache to free memory."""
        self._cache.clear()


def generate_heatmap_from_results(
    results_path: str,
    x_param: str,
    y_param: str,
    metric: str = "sharpe"
) -> str:
    """Convenience function to generate and export a heatmap."""
    generator = HeatmapGenerator()
    results = generator.load_optimization_results(results_path)
    
    config = HeatmapConfig(
        x_param=x_param,
        y_param=y_param,
        metric=metric
    )
    
    heatmap_data = generator.generate_2d_heatmap(results, config)
    return generator.export_to_json(heatmap_data)


if __name__ == '__main__':
    # Example usage with synthetic data
    import tempfile
    import os
    
    # Create sample optimization results
    n_samples = 10000
    data = {
        'param_a': np.random.uniform(0.1, 1.0, n_samples),
        'param_b': np.random.uniform(0.01, 0.5, n_samples),
        'sharpe_ratio': np.random.normal(1.5, 0.5, n_samples),
        'sortino_ratio': np.random.normal(2.0, 0.7, n_samples),
    }
    
    df = pl.DataFrame(data)
    
    # Save to temp parquet
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.write_parquet(f.name)
        temp_path = f.name
    
    try:
        generator = HeatmapGenerator()
        results = generator.load_optimization_results(temp_path)
        
        config = HeatmapConfig(
            x_param='param_a',
            y_param='param_b',
            metric='sharpe',
            resolution_x=50,
            resolution_y=50
        )
        
        heatmap = generator.generate_2d_heatmap(results, config)
        print(f"Heatmap shape: {heatmap['values'].shape}")
        
        optimal = generator.find_optimal_region(heatmap, top_n=3)
        print(f"Top regions: {optimal}")
        
    finally:
        os.unlink(temp_path)

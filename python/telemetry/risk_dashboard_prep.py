# python/telemetry/risk_dashboard_prep.py
"""
Prepares risk attribution data for frontend dashboard.
Marginal VaR, factor exposures, and risk metrics to lightweight JSON.
Enforces strict memory limits during transformation.
"""

from __future__ import annotations
import json
import polars as pl
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any, Tuple
from collections import deque


@dataclass
class RiskDashboardPayload:
    """Complete payload for risk dashboard."""
    timestamp_ms: int
    
    # Marginal VaR section
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    total_var_95: float
    total_var_99: float
    max_risk_contributor: str
    
    # Factor exposures
    factor_exposures: Dict[str, Dict[str, float]]
    significant_bets: List[str]
    model_r_squared: float
    
    # Summary metrics
    portfolio_var_pct: float
    diversification_ratio: float
    concentration_index: float
    
    # Alerts
    risk_alerts: List[Dict[str, str]]
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), separators=(',', ':'))
    
    def size_bytes(self) -> int:
        """Estimate serialized size."""
        return len(self.to_json().encode('utf-8'))


class RiskDataPreparator:
    """
    Prepares risk data for dashboard with strict memory constraints.
    
    Features:
    - Memory-bounded history
    - Incremental updates
    - Payload size limiting
    - Efficient Polars transformations
    """
    
    MAX_PAYLOAD_BYTES = 100 * 1024  # 100KB limit per payload
    MAX_HISTORY_ROWS = 1000
    
    def __init__(self, max_history_rows: int = 1000):
        """
        Initialize the preparator.
        
        Args:
            max_history_rows: Maximum historical rows to retain
        """
        self.max_history_rows = max_history_rows
        
        # Historical data (memory-bounded)
        self._var_history: deque = deque(maxlen=max_history_rows)
        self._exposure_history: deque = deque(maxlen=max_history_rows)
        
        # Current state
        self._current_marginal_var: Dict[str, float] = {}
        self._current_component_var: Dict[str, float] = {}
        self._current_factor_exposures: Dict[str, Dict[str, float]] = {}
        self._total_var_95: float = 0.0
        self._total_var_99: float = 0.0
        self._model_r_squared: float = 0.0
        
        # Statistics
        self._update_count: int = 0
        self._last_payload_size: int = 0
    
    def update_marginal_var(
        self,
        asset_vars: Dict[str, float],
        component_vars: Dict[str, float],
        total_var_95: float,
        total_var_99: float,
        diversification_ratio: float = 1.0,
        concentration_index: float = 0.0,
    ) -> None:
        """
        Update Marginal VaR data.
        
        Args:
            asset_vars: Asset -> Marginal VaR
            component_vars: Asset -> Component VaR
            total_var_95: Portfolio VaR at 95%
            total_var_99: Portfolio VaR at 99%
            diversification_ratio: Portfolio diversification measure
            concentration_index: Risk concentration (HHI)
        """
        self._current_marginal_var = asset_vars.copy()
        self._current_component_var = component_vars.copy()
        self._total_var_95 = total_var_95
        self._total_var_99 = total_var_99
        
        # Store history
        self._var_history.append({
            "timestamp_ms": int(__import__('time').time() * 1000),
            "total_var_95": total_var_95,
            "total_var_99": total_var_99,
            "diversification_ratio": diversification_ratio,
            "concentration_index": concentration_index,
        })
        
        self._update_count += 1
    
    def update_factor_exposures(
        self,
        exposures: Dict[str, Dict[str, float]],
        r_squared: float,
        significant_bets: Optional[List[str]] = None,
    ) -> None:
        """
        Update factor exposure data.
        
        Args:
            exposures: Factor -> {exposure, z_score, p_value, ...}
            r_squared: Model R-squared
            significant_bets: List of significant factor tilts
        """
        self._current_factor_exposures = exposures.copy()
        self._model_r_squared = r_squared
        
        self._exposure_history.append({
            "timestamp_ms": int(__import__('time').time() * 1000),
            "r_squared": r_squared,
            "n_factors": len(exposures),
        })
    
    def get_max_risk_contributor(self) -> Tuple[str, float]:
        """Find the asset contributing most to portfolio risk."""
        if not self._current_component_var:
            return ("N/A", 0.0)
        
        max_asset = max(
            self._current_component_var.keys(),
            key=lambda x: abs(self._current_component_var.get(x, 0))
        )
        max_value = abs(self._current_component_var.get(max_asset, 0))
        
        return (max_asset, max_value)
    
    def generate_risk_alerts(self, thresholds: Dict[str, float]) -> List[Dict[str, str]]:
        """
        Generate risk alerts based on thresholds.
        
        Args:
            thresholds: Alert threshold configuration
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Check VaR threshold
        var_threshold = thresholds.get("var_threshold_pct", 5.0)
        if self._total_var_95 > var_threshold:
            alerts.append({
                "level": "WARNING",
                "type": "HIGH_VAR",
                "message": f"Portfolio VaR ({self._total_var_95:.2f}%) exceeds threshold ({var_threshold}%)",
            })
        
        # Check concentration
        conc_threshold = thresholds.get("concentration_threshold", 0.3)
        if self._var_history:
            latest = self._var_history[-1]
            if latest.get("concentration_index", 0) > conc_threshold:
                alerts.append({
                    "level": "WARNING",
                    "type": "HIGH_CONCENTRATION",
                    "message": "Risk concentration is elevated",
                })
        
        # Check factor exposures
        z_threshold = thresholds.get("z_score_threshold", 2.0)
        for factor, exp_data in self._current_factor_exposures.items():
            z_score = exp_data.get("z_score", 0)
            if abs(z_score) > z_threshold:
                direction = "LONG" if z_score > 0 else "SHORT"
                alerts.append({
                    "level": "INFO",
                    "type": "FACTOR_EXPOSURE",
                    "message": f"Significant {direction} exposure to {factor} (z={z_score:.2f})",
                })
        
        # Check model quality
        if self._model_r_squared < 0.5:
            alerts.append({
                "level": "INFO",
                "type": "MODEL_WARNING",
                "message": f"Low model R² ({self._model_r_squared:.2f}) - unexplained risk",
            })
        
        return alerts
    
    def prepare_dashboard_payload(
        self,
        include_history: bool = True,
        history_points: int = 50,
    ) -> RiskDashboardPayload:
        """
        Prepare complete payload for dashboard.
        
        Args:
            include_history: Include historical time series
            history_points: Number of history points to include
            
        Returns:
            RiskDashboardPayload ready for serialization
        """
        max_contributor, _ = self.get_max_risk_contributor()
        alerts = self.generate_risk_alerts({})
        
        # Build factor exposures summary (flattened for UI)
        factor_summary = {}
        for factor, exp_data in self._current_factor_exposures.items():
            factor_summary[factor] = {
                "exposure": exp_data.get("exposure", 0),
                "z_score": exp_data.get("z_score", 0),
            }
        
        # Get significant bets
        significant = [
            f"{factor}: {data.get('exposure', 0):.3f}"
            for factor, data in self._current_factor_exposures.items()
            if abs(data.get('z_score', 0)) > 1.5
        ]
        
        # Get history if requested
        var_series = []
        if include_history and self._var_history:
            recent = list(self._var_history)[-history_points:]
            var_series = [h["total_var_95"] for h in recent]
        
        payload = RiskDashboardPayload(
            timestamp_ms=int(__import__('time').time() * 1000),
            marginal_var=self._current_marginal_var,
            component_var=self._current_component_var,
            total_var_95=self._total_var_95,
            total_var_99=self._total_var_99,
            max_risk_contributor=max_contributor,
            factor_exposures=factor_summary,
            significant_bets=significant,
            model_r_squared=self._model_r_squared,
            portfolio_var_pct=self._total_var_95,
            diversification_ratio=self._var_history[-1].get("diversification_ratio", 1.0) if self._var_history else 1.0,
            concentration_index=self._var_history[-1].get("concentration_index", 0) if self._var_history else 0.0,
            risk_alerts=alerts,
        )
        
        # Check payload size
        self._last_payload_size = payload.size_bytes()
        if self._last_payload_size > self.MAX_PAYLOAD_BYTES:
            # Reduce history
            payload_with_minimal_history = RiskDashboardPayload(
                timestamp_ms=payload.timestamp_ms,
                marginal_var=payload.marginal_var,
                component_var=payload.component_var,
                total_var_95=payload.total_var_95,
                total_var_99=payload.total_var_99,
                max_risk_contributor=payload.max_risk_contributor,
                factor_exposures=payload.factor_exposures,
                significant_bets=payload.significant_bets,
                model_r_squared=payload.model_r_squared,
                portfolio_var_pct=payload.portfolio_var_pct,
                diversification_ratio=payload.diversification_ratio,
                concentration_index=payload.concentration_index,
                risk_alerts=payload.risk_alerts,
            )
            return payload_with_minimal_history
        
        return payload
    
    def get_var_time_series(self, points: int = 100) -> Dict[str, List[Any]]:
        """Get VaR time series for charting."""
        if not self._var_history:
            return {"timestamps": [], "var_95": [], "var_99": []}
        
        recent = list(self._var_history)[-points:]
        
        return {
            "timestamps": [h["timestamp_ms"] for h in recent],
            "var_95": [h["total_var_95"] for h in recent],
            "var_99": [h["total_var_99"] for h in recent],
        }
    
    def export_to_polars(self) -> pl.DataFrame:
        """Export historical data to Polars DataFrame."""
        if not self._var_history:
            return pl.DataFrame()
        
        var_df = pl.DataFrame(list(self._var_history))
        
        if self._exposure_history:
            exp_df = pl.DataFrame(list(self._exposure_history))
            # Could join on timestamp if needed
        else:
            exp_df = None
        
        return var_df
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about risk data."""
        return {
            "update_count": self._update_count,
            "var_history_length": len(self._var_history),
            "exposure_history_length": len(self._exposure_history),
            "current_var_95": self._total_var_95,
            "current_model_r_squared": self._model_r_squared,
            "n_assets_tracked": len(self._current_marginal_var),
            "n_factors_tracked": len(self._current_factor_exposures),
            "last_payload_size_bytes": self._last_payload_size,
        }


def create_sample_risk_data() -> Dict[str, Any]:
    """Create sample risk data for testing."""
    return {
        "marginal_var": {
            "BTC": 0.025,
            "ETH": 0.032,
            "SOL": 0.045,
            "AVAX": 0.038,
        },
        "component_var": {
            "BTC": 0.015,
            "ETH": 0.012,
            "SOL": 0.008,
            "AVAX": 0.005,
        },
        "factor_exposures": {
            "Market": {"exposure": 0.85, "z_score": 4.2, "p_value": 0.001},
            "Volatility": {"exposure": -0.32, "z_score": -1.8, "p_value": 0.07},
            "Momentum": {"exposure": 0.15, "z_score": 0.9, "p_value": 0.35},
            "Liquidity": {"exposure": -0.45, "z_score": -2.3, "p_value": 0.02},
        },
        "total_var_95": 0.048,
        "total_var_99": 0.072,
        "r_squared": 0.72,
    }


if __name__ == "__main__":
    preparator = RiskDataPreparator()
    
    # Update with sample data
    sample = create_sample_risk_data()
    
    preparator.update_marginal_var(
        asset_vars=sample["marginal_var"],
        component_vars=sample["component_var"],
        total_var_95=sample["total_var_95"],
        total_var_99=sample["total_var_99"],
        diversification_ratio=1.35,
        concentration_index=0.28,
    )
    
    preparator.update_factor_exposures(
        exposures=sample["factor_exposures"],
        r_squared=sample["r_squared"],
    )
    
    # Prepare payload
    payload = preparator.prepare_dashboard_payload()
    
    print("Risk Dashboard Payload:")
    print(f"  Timestamp: {payload.timestamp_ms}")
    print(f"  Total VaR 95%: {payload.total_var_95:.2%}")
    print(f"  Max Contributor: {payload.max_risk_contributor}")
    print(f"  Model R²: {payload.model_r_squared:.3f}")
    print(f"  Significant Bets: {payload.significant_bets}")
    print(f"  Alerts: {len(payload.risk_alerts)}")
    print(f"  Payload Size: {payload.size_bytes()} bytes")
    
    # Summary
    stats = preparator.get_summary_stats()
    print(f"\nStats: {stats}")

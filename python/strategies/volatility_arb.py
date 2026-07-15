"""
Volatility Surface Arbitrage Executor
Automatically identifies and executes calendar spreads, risk reversals, and butterfly spreads
when IV surface becomes mispriced relative to historical realized volatility.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    EXECUTING = "executing"
    HEDGING = "hedging"
    CLOSING = "closing"


@dataclass
class VolatilitySignal:
    """Signal from IV surface analysis"""
    strategy_type: str  # calendar, risk_reversal, butterfly
    strike: float
    expiry1: int  # days
    expiry2: Optional[int]
    iv_diff: float  # basis points
    expected_edge: float
    confidence: float
    recommended_size: float


@dataclass
class OptionsLeg:
    """Single leg of an options strategy"""
    option_type: str  # call or put
    strike: float
    expiry: int
    side: str  # buy or sell
    quantity: int
    iv: float


class VolatilityArbitrageStrategy:
    """
    Volatility surface arbitrage executor.
    
    Identifies mispricings in the IV surface and executes:
    - Calendar spreads (term structure arb)
    - Risk reversals (skew arb)
    - Butterfly spreads (curvature arb)
    """

    def __init__(
        self,
        realized_vol_window: int = 30,
        min_edge_bps: float = 50.0,
        max_position_notional: float = 1_000_000.0,
        kelly_fraction: float = 0.25,
    ):
        self.realized_vol_window = realized_vol_window
        self.min_edge_bps = min_edge_bps
        self.max_position_notional = max_position_notional
        self.kelly_fraction = kelly_fraction
        
        self.state = StrategyState.IDLE
        self._historical_iv: Dict[int, np.ndarray] = {}
        self._realized_vols: Dict[str, float] = {}
        self._active_positions: List[Dict] = []
        
    def update_realized_vol(self, symbol: str, price_series: np.ndarray) -> None:
        """Update realized volatility from price series"""
        if len(price_series) < 2:
            return
            
        returns = np.diff(np.log(price_series))
        vol = np.std(returns) * np.sqrt(365) * 100  # Annualized %
        self._realized_vols[symbol] = vol
        
    def update_iv_surface(self, iv_data: Dict[Tuple[float, int], float]) -> None:
        """
        Update IV surface data.
        iv_data: {(strike, expiry_days): implied_vol}
        """
        for (strike, expiry), iv in iv_data.items():
            if expiry not in self._historical_iv:
                self._historical_iv[expiry] = []
            self._historical_iv[expiry].append(iv)
            
            # Keep bounded history
            if len(self._historical_iv[expiry]) > self.realized_vol_window:
                self._historical_iv[expiry] = self._historical_iv[expiry][-self.realized_vol_window:]
    
    def scan_for_opportunities(
        self,
        current_iv_surface: Dict[Tuple[float, int], float],
        spot_price: float,
    ) -> List[VolatilitySignal]:
        """Scan IV surface for arbitrage opportunities"""
        self.state = StrategyState.SCANNING
        signals = []
        
        # 1. Calendar Spread Opportunities (term structure)
        calendar_signals = self._scan_calendar_spreads(current_iv_surface, spot_price)
        signals.extend(calendar_signals)
        
        # 2. Risk Reversal Opportunities (skew)
        rr_signals = self._scan_risk_reversals(current_iv_surface, spot_price)
        signals.extend(rr_signals)
        
        # 3. Butterfly Opportunities (curvature)
        butterfly_signals = self._scan_butterflies(current_iv_surface, spot_price)
        signals.extend(butterfly_signals)
        
        # Filter by minimum edge
        filtered = [s for s in signals if abs(s.expected_edge) >= self.min_edge_bps]
        
        logger.info(f"Found {len(filtered)} vol arb opportunities")
        self.state = StrategyState.IDLE
        return filtered
    
    def _scan_calendar_spreads(
        self,
        iv_surface: Dict[Tuple[float, int], float],
        spot_price: float,
    ) -> List[VolatilitySignal]:
        """Find calendar spread opportunities"""
        signals = []
        
        # Group by strike
        by_strike: Dict[float, Dict[int, float]] = {}
        for (strike, expiry), iv in iv_surface.items():
            if strike not in by_strike:
                by_strike[strike] = {}
            by_strike[strike][expiry] = iv
        
        # Compare near-term vs long-term IV
        expiries = sorted(set(exp for d in by_strike.values() for exp in d.keys()))
        if len(expiries) < 2:
            return signals
        
        near_exp = expiries[0]
        far_exp = expiries[-1]
        
        for strike, exp_map in by_strike.items():
            if near_exp not in exp_map or far_exp not in exp_map:
                continue
                
            near_iv = exp_map[near_exp]
            far_iv = exp_map[far_exp]
            
            # Normal contango: far IV > near IV
            # Inversion opportunity: near IV >> far IV
            iv_diff = (near_iv - far_iv) * 10000  # bps
            
            # Calculate expected edge based on mean reversion
            hist_diff = self._get_historical_term_structure_diff(near_exp, far_exp)
            z_score = (iv_diff - hist_diff['mean']) / max(hist_diff['std'], 1.0)
            
            if abs(z_score) > 2.0:  # 2 std dev event
                edge = abs(z_score) * 10  # Simplified edge calculation
                size = self._calculate_position_size(edge, abs(z_score))
                
                signals.append(VolatilitySignal(
                    strategy_type="calendar",
                    strike=strike,
                    expiry1=near_exp,
                    expiry2=far_exp,
                    iv_diff=iv_diff,
                    expected_edge=edge,
                    confidence=min(abs(z_score) / 3.0, 1.0),
                    recommended_size=size,
                ))
        
        return signals
    
    def _scan_risk_reversals(
        self,
        iv_surface: Dict[Tuple[float, int], float],
        spot_price: float,
    ) -> List[VolatilitySignal]:
        """Find risk reversal (skew) opportunities"""
        signals = []
        
        # Find ATM strike
        atm_strike = spot_price
        
        # Group by expiry
        by_expiry: Dict[int, Dict[float, float]] = {}
        for (strike, expiry), iv in iv_surface.items():
            if expiry not in by_expiry:
                by_expiry[expiry] = {}
            by_expiry[expiry][strike] = iv
        
        for expiry, strikes in by_expiry.items():
            # Find 25-delta calls and puts (approximated by moneyness)
            call_strike = atm_strike * 1.05  # Approx 25D call
            put_strike = atm_strike * 0.95   # Approx 25D put
            
            call_iv = self._interpolate_iv(strikes, call_strike)
            put_iv = self._interpolate_iv(strikes, put_strike)
            atm_iv = self._interpolate_iv(strikes, atm_strike)
            
            if call_iv is None or put_iv is None or atm_iv is None:
                continue
            
            # Risk reversal = Call IV - Put IV
            rr = (call_iv - put_iv) * 10000  # bps
            
            # Compare to historical skew
            hist_rr = self._get_historical_skew(expiry)
            z_score = (rr - hist_rr['mean']) / max(hist_rr['std'], 1.0)
            
            if abs(z_score) > 2.0:
                edge = abs(z_score) * 10
                size = self._calculate_position_size(edge, abs(z_score))
                
                signals.append(VolatilitySignal(
                    strategy_type="risk_reversal",
                    strike=atm_strike,
                    expiry1=expiry,
                    expiry2=None,
                    iv_diff=rr,
                    expected_edge=edge,
                    confidence=min(abs(z_score) / 3.0, 1.0),
                    recommended_size=size,
                ))
        
        return signals
    
    def _scan_butterflies(
        self,
        iv_surface: Dict[Tuple[float, int], float],
        spot_price: float,
    ) -> List[VolatilitySignal]:
        """Find butterfly spread opportunities (curvature)"""
        signals = []
        
        by_expiry: Dict[int, Dict[float, float]] = {}
        for (strike, expiry), iv in iv_surface.items():
            if expiry not in by_expiry:
                by_expiry[expiry] = {}
            by_expiry[expiry][strike] = iv
        
        for expiry, strikes in by_expiry.items():
            sorted_strikes = sorted(strikes.keys())
            if len(sorted_strikes) < 3:
                continue
            
            # ATM
            atm_strike = min(sorted_strikes, key=lambda x: abs(x - spot_price))
            atm_iv = strikes[atm_strike]
            
            # Wings (OTM call and put)
            otm_call_strike = min([s for s in sorted_strikes if s > atm_strike], default=None)
            otm_put_strike = max([s for s in sorted_strikes if s < atm_strike], default=None)
            
            if otm_call_strike is None or otm_put_strike is None:
                continue
            
            call_iv = strikes[otm_call_strike]
            put_iv = strikes[otm_put_strike]
            
            # Butterfly value = (Call IV + Put IV) / 2 - ATM IV
            butterfly = ((call_iv + put_iv) / 2 - atm_iv) * 10000  # bps
            
            # Compare to historical butterfly
            hist_bfly = self._get_historical_butterfly(expiry)
            z_score = (butterfly - hist_bfly['mean']) / max(hist_bfly['std'], 1.0)
            
            if abs(z_score) > 2.0:
                edge = abs(z_score) * 10
                size = self._calculate_position_size(edge, abs(z_score))
                
                signals.append(VolatilitySignal(
                    strategy_type="butterfly",
                    strike=atm_strike,
                    expiry1=expiry,
                    expiry2=None,
                    iv_diff=butterfly,
                    expected_edge=edge,
                    confidence=min(abs(z_score) / 3.0, 1.0),
                    recommended_size=size,
                ))
        
        return signals
    
    def _get_historical_term_structure_diff(self, near_exp: int, far_exp: int) -> Dict:
        """Get historical mean/std for term structure difference"""
        if near_exp not in self._historical_iv or far_exp not in self._historical_iv:
            return {'mean': 0.0, 'std': 100.0}
        
        near_hist = np.array(self._historical_iv[near_exp])
        far_hist = np.array(self._historical_iv[far_exp])
        
        if len(near_hist) != len(far_hist):
            min_len = min(len(near_hist), len(far_hist))
            near_hist = near_hist[:min_len]
            far_hist = far_hist[:min_len]
        
        diff = (near_hist - far_hist) * 10000
        return {
            'mean': np.mean(diff),
            'std': np.std(diff),
        }
    
    def _get_historical_skew(self, expiry: int) -> Dict:
        """Get historical skew metrics"""
        return {'mean': 0.0, 'std': 50.0}  # Placeholder
    
    def _get_historical_butterfly(self, expiry: int) -> Dict:
        """Get historical butterfly metrics"""
        return {'mean': 0.0, 'std': 30.0}  # Placeholder
    
    def _interpolate_iv(self, strikes: Dict[float, float], target_strike: float) -> Optional[float]:
        """Linear interpolation of IV"""
        sorted_strikes = sorted(strikes.keys())
        
        if target_strike <= sorted_strikes[0]:
            return strikes[sorted_strikes[0]]
        if target_strike >= sorted_strikes[-1]:
            return strikes[sorted_strikes[-1]]
        
        # Find bracketing strikes
        for i in range(len(sorted_strikes) - 1):
            lower = sorted_strikes[i]
            upper = sorted_strikes[i + 1]
            if lower <= target_strike <= upper:
                t = (target_strike - lower) / (upper - lower)
                return strikes[lower] * (1 - t) + strikes[upper] * t
        
        return None
    
    def _calculate_position_size(self, edge: float, z_score: float) -> float:
        """Calculate position size using Kelly Criterion"""
        # Simplified Kelly: f = (p * b - q) / b
        win_prob = 0.5 + min(z_score / 10.0, 0.3)  # Map z-score to win probability
        payoff_ratio = edge / 10.0
        
        if payoff_ratio <= 0:
            return 0.0
        
        kelly = (win_prob * payoff_ratio - (1 - win_prob)) / payoff_ratio
        kelly = max(0, min(kelly, self.kelly_fraction))
        
        return kelly * self.max_position_notional
    
    def generate_order_legs(self, signal: VolatilitySignal) -> List[OptionsLeg]:
        """Generate options legs for a signal"""
        legs = []
        
        if signal.strategy_type == "calendar":
            # Buy far IV, sell near IV (if near is expensive)
            if signal.iv_diff > 0:
                legs.append(OptionsLeg("call", signal.strike, signal.expiry2, "buy", 1, 0.0))
                legs.append(OptionsLeg("call", signal.strike, signal.expiry1, "sell", 1, 0.0))
            else:
                legs.append(OptionsLeg("call", signal.strike, signal.expiry2, "sell", 1, 0.0))
                legs.append(OptionsLeg("call", signal.strike, signal.expiry1, "buy", 1, 0.0))
        
        elif signal.strategy_type == "risk_reversal":
            if signal.iv_diff > 0:  # Calls expensive
                legs.append(OptionsLeg("call", signal.strike * 1.05, signal.expiry1, "sell", 1, 0.0))
                legs.append(OptionsLeg("put", signal.strike * 0.95, signal.expiry1, "buy", 1, 0.0))
            else:  # Puts expensive
                legs.append(OptionsLeg("call", signal.strike * 1.05, signal.expiry1, "buy", 1, 0.0))
                legs.append(OptionsLeg("put", signal.strike * 0.95, signal.expiry1, "sell", 1, 0.0))
        
        elif signal.strategy_type == "butterfly":
            # Long butterfly: buy wings, sell body
            wings_qty = 1
            body_qty = 2
            legs.append(OptionsLeg("call", signal.strike * 0.95, signal.expiry1, "buy", wings_qty, 0.0))
            legs.append(OptionsLeg("call", signal.strike, signal.expiry1, "sell", body_qty, 0.0))
            legs.append(OptionsLeg("call", signal.strike * 1.05, signal.expiry1, "buy", wings_qty, 0.0))
        
        return legs

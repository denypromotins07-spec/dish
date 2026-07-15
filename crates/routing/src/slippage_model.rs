//! Real-time slippage and Transaction Cost Analysis (TCA) modeling.
//! Predictions based on live order book depth, bid-ask spread, tick size, and trade aggressor data.

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Slippage prediction result
#[derive(Debug, Clone)]
pub struct SlippagePrediction {
    /// Expected slippage in basis points
    pub expected_slippage_bps: f64,
    /// Market impact component (bps)
    pub market_impact_bps: f64,
    /// Spread cost component (bps)
    pub spread_cost_bps: f64,
    /// Timing cost component (bps)
    pub timing_cost_bps: f64,
    /// Confidence interval (95%)
    pub confidence_interval_bps: f64,
    /// Recommended execution strategy
    pub recommended_urgency: f64,
}

/// TCA (Transaction Cost Analysis) report
#[derive(Debug, Clone)]
pub struct TcaReport {
    pub total_slippage_bps: f64,
    pub benchmark_price: f64,
    pub average_fill_price: f64,
    pub total_quantity: f64,
    pub total_notional: f64,
    pub market_impact_bps: f64,
    pub timing_cost_bps: f64,
    pub spread_cost_bps: f64,
    pub fee_cost_bps: f64,
    pub total_cost_bps: f64,
    pub performance_vs_arrival: f64,
    pub performance_vs_vwap: f64,
    pub execution_quality_score: f64,
}

/// Order book state for slippage calculation
#[derive(Debug, Clone)]
pub struct OrderBookState {
    pub bids: Vec<(f64, f64)>, // (price, quantity)
    pub asks: Vec<(f64, f64)>,
    pub spread_bps: f64,
    pub mid_price: f64,
    pub tick_size: f64,
}

/// Slippage model main engine
pub struct SlippageModel {
    /// Recent trade history for calibration
    trade_history: Vec<TradeRecord>,
    /// Model parameters (calibrated)
    impact_coefficient: AtomicF64,
    /// Volatility estimate
    current_volatility: AtomicF64,
    /// Average spread
    avg_spread_bps: AtomicF64,
    /// Maximum trades to keep in history
    max_history_size: usize,
}

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp_ns: u64,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub was_aggressor: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl SlippageModel {
    /// Create new slippage model
    pub fn new(max_history: usize) -> Self {
        Self {
            trade_history: Vec::with_capacity(max_history),
            impact_coefficient: AtomicF64::new(0.1), // Default impact coefficient
            current_volatility: AtomicF64::new(0.01), // 1% daily vol
            avg_spread_bps: AtomicF64::new(5.0),
            max_history_size: max_history,
        }
    }

    /// Record a trade for model calibration
    pub fn record_trade(&mut self, record: TradeRecord) {
        self.trade_history.push(record);
        
        if self.trade_history.len() > self.max_history_size {
            self.trade_history.remove(0);
        }
    }

    /// Predict slippage for a given order
    pub fn predict_slippage(
        &self,
        order_book: &OrderBookState,
        quantity: f64,
        side: TradeSide,
        urgency: f64, // 0.0 (passive) to 1.0 (aggressive)
    ) -> SlippagePrediction {
        let mid_price = order_book.mid_price;
        if mid_price <= 0.0 || quantity <= 0.0 {
            return SlippagePrediction {
                expected_slippage_bps: 0.0,
                market_impact_bps: 0.0,
                spread_cost_bps: 0.0,
                timing_cost_bps: 0.0,
                confidence_interval_bps: 0.0,
                recommended_urgency: 0.5,
            };
        }
        
        let notional = quantity * mid_price;
        
        // Component 1: Spread cost (half spread for crossing)
        let spread_cost = order_book.spread_bps * urgency * 0.5;
        
        // Component 2: Market impact (square-root law)
        // Impact = coefficient * (quantity / daily_volume)^0.5 * volatility
        // Simplified: use order book depth as proxy
        let available_depth = self.get_available_depth(order_book, side);
        let depth_ratio = if available_depth > 0.0 {
            quantity / available_depth
        } else {
            1.0
        };
        
        let impact_coef = self.impact_coefficient.load(Ordering::Relaxed);
        let volatility = self.current_volatility.load(Ordering::Relaxed);
        
        // Square-root impact model
        let market_impact = impact_coef * depth_ratio.sqrt() * volatility * 100.0 * urgency;
        
        // Component 3: Timing cost (risk of price moving against us)
        // Higher urgency = lower timing cost
        let timing_cost = volatility * 100.0 * (1.0 - urgency) * 0.5;
        
        // Total expected slippage
        let total_slippage = spread_cost + market_impact + timing_cost;
        
        // Confidence based on order book quality and history
        let confidence = self.calculate_confidence(order_book, quantity);
        
        // Recommend optimal urgency
        let optimal_urgency = self.find_optimal_urgency(quantity, order_book);
        
        SlippagePrediction {
            expected_slippage_bps: total_slippage,
            market_impact_bps: market_impact,
            spread_cost_bps: spread_cost,
            timing_cost_bps: timing_cost,
            confidence_interval_bps: total_slippage * (1.0 - confidence) * 0.5,
            recommended_urgency: optimal_urgency,
        }
    }

    /// Get available liquidity depth for a side
    fn get_available_depth(&self, book: &OrderBookState, side: TradeSide) -> f64 {
        let levels = match side {
            TradeSide::Buy => &book.asks,
            TradeSide::Sell => &book.bids,
        };
        
        levels.iter().take(10).map(|(_, qty)| qty).sum()
    }

    /// Calculate confidence score for prediction
    fn calculate_confidence(&self, book: &OrderBookState, quantity: f64) -> f64 {
        let mut confidence = 0.5;
        
        // Tighter spread = higher confidence
        let spread_factor = (10.0 / book.spread_bps.max(1.0)).min(1.0);
        confidence += spread_factor * 0.2;
        
        // More history = higher confidence
        let history_factor = (self.trade_history.len() as f64 / self.max_history_size as f64).min(1.0);
        confidence += history_factor * 0.2;
        
        // Smaller order relative to book = higher confidence
        let depth = self.get_available_depth(book, TradeSide::Buy).max(
            self.get_available_depth(book, TradeSide::Sell)
        );
        if depth > 0.0 {
            let size_factor = (1.0 - (quantity / depth).min(1.0));
            confidence += size_factor * 0.1;
        }
        
        confidence.min(1.0)
    }

    /// Find optimal urgency level that minimizes total cost
    fn find_optimal_urgency(&self, quantity: f64, book: &OrderBookState) -> f64 {
        // Binary search for optimal urgency
        let mut best_urgency = 0.5;
        let mut best_cost = f64::MAX;
        
        for i in 0..10 {
            let urgency = i as f64 / 9.0;
            let pred = self.predict_slippage(book, quantity, TradeSide::Buy, urgency);
            
            if pred.expected_slippage_bps < best_cost {
                best_cost = pred.expected_slippage_bps;
                best_urgency = urgency;
            }
        }
        
        best_urgency
    }

    /// Update model parameters from recent trades
    pub fn calibrate_from_history(&mut self) {
        if self.trade_history.len() < 10 {
            return;
        }
        
        // Simple calibration: average realized impact
        let mut total_impact = 0.0;
        let mut count = 0;
        
        for i in 1..self.trade_history.len() {
            let prev = &self.trade_history[i - 1];
            let curr = &self.trade_history[i];
            
            if prev.was_aggressor && curr.was_aggressor {
                let price_change = (curr.price - prev.price).abs();
                let avg_price = (curr.price + prev.price) / 2.0;
                
                if avg_price > 0.0 {
                    let impact_bps = price_change / avg_price * 10000.0;
                    total_impact += impact_bps;
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            let avg_impact = total_impact / count as f64;
            self.impact_coefficient.store((avg_impact / 100.0).clamp(0.01, 1.0), Ordering::Relaxed);
        }
    }

    /// Update volatility estimate
    #[inline(always)]
    pub fn update_volatility(&self, new_vol: f64) {
        self.current_volatility.store(new_vol.clamp(0.001, 0.5), Ordering::Relaxed);
    }

    /// Update average spread
    #[inline(always)]
    pub fn update_avg_spread(&self, spread_bps: f64) {
        self.avg_spread_bps.store(spread_bps.clamp(0.1, 100.0), Ordering::Relaxed);
    }

    /// Generate TCA report for completed execution
    pub fn generate_tca_report(
        &self,
        fills: &[FillRecord],
        benchmark_price: f64,
        vwap_price: f64,
        fee_bps: f64,
    ) -> TcaReport {
        if fills.is_empty() {
            return TcaReport {
                total_slippage_bps: 0.0,
                benchmark_price,
                average_fill_price: 0.0,
                total_quantity: 0.0,
                total_notional: 0.0,
                market_impact_bps: 0.0,
                timing_cost_bps: 0.0,
                spread_cost_bps: 0.0,
                fee_cost_bps: fee_bps,
                total_cost_bps: fee_bps,
                performance_vs_arrival: 0.0,
                performance_vs_vwap: 0.0,
                execution_quality_score: 0.0,
            };
        }
        
        let total_qty: f64 = fills.iter().map(|f| f.quantity).sum();
        let total_notional: f64 = fills.iter().map(|f| f.price * f.quantity).sum();
        let avg_fill_price = total_notional / total_qty;
        
        // Arrival cost (vs benchmark)
        let arrival_cost_bps = if benchmark_price > 0.0 {
            (avg_fill_price - benchmark_price) / benchmark_price * 10000.0
        } else {
            0.0
        };
        
        // VWAP performance
        let vwap_performance_bps = if vwap_price > 0.0 {
            (avg_fill_price - vwap_price) / vwap_price * 10000.0
        } else {
            0.0
        };
        
        // Decompose costs (simplified)
        let spread_estimate = self.avg_spread_bps.load(Ordering::Relaxed) * 0.5;
        let market_impact = (arrival_cost_bps.abs() - spread_estimate - fee_bps).max(0.0);
        let timing_cost = arrival_cost_bps.abs() - spread_estimate - market_impact - fee_bps;
        
        let total_cost = arrival_cost_bps.abs() + fee_bps;
        
        // Quality score (lower cost = higher score)
        let quality_score = (100.0 - total_cost).max(0.0) / 100.0;
        
        TcaReport {
            total_slippage_bps: arrival_cost_bps.abs(),
            benchmark_price,
            average_fill_price: avg_fill_price,
            total_quantity: total_qty,
            total_notional,
            market_impact_bps: market_impact,
            timing_cost_bps: timing_cost.max(0.0),
            spread_cost_bps: spread_estimate,
            fee_cost_bps: fee_bps,
            total_cost_bps: total_cost,
            performance_vs_arrival: -arrival_cost_bps, // Negative is better for buys
            performance_vs_vwap: -vwap_performance_bps,
            execution_quality_score: quality_score,
        }
    }
}

/// Fill record for TCA
#[derive(Debug, Clone)]
pub struct FillRecord {
    pub timestamp_ns: u64,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_prediction() {
        let model = SlippageModel::new(1000);
        
        let book = OrderBookState {
            bids: vec![(49999.0, 10.0), (49998.0, 20.0)],
            asks: vec![(50001.0, 10.0), (50002.0, 20.0)],
            spread_bps: 4.0,
            mid_price: 50000.0,
            tick_size: 0.01,
        };
        
        let prediction = model.predict_slippage(&book, 5.0, TradeSide::Buy, 0.5);
        
        assert!(prediction.expected_slippage_bps > 0.0);
        assert!(prediction.market_impact_bps >= 0.0);
        assert!(prediction.spread_cost_bps >= 0.0);
    }

    #[test]
    fn test_tca_report() {
        let model = SlippageModel::new(1000);
        
        let fills = vec![
            FillRecord { timestamp_ns: 0, price: 50010.0, quantity: 2.0, side: TradeSide::Buy },
            FillRecord { timestamp_ns: 1000, price: 50015.0, quantity: 3.0, side: TradeSide::Buy },
        ];
        
        let report = model.generate_tca_report(&fills, 50000.0, 50005.0, 1.0);
        
        assert!(report.total_cost_bps > 0.0);
        assert!(report.execution_quality_score > 0.0);
    }
}

//! Real-time funding rate predictor and basis trading engine
//! Analyzes order book imbalances and historical funding prints

use std::collections::VecDeque;

/// Historical funding rate data point
#[derive(Debug, Clone, Copy)]
pub struct FundingPrint {
    pub timestamp_ns: u64,
    pub funding_rate: f64,
    pub predicted_rate: f64,
    pub index_price: f64,
    pub mark_price: f64,
}

/// Order book imbalance snapshot
#[derive(Debug, Clone, Copy)]
pub struct OrderBookImbalance {
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub mid_price: f64,
    pub timestamp_ns: u64,
}

impl OrderBookImbalance {
    /// Calculate normalized imbalance (-1.0 to 1.0)
    #[inline]
    pub fn normalized_imbalance(&self) -> f64 {
        let total = self.bid_volume + self.ask_volume;
        if total <= 0.0 {
            return 0.0;
        }
        (self.bid_volume - self.ask_volume) / total
    }
}

/// Funding rate prediction model
pub struct FundingPredictor {
    /// Rolling window of historical funding prints
    history: VecDeque<FundingPrint>,
    /// Rolling window of order book imbalances
    ob_imbalances: VecDeque<OrderBookImbalance>,
    /// Maximum history size (bounded memory)
    max_history_size: usize,
    /// Current predicted funding rate
    current_prediction: f64,
    /// Prediction confidence (0.0 to 1.0)
    confidence: f64,
    /// Annualized funding rate estimate
    annualized_rate: f64,
}

impl FundingPredictor {
    pub fn new(max_history_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history_size),
            ob_imbalances: VecDeque::with_capacity(max_history_size * 2),
            max_history_size,
            current_prediction: 0.0,
            confidence: 0.0,
            annualized_rate: 0.0,
        }
    }

    /// Add a new funding print to history
    pub fn add_funding_print(&mut self, print: FundingPrint) {
        if self.history.len() >= self.max_history_size {
            self.history.pop_front();
        }
        self.history.push_back(print);
        self.update_prediction();
    }

    /// Add order book imbalance snapshot
    pub fn add_orderbook_snapshot(&mut self, snapshot: OrderBookImbalance) {
        if self.ob_imbalances.len() >= self.max_history_size * 2 {
            self.ob_imbalances.pop_front();
        }
        self.ob_imbalances.push_back(snapshot);
        self.update_prediction();
    }

    /// Update funding rate prediction using weighted model
    fn update_prediction(&mut self) {
        if self.history.is_empty() {
            return;
        }

        // Component 1: Historical funding rate mean reversion
        let hist_mean: f64 = self.history.iter().map(|h| h.funding_rate).sum::<f64>()
            / self.history.len() as f64;
        let hist_std: f64 = self.calculate_std_dev(|h| h.funding_rate);

        // Component 2: Recent trend (exponential weighting)
        let recent_trend = self.calculate_ewma(0.3, |h| h.funding_rate);

        // Component 3: Order book imbalance signal
        let ob_signal = if !self.ob_imbalances.is_empty() {
            self.calculate_ewma_ob(0.5, |ob| ob.normalized_imbalance())
        } else {
            0.0
        };

        // Component 4: Basis signal (mark vs index)
        let basis_signal = if let Some(last) = self.history.back() {
            (last.mark_price - last.index_price) / last.index_price
        } else {
            0.0
        };

        // Weighted combination (weights sum to 1.0)
        let w_hist = 0.25;
        let w_trend = 0.30;
        let w_ob = 0.25;
        let w_basis = 0.20;

        self.current_prediction = w_hist * hist_mean
            + w_trend * recent_trend
            + w_ob * ob_signal * 0.0001 // Scale OB signal
            + w_basis * basis_signal;

        // Calculate confidence based on prediction variance
        self.confidence = (1.0 - hist_std.min(1.0)).max(0.0);

        // Annualize (assuming 8-hour funding intervals, 3 per day, 365 days)
        self.annualized_rate = self.current_prediction * 3.0 * 365.0 * 100.0; // As percentage
    }

    /// Calculate standard deviation with mapper
    fn calculate_std_dev<F>(&self, mapper: F) -> f64
    where
        F: Fn(&FundingPrint) -> f64,
    {
        if self.history.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self.history.iter().map(&mapper).sum::<f64>() / self.history.len() as f64;
        let variance: f64 = self
            .history
            .iter()
            .map(|h| (mapper(h) - mean).powi(2))
            .sum::<f64>()
            / (self.history.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate exponentially weighted moving average
    fn calculate_ewma<F>(&self, alpha: f64, mapper: F) -> f64
    where
        F: Fn(&FundingPrint) -> f64,
    {
        if self.history.is_empty() {
            return 0.0;
        }

        let mut ewma = mapper(&self.history[0]);
        for print in self.history.iter().skip(1) {
            ewma = alpha * mapper(print) + (1.0 - alpha) * ewma;
        }
        ewma
    }

    /// Calculate EWMA for order book imbalances
    fn calculate_ewma_ob<F>(&self, alpha: f64, mapper: F) -> f64
    where
        F: Fn(&OrderBookImbalance) -> f64,
    {
        if self.ob_imbalances.is_empty() {
            return 0.0;
        }

        let mut ewma = mapper(&self.ob_imbalances[0]);
        for ob in self.ob_imbalances.iter().skip(1) {
            ewma = alpha * mapper(ob) + (1.0 - alpha) * ewma;
        }
        ewma
    }

    /// Get current funding rate prediction
    #[inline]
    pub fn get_prediction(&self) -> f64 {
        self.current_prediction
    }

    /// Get prediction confidence
    #[inline]
    pub fn get_confidence(&self) -> f64 {
        self.confidence
    }

    /// Get annualized funding rate (as percentage)
    #[inline]
    pub fn get_annualized_rate(&self) -> f64 {
        self.annualized_rate
    }

    /// Check if arbitrage opportunity exists
    #[inline]
    pub fn is_arb_opportunity(&self, threshold_bps: f64) -> bool {
        self.annualized_rate.abs() > threshold_bps
    }

    /// Get recommended position direction for funding arb
    /// Returns: positive for long spot/short perp, negative for short spot/long perp
    pub fn get_arb_direction(&self) -> f64 {
        if self.current_prediction > 0.0 {
            1.0 // Long spot, short perp to collect funding
        } else {
            -1.0 // Short spot, long perp to collect funding
        }
    }
}

/// Basis trading signal generator
pub struct BasisTradingEngine {
    predictor: FundingPredictor,
    /// Minimum edge in basis points to trigger trade
    min_edge_bps: f64,
    /// Position sizing based on Kelly Criterion
    kelly_fraction: f64,
}

impl BasisTradingEngine {
    pub fn new(predictor: FundingPredictor, min_edge_bps: f64) -> Self {
        Self {
            predictor,
            min_edge_bps,
            kelly_fraction: 0.0,
        }
    }

    /// Update and check for trading signals
    pub fn check_signal(&mut self) -> Option<BasisSignal> {
        let prediction = self.predictor.get_prediction();
        let annualized = self.predictor.get_annualized_rate();
        let confidence = self.predictor.get_confidence();

        if annualized.abs() < self.min_edge_bps {
            return None;
        }

        // Calculate Kelly fraction
        let win_prob = 0.5 + confidence * 0.3; // Map confidence to win probability
        let payoff_ratio = annualized.abs() / 10.0; // Simplified
        let kelly = if payoff_ratio > 0.0 {
            (win_prob * payoff_ratio - (1.0 - win_prob)) / payoff_ratio
        } else {
            0.0
        };
        self.kelly_fraction = kelly.clamp(0.0, 0.25); // Cap at 25%

        Some(BasisSignal {
            direction: self.predictor.get_arb_direction(),
            expected_annual_return: annualized,
            kelly_fraction: self.kelly_fraction,
            confidence,
            funding_rate: prediction,
        })
    }

    pub fn get_predictor(&self) -> &FundingPredictor {
        &self.predictor
    }

    pub fn get_predictor_mut(&mut self) -> &mut FundingPredictor {
        &mut self.predictor
    }
}

/// Trading signal from basis engine
#[derive(Debug, Clone, Copy)]
pub struct BasisSignal {
    pub direction: f64,
    pub expected_annual_return: f64,
    pub kelly_fraction: f64,
    pub confidence: f64,
    pub funding_rate: f64,
}

impl Default for FundingPredictor {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_prediction() {
        let mut predictor = FundingPredictor::new(50);

        // Add some historical data
        for i in 0..10 {
            predictor.add_funding_print(FundingPrint {
                timestamp_ns: i * 3600 * 1_000_000_000,
                funding_rate: 0.0001 * (i as f64 % 3 - 1.0),
                predicted_rate: 0.0001,
                index_price: 50000.0,
                mark_price: 50000.0 + (i as f64 * 10.0),
            });
        }

        assert!(predictor.get_prediction().is_finite());
        assert!(predictor.get_confidence() >= 0.0);
    }

    #[test]
    fn test_orderbook_imbalance() {
        let ob = OrderBookImbalance {
            bid_volume: 100.0,
            ask_volume: 50.0,
            mid_price: 50000.0,
            timestamp_ns: 1000000,
        };

        let imbalance = ob.normalized_imbalance();
        assert!(imbalance > 0.0);
        assert!(imbalance <= 1.0);
    }
}

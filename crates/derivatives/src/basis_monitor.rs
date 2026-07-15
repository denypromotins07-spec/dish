//! Perpetual vs. Spot premium/discount tracker
//! Calculates exact annualized basis across multiple exchanges

use std::collections::HashMap;

/// Basis snapshot for a single instrument
#[derive(Debug, Clone, Copy)]
pub struct BasisSnapshot {
    pub timestamp_ns: u64,
    pub spot_price: f64,
    pub perp_price: f64,
    pub futures_price: Option<f64>,
    pub futures_expiry_days: Option<u32>,
    /// Funding rate (per interval)
    pub funding_rate: f64,
}

impl BasisSnapshot {
    /// Calculate perpetual basis (annualized)
    #[inline]
    pub fn perp_basis_annualized(&self) -> f64 {
        if self.spot_price <= 0.0 {
            return 0.0;
        }
        let basis = (self.perp_price - self.spot_price) / self.spot_price;
        // Annualize assuming 3 funding intervals per day
        basis * 3.0 * 365.0 * 100.0 // As percentage
    }

    /// Calculate futures basis (annualized)
    #[inline]
    pub fn futures_basis_annualized(&self) -> Option<f64> {
        let futures_price = self.futures_price?;
        let expiry_days = self.futures_expiry_days?;

        if self.spot_price <= 0.0 || expiry_days == 0 {
            return None;
        }

        let basis = (futures_price - self.spot_price) / self.spot_price;
        Some(basis * 365.0 / expiry_days as f64 * 100.0) // As percentage
    }

    /// Calculate implied funding rate from basis
    #[inline]
    pub fn implied_funding_rate(&self) -> f64 {
        if self.spot_price <= 0.0 {
            return 0.0;
        }
        (self.perp_price - self.spot_price) / self.spot_price
    }
}

/// Multi-exchange basis monitor
pub struct BasisMonitor {
    /// Latest snapshots per exchange per symbol
    snapshots: HashMap<ExchangeSymbolKey, BasisSnapshot>,
    /// Historical basis for trend analysis
    history: HashMap<ExchangeSymbolKey, VecDeque<BasisSnapshot>>,
    /// Maximum history size
    max_history: usize,
    /// Minimum edge threshold for signals (in basis points)
    min_edge_bps: f64,
}

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExchangeSymbolKey {
    pub exchange_id: [u8; 8],
    pub symbol: [u8; 16],
}

impl ExchangeSymbolKey {
    pub fn new(exchange: &str, symbol: &str) -> Self {
        let mut exchange_id = [0u8; 8];
        let mut sym = [0u8; 16];
        
        exchange_id[..exchange.len().min(8)].copy_from_slice(&exchange.as_bytes()[..exchange.len().min(8)]);
        sym[..symbol.len().min(16)].copy_from_slice(&symbol.as_bytes()[..symbol.len().min(16)]);
        
        Self { exchange_id, symbol: sym }
    }
}

/// Basis arbitrage signal
#[derive(Debug, Clone, Copy)]
pub struct BasisSignal {
    pub exchange: [u8; 8],
    pub symbol: [u8; 16],
    /// Positive = perp premium (long spot, short perp)
    /// Negative = perp discount (short spot, long perp)
    pub basis_bps: f64,
    pub annualized_return_bps: f64,
    pub confidence: f64,
    pub recommended_size_fraction: f64,
}

impl BasisMonitor {
    pub fn new(max_history: usize, min_edge_bps: f64) -> Self {
        Self {
            snapshots: HashMap::new(),
            history: HashMap::new(),
            max_history,
            min_edge_bps,
        }
    }

    /// Update basis snapshot for an exchange/symbol
    pub fn update_snapshot(&mut self, key: ExchangeSymbolKey, snapshot: BasisSnapshot) -> Option<BasisSignal> {
        // Update current snapshot
        self.snapshots.insert(key, snapshot);

        // Update history
        let hist = self.history.entry(key).or_insert_with(|| VecDeque::with_capacity(self.max_history));
        if hist.len() >= self.max_history {
            hist.pop_front();
        }
        hist.push_back(snapshot);

        // Check for arbitrage opportunity
        self.check_arb_opportunity(key, snapshot)
    }

    /// Check for cross-exchange arbitrage opportunities
    pub fn check_cross_exchange_arb(&self, symbol: &[u8; 16]) -> Option<CrossExchangeArb> {
        let mut best_premium: Option<(ExchangeSymbolKey, f64)> = None;
        let mut best_discount: Option<(ExchangeSymbolKey, f64)> = None;

        for (&key, &snapshot) in &self.snapshots {
            if &key.symbol != symbol {
                continue;
            }

            let basis = snapshot.perp_basis_annualized();

            if best_premium.is_none() || basis > best_premium.unwrap().1 {
                best_premium = Some((key, basis));
            }
            if best_discount.is_none() || basis < best_discount.unwrap().1 {
                best_discount = Some((key, basis));
            }
        }

        if let (Some((prem_key, prem_val)), Some((disc_key, disc_val))) = (best_premium, best_discount) {
            let spread = prem_val - disc_val;
            if spread > self.min_edge_bps {
                return Some(CrossExchangeArb {
                    long_exchange: disc_key.exchange_id,
                    short_exchange: prem_key.exchange_id,
                    symbol: *symbol,
                    spread_bps: spread,
                    expected_return_bps: spread / 2.0, // Rough estimate after costs
                });
            }
        }

        None
    }

    /// Check single-exchange arb opportunity
    fn check_arb_opportunity(&self, key: ExchangeSymbolKey, snapshot: BasisSnapshot) -> Option<BasisSignal> {
        let basis_bps = snapshot.perp_basis_annualized() * 100.0; // Convert to bps

        if basis_bps.abs() < self.min_edge_bps {
            return None;
        }

        // Calculate confidence based on historical consistency
        let confidence = self.calculate_signal_confidence(key);

        // Kelly-based sizing
        let kelly = self.calculate_kelly_fraction(basis_bps / 10000.0, confidence);

        Some(BasisSignal {
            exchange: key.exchange_id,
            symbol: key.symbol,
            basis_bps,
            annualized_return_bps: basis_bps * 3.0, // Rough annualization
            confidence,
            recommended_size_fraction: kelly.clamp(0.0, 0.2),
        })
    }

    /// Calculate signal confidence from historical data
    fn calculate_signal_confidence(&self, key: ExchangeSymbolKey) -> f64 {
        let history = match self.history.get(&key) {
            Some(h) => h,
            None => return 0.5,
        };

        if history.len() < 5 {
            return 0.3;
        }

        // Calculate mean reversion tendency
        let bases: Vec<f64> = history.iter().map(|s| s.perp_basis_annualized()).collect();
        let mean: f64 = bases.iter().sum::<f64>() / bases.len() as f64;
        let variance: f64 = bases.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / bases.len() as f64;
        let std_dev = variance.sqrt();

        // Lower volatility = higher confidence
        (1.0 - (std_dev * 10.0).min(1.0)).max(0.0)
    }

    /// Calculate Kelly fraction for position sizing
    fn calculate_kelly_fraction(&self, edge: f64, win_prob: f64) -> f64 {
        if edge <= 0.0 {
            return 0.0;
        }

        // Simplified Kelly: f = (p * b - q) / b
        // where p = win probability, b = payoff ratio, q = loss probability
        let b = edge.abs() / 0.01; // Normalize edge
        let p = win_prob;
        let q = 1.0 - p;

        if b <= 0.0 {
            return 0.0;
        }

        (p * b - q) / b
    }

    /// Get latest basis for exchange/symbol
    pub fn get_basis(&self, key: ExchangeSymbolKey) -> Option<&BasisSnapshot> {
        self.snapshots.get(&key)
    }

    /// Get average basis across all exchanges for a symbol
    pub fn get_average_basis(&self, symbol: &[u8; 16]) -> Option<f64> {
        let mut total = 0.0;
        let mut count = 0;

        for (&key, &snapshot) in &self.snapshots {
            if &key.symbol == symbol {
                total += snapshot.perp_basis_annualized();
                count += 1;
            }
        }

        if count > 0 {
            Some(total / count as f64)
        } else {
            None
        }
    }

    /// Get basis spread between exchanges
    pub fn get_basis_spread(&self, symbol: &[u8; 16], exchange1: [u8; 8], exchange2: [u8; 8]) -> Option<f64> {
        let key1 = ExchangeSymbolKey { exchange_id: exchange1, symbol: *symbol };
        let key2 = ExchangeSymbolKey { exchange_id: exchange2, symbol: *symbol };

        let basis1 = self.snapshots.get(&key1)?.perp_basis_annualized();
        let basis2 = self.snapshots.get(&key2)?.perp_basis_annualized();

        Some(basis1 - basis2)
    }
}

/// Cross-exchange arbitrage opportunity
#[derive(Debug, Clone, Copy)]
pub struct CrossExchangeArb {
    pub long_exchange: [u8; 8],
    pub short_exchange: [u8; 8],
    pub symbol: [u8; 16],
    pub spread_bps: f64,
    pub expected_return_bps: f64,
}

/// Basis curve term structure analyzer
pub struct BasisCurveAnalyzer {
    /// Futures curves per symbol (expiry -> basis)
    curves: HashMap<[u8; 16], Vec<(u32, f64)>>,
}

impl BasisCurveAnalyzer {
    pub fn new() -> Self {
        Self {
            curves: HashMap::new(),
        }
    }

    /// Add futures basis point to curve
    pub fn add_curve_point(&mut self, symbol: [u8; 16], expiry_days: u32, basis: f64) {
        let curve = self.curves.entry(symbol).or_insert_with(Vec::new);
        
        // Remove existing point for same expiry
        curve.retain(|(exp, _)| *exp != expiry_days);
        curve.push((expiry_days, basis));
        
        // Sort by expiry
        curve.sort_by_key(|(exp, _)| *exp);
    }

    /// Determine if curve is in contango or backwardation
    pub fn get_curve_shape(&self, symbol: &[u8; 16]) -> Option<CurveShape> {
        let curve = self.curves.get(symbol)?;
        
        if curve.len() < 2 {
            return None;
        }

        let first_basis = curve.first()?.1;
        let last_basis = curve.last()?.1;

        if last_basis > first_basis {
            Some(CurveShape::Contango)
        } else if last_basis < first_basis {
            Some(CurveShape::Backwardation)
        } else {
            Some(CurveShape::Flat)
        }
    }

    /// Calculate roll yield for futures curve
    pub fn calculate_roll_yield(&self, symbol: &[u8; 16]) -> Option<f64> {
        let curve = self.curves.get(symbol)?;
        
        if curve.len() < 2 {
            return None;
        }

        // Approximate roll yield as slope of curve
        let first = curve.first()?;
        let last = curve.last()?;

        if last.0 == first.0 {
            return None;
        }

        Some((last.1 - first.1) / (last.0 - first.0) as f64 * 365.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveShape {
    Contango,
    Backwardation,
    Flat,
}

impl Default for BasisMonitor {
    fn default() -> Self {
        Self::new(100, 50.0) // 50 bps minimum edge
    }
}

impl Default for BasisCurveAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_calculation() {
        let snapshot = BasisSnapshot {
            timestamp_ns: 1000,
            spot_price: 50000.0,
            perp_price: 50100.0,
            futures_price: Some(50200.0),
            futures_expiry_days: Some(30),
            funding_rate: 0.0001,
        };

        let perp_basis = snapshot.perp_basis_annualized();
        assert!(perp_basis > 0.0);

        let futures_basis = snapshot.futures_basis_annualized();
        assert!(futures_basis.is_some());
    }

    #[test]
    fn test_basis_monitor() {
        let mut monitor = BasisMonitor::new(50, 100.0);
        let key = ExchangeSymbolKey::new("BINANCE", "BTC-PERP");

        let snapshot = BasisSnapshot {
            timestamp_ns: 1000,
            spot_price: 50000.0,
            perp_price: 50500.0,
            futures_price: None,
            futures_expiry_days: None,
            funding_rate: 0.0001,
        };

        let signal = monitor.update_snapshot(key, snapshot);
        assert!(signal.is_some());
    }
}

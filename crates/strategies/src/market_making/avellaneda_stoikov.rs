//! Avellaneda-Stoikov Market Making Model
//! High-performance Rust implementation calculating optimal bid/ask quotes
//! with strict inventory risk aversion parameters.

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Core state for Avellaneda-Stoikov model
pub struct AvellanedaStoikovState {
    /// Current mid-price (atomic for lock-free reads)
    pub mid_price: AtomicF64,
    /// Inventory position (signed quantity)
    pub inventory: AtomicF64,
    /// Risk aversion parameter (gamma)
    pub gamma: AtomicF64,
    /// Volatility estimate (sigma)
    pub volatility: AtomicF64,
    /// Order arrival intensity (kappa)
    pub kappa: AtomicF64,
    /// Time horizon in seconds
    pub time_horizon: AtomicF64,
    /// Last update timestamp (nanoseconds)
    pub last_update_ns: AtomicU64,
}

impl AvellanedaStoikovState {
    pub fn new(
        mid_price: f64,
        gamma: f64,
        volatility: f64,
        kappa: f64,
        time_horizon: f64,
    ) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            mid_price: AtomicF64::new(mid_price),
            inventory: AtomicF64::new(0.0),
            gamma: AtomicF64::new(gamma),
            volatility: AtomicF64::new(volatility),
            kappa: AtomicF64::new(kappa),
            time_horizon: AtomicF64::new(time_horizon),
            last_update_ns: AtomicU64::new(now_ns),
        }
    }

    /// Update mid-price (lock-free)
    #[inline]
    pub fn update_mid_price(&self, price: f64) {
        self.mid_price.store(price, Ordering::Relaxed);
        self.last_update_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Update inventory (lock-free)
    #[inline]
    pub fn update_inventory(&self, qty: f64) {
        self.inventory.fetch_add(qty, Ordering::Relaxed);
    }

    /// Calculate reservation price (Stoikov formula)
    /// r = s - q * gamma * sigma^2 * (T - t)
    #[inline]
    pub fn reservation_price(&self) -> f64 {
        let s = self.mid_price.load(Ordering::Relaxed);
        let q = self.inventory.load(Ordering::Relaxed);
        let gamma = self.gamma.load(Ordering::Relaxed);
        let sigma = self.volatility.load(Ordering::Relaxed);
        let t = self.time_horizon.load(Ordering::Relaxed);
        
        // Fast math: avoid unnecessary checks in hot path
        s - q * gamma * sigma * sigma * t
    }

    /// Calculate optimal bid quote
    /// delta_b = 1/gamma * ln(1 + gamma/kappa) + (q - 1) * gamma * sigma^2 * (T - t) / 2
    #[inline]
    pub fn optimal_bid_spread(&self) -> f64 {
        let gamma = self.gamma.load(Ordering::Relaxed);
        let kappa = self.kappa.load(Ordering::Relaxed);
        let sigma = self.volatility.load(Ordering::Relaxed);
        let q = self.inventory.load(Ordering::Relaxed);
        let t = self.time_horizon.load(Ordering::Relaxed);
        
        let term1 = (1.0 / gamma) * ((1.0 + gamma / kappa).ln());
        let term2 = (q - 1.0) * gamma * sigma * sigma * t / 2.0;
        
        term1 + term2
    }

    /// Calculate optimal ask quote
    /// delta_a = 1/gamma * ln(1 + gamma/kappa) + (q + 1) * gamma * sigma^2 * (T - t) / 2
    #[inline]
    pub fn optimal_ask_spread(&self) -> f64 {
        let gamma = self.gamma.load(Ordering::Relaxed);
        let kappa = self.kappa.load(Ordering::Relaxed);
        let sigma = self.volatility.load(Ordering::Relaxed);
        let q = self.inventory.load(Ordering::Relaxed);
        let t = self.time_horizon.load(Ordering::Relaxed);
        
        let term1 = (1.0 / gamma) * ((1.0 + gamma / kappa).ln());
        let term2 = (q + 1.0) * gamma * sigma * sigma * t / 2.0;
        
        term1 + term2
    }

    /// Get optimal bid and ask prices
    #[inline]
    pub fn get_quotes(&self) -> (f64, f64) {
        let res_price = self.reservation_price();
        let bid_spread = self.optimal_bid_spread();
        let ask_spread = self.optimal_ask_spread();
        
        (res_price - bid_spread, res_price + ask_spread)
    }

    /// Update volatility estimate (exponential moving average)
    #[inline]
    pub fn update_volatility(&self, new_vol: f64, alpha: f64) {
        let current = self.volatility.load(Ordering::Relaxed);
        let updated = alpha * new_vol + (1.0 - alpha) * current;
        self.volatility.store(updated, Ordering::Relaxed);
    }
}

/// Quote result with metadata
#[derive(Clone, Copy, Debug)]
pub struct Quote {
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub reservation_price: f64,
    pub timestamp_ns: u64,
}

impl Quote {
    pub fn new(bid: f64, ask: f64, bid_size: f64, ask_size: f64, res_price: f64) -> Self {
        Self {
            bid_price: bid,
            ask_price: ask,
            bid_size,
            ask_size,
            reservation_price: res_price,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
}

/// Market maker engine using Avellaneda-Stoikov
pub struct MarketMakerEngine {
    pub state: AvellanedaStoikovState,
    /// Max inventory limit (absolute value)
    pub max_inventory: AtomicF64,
    /// Min spread (basis points)
    pub min_spread_bps: AtomicF64,
    /// Base order size
    pub base_size: AtomicF64,
}

impl MarketMakerEngine {
    pub fn new(
        mid_price: f64,
        gamma: f64,
        volatility: f64,
        kappa: f64,
        time_horizon: f64,
        max_inventory: f64,
        min_spread_bps: f64,
        base_size: f64,
    ) -> Self {
        Self {
            state: AvellanedaStoikovState::new(mid_price, gamma, volatility, kappa, time_horizon),
            max_inventory: AtomicF64::new(max_inventory),
            min_spread_bps: AtomicF64::new(min_spread_bps),
            base_size: AtomicF64::new(base_size),
        }
    }

    /// Generate quotes with inventory constraints
    #[inline]
    pub fn generate_quotes(&self) -> Option<Quote> {
        let inventory = self.state.inventory.load(Ordering::Relaxed);
        let max_inv = self.max_inventory.load(Ordering::Relaxed);
        
        // Skip if at inventory limits
        if inventory.abs() >= max_inv {
            return None;
        }

        let (bid, ask) = self.state.get_quotes();
        let mid = self.state.mid_price.load(Ordering::Relaxed);
        let min_spread = mid * self.min_spread_bps.load(Ordering::Relaxed) / 10000.0;
        
        // Ensure minimum spread
        let spread = ask - bid;
        if spread < min_spread {
            let adjustment = (min_spread - spread) / 2.0;
            let bid_adj = bid - adjustment;
            let ask_adj = ask + adjustment;
            
            let base_size = self.base_size.load(Ordering::Relaxed);
            // Reduce size near inventory limits
            let size_factor = 1.0 - (inventory.abs() / max_inv);
            let adjusted_size = base_size * size_factor;
            
            Some(Quote::new(
                bid_adj,
                ask_adj,
                adjusted_size,
                adjusted_size,
                self.state.reservation_price(),
            ))
        } else {
            let base_size = self.base_size.load(Ordering::Relaxed);
            let size_factor = 1.0 - (inventory.abs() / max_inv);
            let adjusted_size = base_size * size_factor;
            
            Some(Quote::new(
                bid,
                ask,
                adjusted_size,
                adjusted_size,
                self.state.reservation_price(),
            ))
        }
    }

    /// Process a fill event
    #[inline]
    pub fn on_fill(&self, side: Side, qty: f64, price: f64) {
        match side {
            Side::Bid => self.state.update_inventory(qty),
            Side::Ask => self.state.update_inventory(-qty),
        }
        // Could add PnL tracking here
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservation_price() {
        let mm = MarketMakerEngine::new(
            50000.0,  // mid_price
            0.1,      // gamma
            0.02,     // volatility
            0.5,      // kappa
            60.0,     // time_horizon (1 minute)
            10.0,     // max_inventory
            5.0,      // min_spread_bps
            1.0,      // base_size
        );

        let res_price = mm.state.reservation_price();
        assert!((res_price - 50000.0).abs() < 0.0001); // Zero inventory = mid price

        mm.state.update_inventory(5.0);
        let res_price_long = mm.state.reservation_price();
        assert!(res_price_long < 50000.0); // Long inventory lowers reservation price
    }

    #[test]
    fn test_quote_generation() {
        let mm = MarketMakerEngine::new(
            50000.0,
            0.1,
            0.02,
            0.5,
            60.0,
            10.0,
            5.0,
            1.0,
        );

        let quote = mm.generate_quotes();
        assert!(quote.is_some());
        let q = quote.unwrap();
        assert!(q.bid_price < q.ask_price);
        assert!(q.bid_price > 0.0);
        assert!(q.ask_price > 0.0);
    }

    #[test]
    fn test_inventory_limits() {
        let mm = MarketMakerEngine::new(
            50000.0,
            0.1,
            0.02,
            0.5,
            60.0,
            1.0,  // Very low max_inventory
            5.0,
            1.0,
        );

        mm.state.update_inventory(1.0);
        assert!(mm.generate_quotes().is_none()); // At limit
    }
}

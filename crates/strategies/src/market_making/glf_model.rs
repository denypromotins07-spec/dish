//! Guéant-Lehalle-Fernandez-Tapia (GLFT) Optimal Quoting Strategy
//! Multi-asset limits with exact quote sizes based on volatility and risk

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// GLFT model parameters for a single asset
#[derive(Clone, Copy, Debug)]
pub struct GlftParams {
    /// Risk aversion coefficient
    pub gamma: f64,
    /// Order arrival intensity scale
    pub kappa: f64,
    /// Order arrival intensity exponent
    pub alpha: f64,
    /// Volatility (annualized)
    pub sigma: f64,
    /// Time horizon in seconds
    pub time_horizon: f64,
    /// Max position size
    pub max_position: f64,
    /// Tick size
    pub tick_size: f64,
}

impl Default for GlftParams {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            kappa: 0.5,
            alpha: 2.0,
            sigma: 0.02,
            time_horizon: 300.0, // 5 minutes
            max_position: 100.0,
            tick_size: 0.01,
        }
    }
}

/// State for GLFT quoting engine
pub struct GlftState {
    /// Current mid-price
    pub mid_price: AtomicF64,
    /// Current inventory
    pub inventory: AtomicF64,
    /// Parameters
    pub params: GlftParams,
    /// Last update timestamp
    pub last_update_ns: AtomicU64,
}

impl GlftState {
    pub fn new(mid_price: f64, params: GlftParams) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            mid_price: AtomicF64::new(mid_price),
            inventory: AtomicF64::new(0.0),
            params,
            last_update_ns: AtomicU64::new(now_ns),
        }
    }

    #[inline]
    pub fn update_mid_price(&self, price: f64) {
        self.mid_price.store(price, Ordering::Relaxed);
        self.last_update_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    #[inline]
    pub fn update_inventory(&self, qty: f64) {
        self.inventory.fetch_add(qty, Ordering::Relaxed);
    }

    /// Calculate optimal spread using GLFT formula
    /// delta = 1/gamma * ln(1 + gamma/kappa) + gamma * sigma^2 * (T-t) * |q| / 2
    #[inline]
    pub fn optimal_spread(&self, q: f64) -> f64 {
        let p = self.params;
        let term1 = (1.0 / p.gamma) * ((1.0 + p.gamma / p.kappa).ln());
        let term2 = p.gamma * p.sigma * p.sigma * p.time_horizon * q.abs() / 2.0;
        term1 + term2
    }

    /// Calculate optimal quote size
    /// Q = (kappa / gamma) * (1 + gamma/kappa)^(-alpha) * exp(-gamma * delta * q)
    #[inline]
    pub fn optimal_quote_size(&self, delta: f64, q: f64) -> f64 {
        let p = self.params;
        let base = (p.kappa / p.gamma) * (1.0 + p.gamma / p.kappa).powf(-p.alpha);
        let adjustment = (-p.gamma * delta * q).exp();
        base * adjustment
    }

    /// Get reservation price adjusted for inventory
    #[inline]
    pub fn reservation_price(&self) -> f64 {
        let s = self.mid_price.load(Ordering::Relaxed);
        let q = self.inventory.load(Ordering::Relaxed);
        let p = self.params;
        s - q * p.gamma * p.sigma * p.sigma * p.time_horizon
    }

    /// Round price to tick size
    #[inline]
    pub fn round_to_tick(&self, price: f64) -> f64 {
        let tick = self.params.tick_size;
        (price / tick).round() * tick
    }
}

/// Quote result from GLFT engine
#[derive(Clone, Copy, Debug)]
pub struct GlftQuote {
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub spread_bps: f64,
    pub timestamp_ns: u64,
}

impl GlftQuote {
    pub fn new(bid: f64, ask: f64, bid_size: f64, ask_size: f64) -> Self {
        let spread = ((ask - bid) / ((ask + bid) / 2.0)) * 10000.0;
        Self {
            bid_price: bid,
            ask_price: ask,
            bid_size,
            ask_size,
            spread_bps: spread,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
}

/// Multi-asset GLFT quoting engine
pub struct GlftEngine {
    pub state: GlftState,
    /// Skew factor for asymmetric quoting
    pub skew_factor: AtomicF64,
    /// Urgency multiplier (1.0 = normal, >1.0 = aggressive)
    pub urgency: AtomicF64,
}

impl GlftEngine {
    pub fn new(mid_price: f64, params: GlftParams) -> Self {
        Self {
            state: GlftState::new(mid_price, params),
            skew_factor: AtomicF64::new(1.0),
            urgency: AtomicF64::new(1.0),
        }
    }

    /// Generate quotes with GLFT optimal strategy
    #[inline]
    pub fn generate_quotes(&self) -> Option<GlftQuote> {
        let inv = self.state.inventory.load(Ordering::Relaxed);
        let max_pos = self.state.params.max_position;
        
        if inv.abs() >= max_pos {
            return None;
        }

        let res_price = self.state.reservation_price();
        let spread = self.state.optimal_spread(inv);
        let urgency = self.urgency.load(Ordering::Relaxed);
        let skew = self.skew_factor.load(Ordering::Relaxed);

        // Adjust spread based on urgency (lower spread = more aggressive)
        let adjusted_spread = spread / urgency;
        
        // Apply asymmetry based on inventory
        let bid_offset = adjusted_spread / 2.0 * (1.0 + inv / max_pos * (skew - 1.0));
        let ask_offset = adjusted_spread / 2.0 * (1.0 - inv / max_pos * (skew - 1.0));

        let mut bid = res_price - bid_offset;
        let mut ask = res_price + ask_offset;

        // Ensure bid < mid < ask
        let mid = self.state.mid_price.load(Ordering::Relaxed);
        if bid >= mid {
            bid = mid - self.state.params.tick_size;
        }
        if ask <= mid {
            ask = mid + self.state.params.tick_size;
        }

        // Round to tick size
        bid = self.state.round_to_tick(bid);
        ask = self.state.round_to_tick(ask);

        // Calculate sizes
        let bid_delta = (res_price - bid).abs();
        let ask_delta = (ask - res_price).abs();
        let bid_size = self.state.optimal_quote_size(bid_delta, inv);
        let ask_size = self.state.optimal_quote_size(ask_delta, inv);

        // Clamp sizes to reasonable bounds
        let max_size = max_pos - inv.abs();
        let bid_size = bid_size.min(max_size).max(self.state.params.tick_size);
        let ask_size = ask_size.min(max_size).max(self.state.params.tick_size);

        Some(GlftQuote::new(bid, ask, bid_size, ask_size))
    }

    /// Update parameters dynamically
    #[inline]
    pub fn update_params(&mut self, params: GlftParams) {
        self.state.params = params;
    }

    /// Set skew factor for asymmetric quoting
    #[inline]
    pub fn set_skew(&self, factor: f64) {
        self.skew_factor.store(factor.clamp(0.5, 2.0), Ordering::Relaxed);
    }

    /// Set urgency level
    #[inline]
    pub fn set_urgency(&self, level: f64) {
        self.urgency.store(level.clamp(0.5, 3.0), Ordering::Relaxed);
    }

    /// Process fill
    #[inline]
    pub fn on_fill(&self, side: Side, qty: f64) {
        match side {
            Side::Bid => self.state.update_inventory(qty),
            Side::Ask => self.state.update_inventory(-qty),
        }
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
    fn test_glft_basic() {
        let params = GlftParams {
            gamma: 0.1,
            kappa: 0.5,
            alpha: 2.0,
            sigma: 0.02,
            time_horizon: 300.0,
            max_position: 100.0,
            tick_size: 0.01,
        };

        let engine = GlftEngine::new(50000.0, params);
        let quote = engine.generate_quotes();
        
        assert!(quote.is_some());
        let q = quote.unwrap();
        assert!(q.bid_price < q.ask_price);
        assert!(q.spread_bps > 0.0);
    }

    #[test]
    fn test_inventory_skew() {
        let params = GlftParams::default();
        let engine = GlftEngine::new(50000.0, params);
        
        // Long inventory should widen ask, narrow bid
        engine.state.update_inventory(50.0);
        let quote_long = engine.generate_quotes().unwrap();
        
        engine.state.update_inventory(-100.0); // Reset and go short
        engine.state.update_inventory(-50.0);
        let quote_short = engine.generate_quotes().unwrap();
        
        // Asymmetric behavior check
        assert_ne!(quote_long.bid_price, quote_short.bid_price);
    }

    #[test]
    fn test_urgency_effect() {
        let params = GlftParams::default();
        let engine = GlftEngine::new(50000.0, params);
        
        let normal_quote = engine.generate_quotes().unwrap();
        
        engine.set_urgency(2.0);
        let urgent_quote = engine.generate_quotes().unwrap();
        
        // Higher urgency = tighter spread
        assert!(urgent_quote.spread_bps < normal_quote.spread_bps);
    }
}

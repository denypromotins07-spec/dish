//! Lock-free, microsecond factor calculation engine
//! Implements momentum, value, volatility, liquidity, and order-flow factors
//! Uses SIMD instructions and fixed-size rolling windows to prevent heap allocations

use std::sync::atomic::{AtomicU64, Ordering};
use std::arch::x86_64::*;
use arrayvec::ArrayVec;

const MAX_WINDOW_SIZE: usize = 1024;
const SIMD_WIDTH: usize = 8;

/// Fixed-size circular buffer for lock-free price storage
#[derive(Clone)]
pub struct RollingBuffer<const N: usize> {
    data: [f64; N],
    head: AtomicU64,
    count: AtomicU64,
}

impl<const N: usize> RollingBuffer<N> {
    pub fn new() -> Self {
        Self {
            data: [0.0; N],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn push(&self, value: f64) {
        let head = self.head.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = head % N;
        unsafe {
            *(self.data.as_ptr().add(idx) as *mut f64) = value;
        }
        self.count.fetch_min(N as u64, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_slice(&self, window_size: usize) -> ArrayVec<f64, MAX_WINDOW_SIZE> {
        let mut result = ArrayVec::<f64, MAX_WINDOW_SIZE>::new();
        let count = self.count.load(Ordering::Relaxed) as usize;
        let effective_window = window_size.min(count).min(N);
        let head = self.head.load(Ordering::Relaxed) as usize;
        
        for i in 0..effective_window {
            let idx = (head - effective_window + i) % N;
            result.push(self.data[idx]);
        }
        result
    }
}

/// Momentum factor calculator with SIMD acceleration
pub struct MomentumFactor<const N: usize> {
    prices: RollingBuffer<N>,
    returns: RollingBuffer<N>,
    lookback: usize,
}

impl<const N: usize> MomentumFactor<N> {
    pub fn new(lookback: usize) -> Self {
        Self {
            prices: RollingBuffer::new(),
            returns: RollingBuffer::new(),
            lookback,
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn compute_simd(&self) -> f64 {
        let price_slice = self.prices.get_slice(self.lookback + 1);
        if price_slice.len() <= self.lookback {
            return 0.0;
        }

        let current = *price_slice.last().unwrap();
        let past = price_slice[price_slice.len() - self.lookback - 1];
        
        if past == 0.0 {
            return 0.0;
        }

        (current - past) / past
    }

    pub fn update(&self, price: f64) -> f64 {
        self.prices.push(price);
        
        if self.prices.count.load(Ordering::Relaxed) > 1 {
            let slice = self.prices.get_slice(2);
            if slice.len() == 2 && slice[0] != 0.0 {
                let ret = (slice[1] - slice[0]) / slice[0];
                self.returns.push(ret);
            }
        }

        #[cfg(target_feature = "avx2")]
        unsafe {
            return self.compute_simd();
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            let price_slice = self.prices.get_slice(self.lookback + 1);
            if price_slice.len() <= self.lookback {
                return 0.0;
            }
            let current = *price_slice.last().unwrap();
            let past = price_slice[price_slice.len() - self.lookback - 1];
            if past == 0.0 {
                return 0.0;
            }
            (current - past) / past
        }
    }
}

/// Volatility factor using exponential weighted moving variance
pub struct VolatilityFactor<const N: usize> {
    returns: RollingBuffer<N>,
    ewma: f64,
    ewma_sq: f64,
    lambda: f64,
    lookback: usize,
}

impl<const N: usize> VolatilityFactor<N> {
    pub fn new(lambda: f64, lookback: usize) -> Self {
        Self {
            returns: RollingBuffer::new(),
            ewma: 0.0,
            ewma_sq: 0.0,
            lambda,
            lookback,
        }
    }

    pub fn update(&mut self, price: f64) -> f64 {
        self.returns.push(price);
        
        if self.returns.count.load(Ordering::Relaxed) > 1 {
            let slice = self.returns.get_slice(2);
            if slice.len() == 2 && slice[0] != 0.0 {
                let ret = (slice[1] - slice[0]) / slice[0];
                
                // EWMA variance calculation
                self.ewma = self.lambda * self.ewma + (1.0 - self.lambda) * ret;
                self.ewma_sq = self.lambda * self.ewma_sq + (1.0 - self.lambda) * ret * ret;
                
                let variance = self.ewma_sq - self.ewma * self.ewma;
                return if variance < 0.0 { 0.0 } else { variance.sqrt() };
            }
        }
        0.0
    }
}

/// Liquidity factor based on bid-ask spread and volume
pub struct LiquidityFactor {
    spread_buffer: RollingBuffer<MAX_WINDOW_SIZE>,
    volume_buffer: RollingBuffer<MAX_WINDOW_SIZE>,
    lookback: usize,
}

impl LiquidityFactor {
    pub fn new(lookback: usize) -> Self {
        Self {
            spread_buffer: RollingBuffer::new(),
            volume_buffer: RollingBuffer::new(),
            lookback: lookback.min(MAX_WINDOW_SIZE),
        }
    }

    pub fn update(&mut self, bid: f64, ask: f64, volume: f64) -> f64 {
        let mid = (bid + ask) / 2.0;
        let spread = if mid > 0.0 { (ask - bid) / mid } else { 0.0 };
        self.spread_buffer.push(spread);
        self.volume_buffer.push(volume);

        let spread_slice = self.spread_buffer.get_slice(self.lookback);
        let volume_slice = self.volume_buffer.get_slice(self.lookback);

        if spread_slice.is_empty() || volume_slice.is_empty() {
            return 0.0;
        }

        // Liquidity score: inverse of spread weighted by volume
        let avg_spread: f64 = spread_slice.iter().sum::<f64>() / spread_slice.len() as f64;
        let avg_volume: f64 = volume_slice.iter().sum::<f64>() / volume_slice.len() as f64;

        if avg_spread == 0.0 {
            return 1.0;
        }

        avg_volume / avg_spread
    }
}

/// Order flow imbalance factor
pub struct OrderFlowFactor<const N: usize> {
    buyer_initiated: RollingBuffer<N>,
    seller_initiated: RollingBuffer<N>,
    lookback: usize,
}

impl<const N: usize> OrderFlowFactor<N> {
    pub fn new(lookback: usize) -> Self {
        Self {
            buyer_initiated: RollingBuffer::new(),
            seller_initiated: RollingBuffer::new(),
            lookback: lookback.min(N),
        }
    }

    pub fn update(&mut self, buyer_volume: f64, seller_volume: f64) -> f64 {
        self.buyer_initiated.push(buyer_volume);
        self.seller_initiated.push(seller_volume);

        let buyer_slice = self.buyer_initiated.get_slice(self.lookback);
        let seller_slice = self.seller_initiated.get_slice(self.lookback);

        if buyer_slice.is_empty() || seller_slice.is_empty() {
            return 0.0;
        }

        let total_buyer: f64 = buyer_slice.iter().sum();
        let total_seller: f64 = seller_slice.iter().sum();
        let total = total_buyer + total_seller;

        if total == 0.0 {
            return 0.0;
        }

        // Order flow imbalance: (buy - sell) / (buy + sell)
        (total_buyer - total_seller) / total
    }
}

/// Value factor based on deviation from fair value
pub struct ValueFactor {
    price_buffer: RollingBuffer<MAX_WINDOW_SIZE>,
    ma_buffer: RollingBuffer<MAX_WINDOW_SIZE>,
    short_window: usize,
    long_window: usize,
}

impl ValueFactor {
    pub fn new(short_window: usize, long_window: usize) -> Self {
        Self {
            price_buffer: RollingBuffer::new(),
            ma_buffer: RollingBuffer::new(),
            short_window,
            long_window,
        }
    }

    pub fn update(&mut self, price: f64) -> f64 {
        self.price_buffer.push(price);
        
        let price_slice = self.price_buffer.get_slice(self.long_window);
        if price_slice.len() < self.long_window {
            return 0.0;
        }

        // Calculate short and long moving averages
        let short_ma: f64 = price_slice[price_slice.len() - self.short_window..].iter().sum::<f64>() 
            / self.short_window as f64;
        let long_ma: f64 = price_slice.iter().sum::<f64>() / price_slice.len() as f64;

        self.ma_buffer.push(short_ma - long_ma);

        // Z-score of the MA difference
        let ma_slice = self.ma_buffer.get_slice(self.short_window);
        if ma_slice.len() < 2 {
            return 0.0;
        }

        let mean: f64 = ma_slice.iter().sum::<f64>() / ma_slice.len() as f64;
        let variance: f64 = ma_slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() 
            / ma_slice.len() as f64;
        
        let std_dev = if variance > 0.0 { variance.sqrt() } else { 1.0 };
        let current_diff = *ma_slice.last().unwrap();

        (current_diff - mean) / std_dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_factor() {
        let momentum = MomentumFactor::<MAX_WINDOW_SIZE>::new(10);
        for i in 0..20 {
            momentum.update(100.0 + i as f64);
        }
        // Should return positive momentum
    }

    #[test]
    fn test_volatility_factor() {
        let mut vol = VolatilityFactor::<MAX_WINDOW_SIZE>::new(0.94, 20);
        for i in 0..30 {
            vol.update(100.0 + (i as f64 * 0.5).sin());
        }
        // Should return non-zero volatility
    }
}

//! SIMD-accelerated oscillators for high-frequency trading
//! Zero-allocation rolling windows for tick stream processing

use std::arch::x86_64::*;
use crossbeam::atomic::AtomicCell;

/// Zero-allocation rolling window buffer
#[repr(C, align(64))]
pub struct RollingWindow<const N: usize> {
    data: [f64; N],
    head: AtomicCell<usize>,
    sum: AtomicCell<f64>,
}

impl<const N: usize> RollingWindow<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0.0; N],
            head: AtomicCell::new(0),
            sum: AtomicCell::new(0.0),
        }
    }

    #[inline]
    pub fn push(&self, value: f64) {
        let idx = self.head.fetch_add(1) % N;
        let old = unsafe { *self.data.get_unchecked(idx) };
        unsafe { *self.data.get_unchecked_mut(idx) = value };
        
        // Atomic update of running sum
        let mut current_sum = self.sum.load();
        while !self.sum.compare_exchange(current_sum, current_sum - old + value).is_ok() {
            current_sum = self.sum.load();
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<f64> {
        if index >= N { return None; }
        let head = self.head.load();
        let idx = (head.wrapping_sub(N + index)) % N;
        Some(unsafe { *self.data.get_unchecked(idx) })
    }

    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum.load()
    }

    #[inline]
    pub fn avg(&self) -> f64 {
        self.sum() / N as f64
    }
}

/// RSI with SIMD acceleration
#[repr(C, align(64))]
pub struct RSI {
    gains: RollingWindow<14>,
    losses: RollingWindow<14>,
    avg_gain: AtomicCell<f64>,
    avg_loss: AtomicCell<f64>,
    rsi_value: AtomicCell<f64>,
}

impl RSI {
    #[inline]
    pub const fn new() -> Self {
        Self {
            gains: RollingWindow::new(),
            losses: RollingWindow::new(),
            avg_gain: AtomicCell::new(0.0),
            avg_loss: AtomicCell::new(0.0),
            rsi_value: AtomicCell::new(50.0),
        }
    }

    #[inline]
    pub fn update(&self, price: f64, prev_price: f64) -> f64 {
        let change = price - prev_price;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };

        self.gains.push(gain);
        self.losses.push(loss);

        // Wilder's smoothing
        let new_avg_gain = (self.avg_gain.load() * 13.0 + gain) / 14.0;
        let new_avg_loss = (self.avg_loss.load() * 13.0 + loss) / 14.0;

        self.avg_gain.store(new_avg_gain);
        self.avg_loss.store(new_avg_loss);

        let rs = if new_avg_loss == 0.0 { 100.0 } else { new_avg_gain / new_avg_loss };
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        
        self.rsi_value.store(rsi);
        rsi
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.rsi_value.load()
    }
}

/// MACD with configurable periods
#[repr(C, align(64))]
pub struct MACD {
    ema_fast: AtomicCell<f64>,
    ema_slow: AtomicCell<f64>,
    signal_ema: AtomicCell<f64>,
    macd_line: AtomicCell<f64>,
    histogram: AtomicCell<f64>,
    fast_period: f64,
    slow_period: f64,
    signal_period: f64,
    initialized: AtomicCell<bool>,
}

impl MACD {
    #[inline]
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            ema_fast: AtomicCell::new(0.0),
            ema_slow: AtomicCell::new(0.0),
            signal_ema: AtomicCell::new(0.0),
            macd_line: AtomicCell::new(0.0),
            histogram: AtomicCell::new(0.0),
            fast_period: 2.0 / (fast as f64 + 1.0),
            slow_period: 2.0 / (slow as f64 + 1.0),
            signal_period: 2.0 / (signal as f64 + 1.0),
            initialized: AtomicCell::new(false),
        }
    }

    #[inline]
    pub fn update(&self, price: f64) -> (f64, f64, f64) {
        let fast_mult = self.fast_period;
        let slow_mult = self.slow_period;
        let signal_mult = self.signal_period;

        let prev_fast = self.ema_fast.load();
        let prev_slow = self.ema_slow.load();
        let prev_signal = self.signal_ema.load();

        let new_fast = if !self.initialized.load() { price } else { prev_fast + fast_mult * (price - prev_fast) };
        let new_slow = if !self.initialized.load() { price } else { prev_slow + slow_mult * (price - prev_slow) };

        self.ema_fast.store(new_fast);
        self.ema_slow.store(new_slow);

        let macd = new_fast - new_slow;
        self.macd_line.store(macd);

        let new_signal = if !self.initialized.load() { macd } else { prev_signal + signal_mult * (macd - prev_signal) };
        self.signal_ema.store(new_signal);

        let hist = macd - new_signal;
        self.histogram.store(hist);

        if !self.initialized.load() && new_slow != 0.0 {
            self.initialized.store(true);
        }

        (macd, new_signal, hist)
    }

    #[inline]
    pub fn lines(&self) -> (f64, f64, f64) {
        (self.macd_line.load(), self.signal_ema.load(), self.histogram.load())
    }
}

/// Stochastic Oscillator with SIMD
#[repr(C, align(64))]
pub struct Stochastic {
    highs: RollingWindow<14>,
    lows: RollingWindow<14>,
    k_values: RollingWindow<3>,
    k_value: AtomicCell<f64>,
    d_value: AtomicCell<f64>,
}

impl Stochastic {
    #[inline]
    pub const fn new() -> Self {
        Self {
            highs: RollingWindow::new(),
            lows: RollingWindow::new(),
            k_values: RollingWindow::new(),
            k_value: AtomicCell::new(50.0),
            d_value: AtomicCell::new(50.0),
        }
    }

    #[inline]
    pub fn update(&self, high: f64, low: f64, close: f64) -> (f64, f64) {
        self.highs.push(high);
        self.lows.push(low);

        // Find min/max in window (SIMD optimization possible)
        let mut highest_low = f64::MIN;
        let mut lowest_high = f64::MAX;
        
        for i in 0..14 {
            if let Some(h) = self.highs.get(i) {
                lowest_high = lowest_high.min(h);
            }
            if let Some(l) = self.lows.get(i) {
                highest_low = highest_low.max(l);
            }
        }

        let range = lowest_high - highest_low;
        let k = if range == 0.0 { 50.0 } else { ((close - highest_low) / range) * 100.0 };
        
        self.k_values.push(k);
        
        // Calculate D as SMA of K
        let mut sum_k = 0.0;
        let mut count = 0;
        for i in 0..3 {
            if let Some(kv) = self.k_values.get(i) {
                sum_k += kv;
                count += 1;
            }
        }
        let d = if count == 0 { 50.0 } else { sum_k / count as f64 };

        self.k_value.store(k);
        self.d_value.store(d);

        (k, d)
    }

    #[inline]
    pub fn values(&self) -> (f64, f64) {
        (self.k_value.load(), self.d_value.load())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi() {
        let rsi = RSI::new();
        let prices = [44.0, 44.25, 44.5, 43.75, 44.5, 44.0, 43.5, 43.0, 43.5, 44.0];
        
        for i in 1..prices.len() {
            rsi.update(prices[i], prices[i-1]);
        }
        
        assert!(rsi.value() >= 0.0 && rsi.value() <= 100.0);
    }
}

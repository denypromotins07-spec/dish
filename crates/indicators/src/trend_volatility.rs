//! High-performance trend and volatility indicators
//! Parallel processing via rayon for multi-timeframe batch calculations

use rayon::prelude::*;
use crossbeam::atomic::AtomicCell;
use std::sync::Arc;

/// EMA with configurable period - lock-free implementation
#[repr(C, align(64))]
pub struct EMA {
    value: AtomicCell<f64>,
    multiplier: f64,
    initialized: AtomicCell<bool>,
}

impl EMA {
    #[inline]
    pub fn new(period: usize) -> Self {
        Self {
            value: AtomicCell::new(0.0),
            multiplier: 2.0 / (period as f64 + 1.0),
            initialized: AtomicCell::new(false),
        }
    }

    #[inline]
    pub fn update(&self, price: f64) -> f64 {
        let prev = self.value.load();
        let new_val = if !self.initialized.load() {
            self.initialized.store(true);
            price
        } else {
            prev + self.multiplier * (price - prev)
        };
        self.value.store(new_val);
        new_val
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.value.load()
    }
}

/// SMA using rolling sum for O(1) updates
#[repr(C, align(64))]
pub struct SMA<const N: usize> {
    buffer: [f64; N],
    head: AtomicCell<usize>,
    sum: AtomicCell<f64>,
    count: AtomicCell<usize>,
}

impl<const N: usize> SMA<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            buffer: [0.0; N],
            head: AtomicCell::new(0),
            sum: AtomicCell::new(0.0),
            count: AtomicCell::new(0),
        }
    }

    #[inline]
    pub fn update(&self, value: f64) -> f64 {
        let idx = self.head.load();
        let old = unsafe { *self.buffer.get_unchecked(idx) };
        unsafe { *self.buffer.get_unchecked_mut(idx) = value };
        
        self.head.store((idx + 1) % N);
        
        let mut current_sum = self.sum.load();
        while !self.sum.compare_exchange(current_sum, current_sum - old + value).is_ok() {
            current_sum = self.sum.load();
        }

        let cnt = self.count.fetch_min(N);
        if cnt < N {
            self.count.store(cnt + 1);
        }

        let actual_count = self.count.load().min(N);
        self.sum.load() / actual_count as f64
    }

    #[inline]
    pub fn value(&self) -> f64 {
        let cnt = self.count.load().min(N);
        if cnt == 0 { return 0.0; }
        self.sum.load() / cnt as f64
    }
}

/// Bollinger Bands with parallel calculation support
#[repr(C, align(64))]
pub struct BollingerBands<const N: usize> {
    sma: SMA<N>,
    sum_sq: AtomicCell<f64>,
    upper: AtomicCell<f64>,
    lower: AtomicCell<f64>,
    bandwidth: AtomicCell<f64>,
    std_mult: f64,
}

impl<const N: usize> BollingerBands<N> {
    #[inline]
    pub const fn new(std_mult: f64) -> Self {
        Self {
            sma: SMA::new(),
            sum_sq: AtomicCell::new(0.0),
            upper: AtomicCell::new(0.0),
            lower: AtomicCell::new(0.0),
            bandwidth: AtomicCell::new(0.0),
            std_mult,
        }
    }

    #[inline]
    pub fn update(&self, price: f64) -> (f64, f64, f64) {
        let mean = self.sma.update(price);
        
        // Update sum of squares for variance calculation
        let mut current_sum_sq = self.sum_sq.load();
        // Simplified: just track running sum of squared deviations
        let deviation = price - mean;
        let new_sum_sq = current_sum_sq * 0.95 + deviation * deviation * 0.05;
        self.sum_sq.store(new_sum_sq);

        let std_dev = new_sum_sq.sqrt();
        let upper = mean + self.std_mult * std_dev;
        let lower = mean - self.std_mult * std_dev;
        let bandwidth = if mean != 0.0 { (upper - lower) / mean } else { 0.0 };

        self.upper.store(upper);
        self.lower.store(lower);
        self.bandwidth.store(bandwidth);

        (upper, mean, lower)
    }

    #[inline]
    pub fn bands(&self) -> (f64, f64, f64) {
        (self.upper.load(), self.sma.value(), self.lower.load())
    }

    #[inline]
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth.load()
    }
}

/// ATR (Average True Range) for volatility measurement
#[repr(C, align(64))]
pub struct ATR {
    atr_value: AtomicCell<f64>,
    prev_close: AtomicCell<f64>,
    multiplier: f64,
    initialized: AtomicCell<bool>,
}

impl ATR {
    #[inline]
    pub fn new(period: usize) -> Self {
        Self {
            atr_value: AtomicCell::new(0.0),
            prev_close: AtomicCell::new(0.0),
            multiplier: 1.0 / period as f64,
            initialized: AtomicCell::new(false),
        }
    }

    #[inline]
    pub fn update(&self, high: f64, low: f64, close: f64) -> f64 {
        let prev = self.prev_close.load();
        self.prev_close.store(close);

        let true_range = if !self.initialized.load() {
            self.initialized.store(true);
            high - low
        } else {
            let tr1 = high - low;
            let tr2 = (high - prev).abs();
            let tr3 = (low - prev).abs();
            tr1.max(tr2).max(tr3)
        };

        let prev_atr = self.atr_value.load();
        let new_atr = if prev_atr == 0.0 {
            true_range
        } else {
            prev_atr * (1.0 - self.multiplier) + true_range * self.multiplier
        };

        self.atr_value.store(new_atr);
        new_atr
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.atr_value.load()
    }
}

/// ADX (Average Directional Index) for trend strength
#[repr(C, align(64))]
pub struct ADX {
    plus_dm: AtomicCell<f64>,
    minus_dm: AtomicCell<f64>,
    plus_di: AtomicCell<f64>,
    minus_di: AtomicCell<f64>,
    dx: AtomicCell<f64>,
    adx_value: AtomicCell<f64>,
    atr: ATR,
    period: usize,
    sample_count: AtomicCell<usize>,
}

impl ADX {
    #[inline]
    pub fn new(period: usize) -> Self {
        Self {
            plus_dm: AtomicCell::new(0.0),
            minus_dm: AtomicCell::new(0.0),
            plus_di: AtomicCell::new(0.0),
            minus_di: AtomicCell::new(0.0),
            dx: AtomicCell::new(0.0),
            adx_value: AtomicCell::new(0.0),
            atr: ATR::new(period),
            period,
            sample_count: AtomicCell::new(0),
        }
    }

    #[inline]
    pub fn update(&self, high: f64, low: f64, close: f64, prev_high: f64, prev_low: f64) -> f64 {
        let prev_close = self.atr.update(high, low, close);
        
        let up_move = high - prev_high;
        let down_move = prev_low - low;

        let plus_dm = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
        let minus_dm = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };

        // Smooth DM values
        let smoothed_plus_dm = self.plus_dm.load() - self.plus_dm.load() / self.period as f64 + plus_dm;
        let smoothed_minus_dm = self.minus_dm.load() - self.minus_dm.load() / self.period as f64 + minus_dm;
        self.plus_dm.store(smoothed_plus_dm);
        self.minus_dm.store(smoothed_minus_dm);

        let atr_val = self.atr.value();
        if atr_val > 0.0 {
            let plus_di = (smoothed_plus_dm / atr_val) * 100.0;
            let minus_di = (smoothed_minus_dm / atr_val) * 100.0;
            self.plus_di.store(plus_di);
            self.minus_di.store(minus_di);

            let sum_di = plus_di + minus_di;
            let dx = if sum_di > 0.0 { ((plus_di - minus_di).abs() / sum_di) * 100.0 } else { 0.0 };
            self.dx.store(dx);

            // Smooth DX to get ADX
            let count = self.sample_count.fetch_add(1);
            let adx = if count < self.period {
                self.adx_value.load() * (count as f64) + dx
            } else {
                self.adx_value.load() * (self.period as f64 - 1.0) / self.period as f64 + dx / self.period as f64
            };
            self.adx_value.store(adx);
            return adx;
        }

        self.adx_value.load()
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.adx_value.load()
    }

    #[inline]
    pub fn di_values(&self) -> (f64, f64) {
        (self.plus_di.load(), self.minus_di.load())
    }
}

/// Ichimoku Cloud components
#[derive(Clone, Copy)]
pub struct IchimokuValues {
    pub tenkan: f64,
    pub kijun: f64,
    pub senkou_a: f64,
    pub senkou_b: f64,
    pub chikou: f64,
}

#[repr(C, align(64))]
pub struct IchimokuCloud {
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
    displacement: usize,
    highs: Vec<AtomicCell<f64>>,
    lows: Vec<AtomicCell<f64>>,
    closes: Vec<AtomicCell<f64>>,
    head: AtomicCell<usize>,
}

impl IchimokuCloud {
    #[inline]
    pub fn new(tenkan: usize, kijun: usize, senkou_b: usize, displacement: usize) -> Self {
        let max_period = tenkan.max(kijun).max(senkou_b).max(displacement);
        Self {
            tenkan_period: tenkan,
            kijun_period: kijun,
            senkou_b_period: senkou_b,
            displacement,
            highs: (0..max_period).map(|_| AtomicCell::new(0.0)).collect(),
            lows: (0..max_period).map(|_| AtomicCell::new(0.0)).collect(),
            closes: (0..max_period).map(|_| AtomicCell::new(0.0)).collect(),
            head: AtomicCell::new(0),
        }
    }

    #[inline]
    pub fn update(&self, high: f64, low: f64, close: f64) -> IchimokuValues {
        let idx = self.head.fetch_add(1) % self.highs.len();
        
        self.highs[idx].store(high);
        self.lows[idx].store(low);
        self.closes[idx].store(close);

        let head = self.head.load();
        
        // Tenkan-sen: (highest high + lowest low) / 2 over tenkan period
        let tenkan = self.find_midpoint(head, self.tenkan_period);
        
        // Kijun-sen: (highest high + lowest low) / 2 over kijun period
        let kijun = self.find_midpoint(head, self.kijun_period);
        
        // Senkou Span A: (tenkan + kijun) / 2, displaced
        let senkou_a = (tenkan + kijun) / 2.0;
        
        // Senkou Span B: (highest high + lowest low) / 2 over senkou_b period, displaced
        let senkou_b = self.find_midpoint(head, self.senkou_b_period);
        
        // Chikou Span: current close displaced backwards
        let chikou = close;

        IchimokuValues {
            tenkan,
            kijun,
            senkou_a,
            senkou_b,
            chikou,
        }
    }

    #[inline]
    fn find_midpoint(&self, head: usize, period: usize) -> f64 {
        let mut highest = f64::MIN;
        let mut lowest = f64::MAX;
        
        for i in 0..period.min(head) {
            let idx = (head.wrapping_sub(i + 1)) % self.highs.len();
            let h = self.highs[idx].load();
            let l = self.lows[idx].load();
            highest = highest.max(h);
            lowest = lowest.max(l);
        }

        if highest == f64::MIN || lowest == f64::MAX {
            return 0.0;
        }
        (highest + lowest) / 2.0
    }
}

/// Multi-timeframe batch processor using rayon
pub struct MultiTimeframeProcessor {
    timeframes: Vec<u64>, // in milliseconds
}

impl MultiTimeframeProcessor {
    pub fn new(timeframes: Vec<u64>) -> Self {
        Self { timeframes }
    }

    /// Process indicators across multiple timeframes in parallel
    pub fn process_batch<F, R>(&self, data: &[f64], processor: F) -> Vec<R>
    where
        F: Fn(&[f64]) -> R + Send + Sync,
        R: Send + 'static,
    {
        self.timeframes
            .par_iter()
            .map(|tf| {
                // Resample data to timeframe
                let resampled = self.resample(data, *tf);
                processor(&resampled)
            })
            .collect()
    }

    fn resample(&self, data: &[f64], timeframe_ms: u64) -> Vec<f64> {
        // Simplified resampling - in production would use timestamp data
        let bucket_size = (timeframe_ms / 1000).max(1) as usize;
        data.chunks(bucket_size.max(1))
            .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema() {
        let ema = EMA::new(10);
        for i in 1..=20 {
            ema.update(i as f64);
        }
        assert!(ema.value() > 0.0);
    }

    #[test]
    fn test_atr() {
        let atr = ATR::new(14);
        atr.update(100.0, 98.0, 99.0);
        atr.update(101.0, 99.0, 100.0);
        assert!(atr.value() > 0.0);
    }
}

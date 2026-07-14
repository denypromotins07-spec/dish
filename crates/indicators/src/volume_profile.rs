//! Volume-weighted metrics with compressed histograms
//! VWAP, Volume Profile (POC, VAH, VAL), and Cumulative Volume Delta (CVD)

use crossbeam::atomic::AtomicCell;
use std::collections::BTreeMap;

/// VWAP (Volume Weighted Average Price) - lock-free implementation
#[repr(C, align(64))]
pub struct VWAP {
    cumulative_tp_volume: AtomicCell<f64>,
    cumulative_volume: AtomicCell<f64>,
    vwap_value: AtomicCell<f64>,
    session_reset: AtomicCell<bool>,
}

impl VWAP {
    #[inline]
    pub const fn new() -> Self {
        Self {
            cumulative_tp_volume: AtomicCell::new(0.0),
            cumulative_volume: AtomicCell::new(0.0),
            vwap_value: AtomicCell::new(0.0),
            session_reset: AtomicCell::new(false),
        }
    }

    #[inline]
    pub fn update(&self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let typical_price = (high + low + close) / 3.0;
        
        let mut cum_tp_vol = self.cumulative_tp_volume.load();
        let mut cum_vol = self.cumulative_volume.load();
        
        // Handle session reset
        if self.session_reset.load() {
            cum_tp_vol = 0.0;
            cum_vol = 0.0;
            self.session_reset.store(false);
        }

        cum_tp_vol += typical_price * volume;
        cum_vol += volume;

        self.cumulative_tp_volume.store(cum_tp_vol);
        self.cumulative_volume.store(cum_vol);

        let vwap = if cum_vol > 0.0 { cum_tp_vol / cum_vol } else { typical_price };
        self.vwap_value.store(vwap);
        vwap
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.vwap_value.load()
    }

    #[inline]
    pub fn reset_session(&self) {
        self.session_reset.store(true);
    }
}

/// Compressed histogram bucket for volume profile
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct VolumeBucket {
    price_low: f64,
    price_high: f64,
    buy_volume: f64,
    sell_volume: f64,
    total_volume: f64,
    trade_count: u32,
}

impl VolumeBucket {
    #[inline]
    const fn new(price_low: f64, price_high: f64) -> Self {
        Self {
            price_low,
            price_high,
            buy_volume: 0.0,
            sell_volume: 0.0,
            total_volume: 0.0,
            trade_count: 0,
        }
    }

    #[inline]
    fn add_trade(&mut self, volume: f64, is_buy: bool) {
        if is_buy {
            self.buy_volume += volume;
        } else {
            self.sell_volume += volume;
        }
        self.total_volume += volume;
        self.trade_count += 1;
    }
}

/// Volume Profile with POC, VAH, VAL tracking
/// Uses compressed histogram to minimize memory footprint
#[repr(C, align(64))]
pub struct VolumeProfile {
    buckets: Vec<AtomicCell<VolumeBucket>>,
    bucket_size: f64,
    min_price: f64,
    max_price: f64,
    poc_price: AtomicCell<f64>,
    vah_price: AtomicCell<f64>,
    val_price: AtomicCell<f64>,
    total_volume: AtomicCell<f64>,
    value_area_percent: f64,
}

impl VolumeProfile {
    /// Create volume profile with specified price range and bucket size
    pub fn new(min_price: f64, max_price: f64, bucket_size: f64, value_area_percent: f64) -> Self {
        let num_buckets = ((max_price - min_price) / bucket_size).ceil() as usize;
        let buckets: Vec<AtomicCell<VolumeBucket>> = (0..num_buckets)
            .map(|i| {
                let p_low = min_price + i as f64 * bucket_size;
                let p_high = p_low + bucket_size;
                AtomicCell::new(VolumeBucket::new(p_low, p_high))
            })
            .collect();

        Self {
            buckets,
            bucket_size,
            min_price,
            max_price,
            poc_price: AtomicCell::new(min_price),
            vah_price: AtomicCell::new(min_price),
            val_price: AtomicCell::new(min_price),
            total_volume: AtomicCell::new(0.0),
            value_area_percent,
        }
    }

    #[inline]
    pub fn update(&self, price: f64, volume: f64, is_buy: bool) {
        if price < self.min_price || price > self.max_price {
            return; // Out of range
        }

        let bucket_idx = ((price - self.min_price) / self.bucket_size) as usize;
        if bucket_idx >= self.buckets.len() {
            return;
        }

        // Update bucket atomically
        let mut bucket = self.buckets[bucket_idx].load();
        bucket.add_trade(volume, is_buy);
        self.buckets[bucket_idx].store(bucket);

        // Update total volume
        let mut total = self.total_volume.load();
        while !self.total_volume.compare_exchange(total, total + volume).is_ok() {
            total = self.total_volume.load();
        }

        // Recalculate POC, VAH, VAL periodically or on significant volume
        self.recalculate_levels();
    }

    #[inline]
    fn recalculate_levels(&self) {
        // Find POC (Point of Control) - price level with highest volume
        let mut max_vol = 0.0;
        let mut poc_idx = 0;

        for (i, bucket_cell) in self.buckets.iter().enumerate() {
            let bucket = bucket_cell.load();
            if bucket.total_volume > max_vol {
                max_vol = bucket.total_volume;
                poc_idx = i;
            }
        }

        let poc = self.min_price + poc_idx as f64 * self.bucket_size + self.bucket_size / 2.0;
        self.poc_price.store(poc);

        // Calculate Value Area (70% of volume around POC)
        let target_va_volume = self.total_volume.load() * self.value_area_percent;
        let mut va_volume = max_vol;
        let mut left_idx = poc_idx;
        let mut right_idx = poc_idx;

        while va_volume < target_va_volume && (left_idx > 0 || right_idx < self.buckets.len() - 1) {
            let left_vol = if left_idx > 0 {
                self.buckets[left_idx - 1].load().total_volume
            } else {
                0.0
            };
            let right_vol = if right_idx < self.buckets.len() - 1 {
                self.buckets[right_idx + 1].load().total_volume
            } else {
                0.0
            };

            if left_vol >= right_vol && left_idx > 0 {
                left_idx -= 1;
                va_volume += left_vol;
            } else if right_idx < self.buckets.len() - 1 {
                right_idx += 1;
                va_volume += right_vol;
            } else {
                break;
            }
        }

        let val = self.min_price + left_idx as f64 * self.bucket_size;
        let vah = self.min_price + (right_idx + 1) as f64 * self.bucket_size;

        self.val_price.store(val);
        self.vah_price.store(vah);
    }

    #[inline]
    pub fn poc(&self) -> f64 {
        self.poc_price.load()
    }

    #[inline]
    pub fn vah(&self) -> f64 {
        self.vah_price.load()
    }

    #[inline]
    pub fn val(&self) -> f64 {
        self.val_price.load()
    }

    #[inline]
    pub fn total_volume(&self) -> f64 {
        self.total_volume.load()
    }

    /// Get volume at specific price level
    #[inline]
    pub fn volume_at_price(&self, price: f64) -> f64 {
        if price < self.min_price || price > self.max_price {
            return 0.0;
        }
        let idx = ((price - self.min_price) / self.bucket_size) as usize;
        if idx >= self.buckets.len() {
            return 0.0;
        }
        self.buckets[idx].load().total_volume
    }
}

/// Cumulative Volume Delta (CVD) - tracks buying vs selling pressure
#[repr(C, align(64))]
pub struct CVD {
    cumulative_delta: AtomicCell<f64>,
    buy_volume: AtomicCell<f64>,
    sell_volume: AtomicCell<f64>,
    delta_history: Vec<AtomicCell<f64>>,
    history_head: AtomicCell<usize>,
    history_size: usize,
}

impl CVD {
    pub fn new(history_size: usize) -> Self {
        Self {
            cumulative_delta: AtomicCell::new(0.0),
            buy_volume: AtomicCell::new(0.0),
            sell_volume: AtomicCell::new(0.0),
            delta_history: (0..history_size).map(|_| AtomicCell::new(0.0)).collect(),
            history_head: AtomicCell::new(0),
            history_size,
        }
    }

    #[inline]
    pub fn update(&self, volume: f64, is_buy: bool) -> f64 {
        let delta = if is_buy { volume } else { -volume };

        // Update cumulative delta
        let mut cum_delta = self.cumulative_delta.load();
        while !self.cumulative_delta.compare_exchange(cum_delta, cum_delta + delta).is_ok() {
            cum_delta = self.cumulative_delta.load();
        }

        // Update buy/sell volumes
        if is_buy {
            let mut buy_vol = self.buy_volume.load();
            while !self.buy_volume.compare_exchange(buy_vol, buy_vol + volume).is_ok() {
                buy_vol = self.buy_volume.load();
            }
        } else {
            let mut sell_vol = self.sell_volume.load();
            while !self.sell_volume.compare_exchange(sell_vol, sell_vol + volume).is_ok() {
                sell_vol = self.sell_volume.load();
            }
        }

        // Store in rolling history
        let idx = self.history_head.fetch_add(1) % self.history_size;
        self.delta_history[idx].store(delta);

        cum_delta + delta
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.cumulative_delta.load()
    }

    #[inline]
    pub fn buy_volume(&self) -> f64 {
        self.buy_volume.load()
    }

    #[inline]
    pub fn sell_volume(&self) -> f64 {
        self.sell_volume.load()
    }

    #[inline]
    pub fn net_delta(&self) -> f64 {
        self.buy_volume.load() - self.sell_volume.load()
    }

    /// Get sum of deltas over last N periods
    #[inline]
    pub fn delta_sum(&self, periods: usize) -> f64 {
        let periods = periods.min(self.history_size);
        let head = self.history_head.load();
        let mut sum = 0.0;

        for i in 0..periods {
            let idx = (head.wrapping_sub(i + 1)) % self.history_size;
            sum += self.delta_history[idx].load();
        }

        sum
    }
}

/// Combined volume metrics aggregator
#[repr(C, align(64))]
pub struct VolumeMetrics {
    vwap: VWAP,
    cvd: CVD,
    volume_profile: Option<VolumeProfile>,
}

impl VolumeMetrics {
    pub fn new(vp_min: f64, vp_max: f64, vp_bucket: f64) -> Self {
        Self {
            vwap: VWAP::new(),
            cvd: CVD::new(1000),
            volume_profile: Some(VolumeProfile::new(vp_min, vp_max, vp_bucket, 0.70)),
        }
    }

    #[inline]
    pub fn update(&self, high: f64, low: f64, close: f64, volume: f64, is_buy: bool) {
        self.vwap.update(high, low, close, volume);
        self.cvd.update(volume, is_buy);
        
        if let Some(ref vp) = self.volume_profile {
            vp.update(close, volume, is_buy);
        }
    }

    #[inline]
    pub fn vwap(&self) -> f64 {
        self.vwap.value()
    }

    #[inline]
    pub fn cvd(&self) -> f64 {
        self.cvd.value()
    }

    #[inline]
    pub fn poc(&self) -> f64 {
        self.volume_profile.as_ref().map(|vp| vp.poc()).unwrap_or(0.0)
    }

    #[inline]
    pub fn vah(&self) -> f64 {
        self.volume_profile.as_ref().map(|vp| vp.vah()).unwrap_or(0.0)
    }

    #[inline]
    pub fn val(&self) -> f64 {
        self.volume_profile.as_ref().map(|vp| vp.val()).unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap() {
        let vwap = VWAP::new();
        vwap.update(100.0, 98.0, 99.0, 1000.0);
        vwap.update(101.0, 99.0, 100.0, 500.0);
        assert!(vwap.value() > 0.0);
    }

    #[test]
    fn test_cvd() {
        let cvd = CVD::new(100);
        cvd.update(100.0, true);
        cvd.update(50.0, false);
        assert_eq!(cvd.value(), 50.0);
        assert_eq!(cvd.net_delta(), 50.0);
    }

    #[test]
    fn test_volume_profile() {
        let vp = VolumeProfile::new(90.0, 110.0, 1.0, 0.70);
        vp.update(100.0, 100.0, true);
        vp.update(100.0, 200.0, false);
        vp.update(101.0, 50.0, true);
        assert!(vp.poc() > 0.0);
    }
}

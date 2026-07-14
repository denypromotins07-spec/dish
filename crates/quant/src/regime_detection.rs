//! Lightweight Hidden Markov Model for market regime detection
//! Classifies market regimes: high-volatility trend vs low-volatility range

use crossbeam::atomic::AtomicCell;

/// Number of hidden states (configurable, typically 2-4)
const NUM_STATES: usize = 3;

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegime {
    LowVolRange,
    HighVolRange,
    BullTrend,
    BearTrend,
    Transition,
}

impl MarketRegime {
    #[inline]
    pub fn from_state(state: usize) -> Self {
        match state {
            0 => MarketRegime::LowVolRange,
            1 => MarketRegime::HighVolRange,
            2 => MarketRegime::BullTrend,
            _ => MarketRegime::BearTrend,
        }
    }

    #[inline]
    pub fn is_trending(&self) -> bool {
        matches!(self, MarketRegime::BullTrend | MarketRegime::BearTrend)
    }

    #[inline]
    pub fn is_ranging(&self) -> bool {
        matches!(self, MarketRegime::LowVolRange | MarketRegime::HighVolRange)
    }

    #[inline]
    pub fn volatility_level(&self) -> f64 {
        match self {
            MarketRegime::LowVolRange => 0.3,
            MarketRegime::HighVolRange => 0.8,
            MarketRegime::BullTrend => 0.6,
            MarketRegime::BearTrend => 0.7,
            MarketRegime::Transition => 0.5,
        }
    }
}

/// HMM state probabilities
#[repr(C, align(64))]
pub struct HMMState {
    /// Emission means for each state
    emission_means: [f64; NUM_STATES],
    /// Emission variances for each state
    emission_vars: [f64; NUM_STATES],
    /// Transition probability matrix (flattened)
    transition_probs: [f64; NUM_STATES * NUM_STATES],
    /// Current state probabilities (belief state)
    state_probs: [AtomicCell<f64>; NUM_STATES],
    /// Most likely current state
    current_state: AtomicCell<usize>,
    /// Log-likelihood accumulator
    log_likelihood: AtomicCell<f64>,
}

impl HMMState {
    #[inline]
    pub fn new() -> Self {
        // Initialize with reasonable defaults for market regimes
        Self {
            emission_means: [0.0, 0.0, 0.001, -0.001], // Low vol, high vol, bull, bear
            emission_vars: [0.0001, 0.0004, 0.0002, 0.0003],
            transition_probs: [
                0.7, 0.2, 0.05, 0.05,  // From state 0
                0.2, 0.6, 0.1, 0.1,    // From state 1
                0.1, 0.1, 0.7, 0.1,    // From state 2
                0.1, 0.1, 0.1, 0.7,    // From state 3
            ],
            state_probs: [
                AtomicCell::new(0.25),
                AtomicCell::new(0.25),
                AtomicCell::new(0.25),
                AtomicCell::new(0.25),
            ],
            current_state: AtomicCell::new(0),
            log_likelihood: AtomicCell::new(0.0),
        }
    }

    /// Forward algorithm step - update belief state with new observation
    #[inline]
    pub fn forward_step(&self, observation: f64) {
        let mut new_probs = [0.0; NUM_STATES];
        let mut total = 0.0;

        // Compute emission probabilities and apply transition
        for j in 0..NUM_STATES {
            let emission_prob = self.gaussian_pdf(observation, j);
            let mut prob = 0.0;

            for i in 0..NUM_STATES {
                let prev_prob = self.state_probs[i].load();
                let trans_prob = self.transition_probs[i * NUM_STATES + j];
                prob += prev_prob * trans_prob;
            }

            new_probs[j] = prob * emission_prob;
            total += new_probs[j];
        }

        // Normalize
        if total > 0.0 {
            for j in 0..NUM_STATES {
                new_probs[j] /= total;
                self.state_probs[j].store(new_probs[j]);
            }

            // Update log-likelihood
            let ll = self.log_likelihood.load();
            self.log_likelihood.store(ll + total.ln());

            // Find most likely state
            let mut max_prob = 0.0;
            let mut max_state = 0;
            for j in 0..NUM_STATES {
                if new_probs[j] > max_prob {
                    max_prob = new_probs[j];
                    max_state = j;
                }
            }
            self.current_state.store(max_state);
        }
    }

    #[inline]
    fn gaussian_pdf(&self, x: f64, state: usize) -> f64 {
        let mean = self.emission_means[state];
        let var = self.emission_vars[state];
        
        if var <= 0.0 {
            return 0.0;
        }

        let coeff = 1.0 / (2.0 * std::f64::consts::PI * var).sqrt();
        let exponent = -((x - mean) * (x - mean)) / (2.0 * var);
        
        coeff * exponent.exp()
    }

    #[inline]
    pub fn current_regime(&self) -> MarketRegime {
        MarketRegime::from_state(self.current_state.load())
    }

    #[inline]
    pub fn state_probabilities(&self) -> [f64; NUM_STATES] {
        [
            self.state_probs[0].load(),
            self.state_probs[1].load(),
            self.state_probs[2].load(),
            self.state_probs[3].load(),
        ]
    }

    #[inline]
    pub fn regime_confidence(&self) -> f64 {
        let probs = self.state_probabilities();
        probs.iter().cloned().fold(0.0, f64::max)
    }

    /// Adapt emission parameters online (simplified EM step)
    #[inline]
    pub fn adapt(&mut self, observation: f64, learning_rate: f64) {
        let probs = self.state_probabilities();
        let current = self.current_state.load();

        // Update emission parameters for the most likely state
        let old_mean = self.emission_means[current];
        let old_var = self.emission_vars[current];

        self.emission_means[current] = old_mean + learning_rate * (observation - old_mean);
        
        let diff_sq = (observation - old_mean) * (observation - old_mean);
        self.emission_vars[current] = old_var + learning_rate * (diff_sq - old_var);

        // Ensure variance stays positive
        self.emission_vars[current] = self.emission_vars[current].max(0.00001);
    }
}

/// Regime detection engine with multiple observations
#[repr(C, align(64))]
pub struct RegimeDetector {
    hmm: HMMState,
    returns_buffer: [AtomicCell<f64>; 20],
    returns_head: AtomicCell<usize>,
    prev_price: AtomicCell<f64>,
    regime_changes: AtomicCell<u32>,
    last_regime: AtomicCell<usize>,
    smoothing_window: usize,
}

impl RegimeDetector {
    #[inline]
    pub fn new(smoothing_window: usize) -> Self {
        Self {
            hmm: HMMState::new(),
            returns_buffer: std::array::from_fn(|_| AtomicCell::new(0.0)),
            returns_head: AtomicCell::new(0),
            prev_price: AtomicCell::new(0.0),
            regime_changes: AtomicCell::new(0),
            last_regime: AtomicCell::new(0),
            smoothing_window,
        }
    }

    /// Update with new price and detect regime
    #[inline]
    pub fn update(&self, price: f64) -> MarketRegime {
        let prev = self.prev_price.load();
        self.prev_price.store(price);

        if prev == 0.0 {
            return self.hmm.current_regime();
        }

        // Calculate log return
        let log_return = (price / prev).ln();

        // Store return in buffer
        let idx = self.returns_head.fetch_add(1) % 20;
        self.returns_buffer[idx].store(log_return);

        // Use smoothed return for HMM
        let smoothed_return = self.compute_smoothed_return();

        // Update HMM
        self.hmm.forward_step(smoothed_return);

        // Get current regime
        let regime = self.hmm.current_regime();
        let regime_idx = self.hmm.current_state.load();

        // Track regime changes
        let last = self.last_regime.load();
        if regime_idx != last && last != 0 {
            self.regime_changes.fetch_add(1);
        }
        self.last_regime.store(regime_idx);

        // Adapt model online
        let confidence = self.hmm.regime_confidence();
        if confidence > 0.5 {
            // Safe to adapt when confident
            let mut hmm_mut = unsafe { &mut *(std::ptr::from_ref(&self.hmm) as *mut HMMState) };
            hmm_mut.adapt(smoothed_return, 0.01);
        }

        regime
    }

    #[inline]
    fn compute_smoothed_return(&self) -> f64 {
        let head = self.returns_head.load();
        let count = head.min(self.smoothing_window);
        
        if count == 0 {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..count {
            let idx = (head.wrapping_sub(i + 1)) % 20;
            sum += self.returns_buffer[idx].load();
        }

        sum / count as f64
    }

    #[inline]
    pub fn current_regime(&self) -> MarketRegime {
        self.hmm.current_regime()
    }

    #[inline]
    pub fn regime_confidence(&self) -> f64 {
        self.hmm.regime_confidence()
    }

    #[inline]
    pub fn state_probabilities(&self) -> [f64; NUM_STATES] {
        self.hmm.state_probabilities()
    }

    #[inline]
    pub fn regime_change_count(&self) -> u32 {
        self.regime_changes.load()
    }

    #[inline]
    pub fn is_transitioning(&self) -> bool {
        let confidence = self.hmm.regime_confidence();
        confidence < 0.5 // Low confidence indicates transition
    }
}

/// Volatility regime classifier (simpler alternative to full HMM)
#[repr(C, align(64))]
pub struct VolatilityRegimeClassifier {
    short_vol: AtomicCell<f64>,
    long_vol: AtomicCell<f64>,
    threshold_ratio: f64,
    current_regime: AtomicCell<MarketRegime>,
}

impl VolatilityRegimeClassifier {
    #[inline]
    pub fn new(short_window: usize, long_window: usize, threshold_ratio: f64) -> Self {
        Self {
            short_vol: AtomicCell::new(0.0),
            long_vol: AtomicCell::new(0.0),
            threshold_ratio,
            current_regime: AtomicCell::new(MarketRegime::LowVolRange),
        }
    }

    #[inline]
    pub fn update(&self, return_val: f64) -> MarketRegime {
        // Exponential moving average of volatility
        let short_alpha = 0.2;
        let long_alpha = 0.05;

        let abs_ret = return_val.abs();
        
        let short_v = self.short_vol.load();
        let long_v = self.long_vol.load();

        let new_short = short_v * (1.0 - short_alpha) + abs_ret * short_alpha;
        let new_long = long_v * (1.0 - long_alpha) + abs_ret * long_alpha;

        self.short_vol.store(new_short);
        self.long_vol.store(new_long);

        // Classify regime based on volatility ratio and level
        let ratio = if new_long > 0.0 { new_short / new_long } else { 1.0 };
        
        let regime = if new_long < 0.005 {
            MarketRegime::LowVolRange
        } else if ratio > self.threshold_ratio {
            // Short vol spiking relative to long - potential regime change
            MarketRegime::Transition
        } else if new_long > 0.02 {
            MarketRegime::HighVolRange
        } else {
            MarketRegime::LowVolRange
        };

        self.current_regime.store(regime);
        regime
    }

    #[inline]
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime.load()
    }

    #[inline]
    pub fn volatility_ratio(&self) -> f64 {
        let short = self.short_vol.load();
        let long = self.long_vol.load();
        if long == 0.0 { return 1.0; }
        short / long
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_basic() {
        let hmm = HMMState::new();
        
        // Feed some observations
        for i in 0..50 {
            let obs = (i as f64 * 0.01).sin() * 0.01;
            hmm.forward_step(obs);
        }

        let regime = hmm.current_regime();
        assert!(matches!(regime, MarketRegime::_));
    }

    #[test]
    fn test_regime_detector() {
        let detector = RegimeDetector::new(10);
        
        // Simulate price series
        let mut price = 100.0;
        for i in 0..100 {
            price *= 1.0 + (i as f64 * 0.01).sin() * 0.001;
            detector.update(price);
        }

        let regime = detector.current_regime();
        let confidence = detector.regime_confidence();
        
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_volatility_classifier() {
        let classifier = VolatilityRegimeClassifier::new(20, 50, 1.5);
        
        // Low volatility period
        for _ in 0..50 {
            classifier.update(0.001);
        }
        assert_eq!(classifier.current_regime(), MarketRegime::LowVolRange);
        
        // High volatility period
        for _ in 0..50 {
            classifier.update(0.03);
        }
        // Should eventually transition to high vol
    }
}

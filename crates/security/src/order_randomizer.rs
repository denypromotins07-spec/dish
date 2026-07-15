//! Microsecond execution randomizer that adds imperceptible, randomized time delays (jitter)
//! and size variations to TWAP/VWAP child orders.
//! Prevents algorithmic predators from reverse-engineering and front-running the parent order.

use std::time::{Duration, Instant};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};

/// Configuration for order randomization.
#[derive(Clone, Debug)]
pub struct OrderRandomizerConfig {
    /// Minimum time jitter in microseconds
    pub min_jitter_us: u64,
    /// Maximum time jitter in microseconds
    pub max_jitter_us: u64,
    /// Minimum size variation as a fraction (e.g., 0.01 = 1%)
    pub min_size_variation: f64,
    /// Maximum size variation as a fraction (e.g., 0.05 = 5%)
    pub max_size_variation: f64,
    /// Whether to use Gaussian distribution for more natural-looking variations
    pub use_gaussian: bool,
    /// Seed for deterministic testing (None for random seed)
    pub seed: Option<u64>,
}

impl Default for OrderRandomizerConfig {
    fn default() -> Self {
        OrderRandomizerConfig {
            min_jitter_us: 100,      // 100 microseconds
            max_jitter_us: 50000,    // 50 milliseconds
            min_size_variation: 0.001, // 0.1%
            max_size_variation: 0.02,  // 2%
            use_gaussian: true,
            seed: None,
        }
    }
}

/// Represents a randomized child order.
#[derive(Clone, Debug)]
pub struct RandomizedOrder {
    pub original_sequence: usize,
    pub randomized_size: f64,
    pub delay_duration: Duration,
    pub price_offset_bps: Option<f64>,
    pub execution_id: String,
}

/// Microsecond-precision order randomizer for MEV protection.
pub struct OrderRandomizer {
    config: OrderRandomizerConfig,
    rng: Box<dyn RngCore + Send>,
    sequence_counter: usize,
}

use rand::RngCore;

impl OrderRandomizer {
    /// Creates a new order randomizer with the given configuration.
    pub fn new(config: OrderRandomizerConfig) -> Self {
        let rng: Box<dyn RngCore + Send> = if let Some(seed) = config.seed {
            Box::new(rand::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(rand::thread_rng())
        };

        OrderRandomizer {
            config,
            rng,
            sequence_counter: 0,
        }
    }

    /// Applies randomization to a TWAP/VWAP child order.
    pub fn randomize_order(
        &mut self,
        original_size: f64,
        sequence: usize,
    ) -> RandomizedOrder {
        self.sequence_counter += 1;

        // Generate time jitter
        let jitter_us = self.generate_jitter_microseconds();
        let delay_duration = Duration::from_micros(jitter_us);

        // Generate size variation
        let randomized_size = self.apply_size_variation(original_size);

        // Optionally generate small price offset for limit orders
        let price_offset_bps = if self.config.use_gaussian {
            Some(self.generate_price_offset())
        } else {
            None
        };

        // Generate unique execution ID
        let execution_id = format!(
            "ORD-{}-{:08x}",
            sequence,
            self.rng.gen::<u32>()
        );

        RandomizedOrder {
            original_sequence: sequence,
            randomized_size,
            delay_duration,
            price_offset_bps,
            execution_id,
        }
    }

    /// Generates a batch of randomized orders for a parent order.
    pub fn randomize_batch(
        &mut self,
        total_size: f64,
        num_child_orders: usize,
    ) -> Vec<RandomizedOrder> {
        let base_size = total_size / num_child_orders as f64;
        let mut orders = Vec::with_capacity(num_child_orders);
        let mut remaining_size = total_size;

        for i in 0..num_child_orders {
            let current_base = if i == num_child_orders - 1 {
                // Last order takes remaining to ensure exact total
                remaining_size
            } else {
                base_size
            };

            let mut order = self.randomize_order(current_base, i);

            // Adjust for cumulative rounding errors
            if i == num_child_orders - 1 {
                order.randomized_size = remaining_size;
            }

            remaining_size -= order.randomized_size;
            orders.push(order);
        }

        // Shuffle order sequence slightly to prevent pattern detection
        self.shuffle_execution_order(&mut orders);

        orders
    }

    /// Waits for the specified delay with microsecond precision.
    pub async fn wait_for_execution(&self, order: &RandomizedOrder) {
        tokio::time::sleep(order.delay_duration).await;
    }

    /// Generates random time jitter in microseconds.
    fn generate_jitter_microseconds(&mut self) -> u64 {
        let between = Uniform::new(self.config.min_jitter_us, self.config.max_jitter_us);
        
        if self.config.use_gaussian {
            // Use Gaussian for more natural distribution centered around midpoint
            let midpoint = (self.config.min_jitter_us + self.config.max_jitter_us) / 2;
            let stddev = (self.config.max_jitter_us - self.config.min_jitter_us) / 6;
            
            let mut rng = &mut self.rng;
            let gaussian = rand_distr::Normal::new(midpoint as f64, stddev as f64).unwrap();
            let value = gaussian.sample(&mut rng);
            
            value.clamp(self.config.min_jitter_us as f64, self.config.max_jitter_us as f64) as u64
        } else {
            between.sample(&mut self.rng)
        }
    }

    /// Applies random size variation while staying within bounds.
    fn apply_size_variation(&mut self, original_size: f64) -> f64 {
        let variation = if self.config.use_gaussian {
            // Gaussian distribution centered at 0
            let stddev = (self.config.max_size_variation - self.config.min_size_variation) / 3;
            let gaussian = rand_distr::Normal::new(0.0, stddev).unwrap();
            gaussian.sample(&mut self.rng)
        } else {
            // Uniform distribution
            let between = Uniform::new(
                self.config.min_size_variation,
                self.config.max_size_variation,
            );
            let sign = if self.rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            between.sample(&mut self.rng) * sign
        };

        let varied_size = original_size * (1.0 + variation);
        
        // Ensure size stays positive and reasonable
        varied_size.max(original_size * 0.9).min(original_size * 1.1)
    }

    /// Generates a small price offset in basis points for limit orders.
    fn generate_price_offset(&mut self) -> f64 {
        // Small offset between -2 and +2 bps to avoid being too predictable
        let between = Uniform::new(-2.0, 2.0);
        between.sample(&mut self.rng)
    }

    /// Shuffles the execution order slightly to prevent pattern detection.
    fn shuffle_execution_order(&mut self, orders: &mut [RandomizedOrder]) {
        // Fisher-Yates shuffle with limited swaps to maintain some order
        let mut rng = &mut self.rng;
        let len = orders.len();
        
        // Only swap adjacent elements to maintain general TWAP/VWAP structure
        for i in 0..len.saturating_sub(1) {
            if rng.gen_bool(0.3) {
                orders.swap(i, i + 1);
            }
        }
    }

    /// Gets statistics about the randomization patterns (for monitoring).
    pub fn get_randomization_stats(&self) -> RandomizationStats {
        RandomizationStats {
            sequence_count: self.sequence_counter,
            config: self.config.clone(),
        }
    }
}

/// Statistics about randomization patterns.
#[derive(Clone, Debug)]
pub struct RandomizationStats {
    pub sequence_count: usize,
    pub config: OrderRandomizerConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_randomization() {
        let config = OrderRandomizerConfig {
            min_jitter_us: 100,
            max_jitter_us: 1000,
            min_size_variation: 0.01,
            max_size_variation: 0.05,
            use_gaussian: false,
            seed: Some(42), // Fixed seed for reproducibility
        };

        let mut randomizer = OrderRandomizer::new(config);
        let original_size = 1.0;
        
        let order = randomizer.randomize_order(original_size, 0);
        
        // Verify size is within expected bounds
        assert!(order.randomized_size >= original_size * 0.95);
        assert!(order.randomized_size <= original_size * 1.05);
        
        // Verify delay is within bounds
        assert!(order.delay_duration.as_micros() >= 100);
        assert!(order.delay_duration.as_micros() <= 1000);
        
        // Verify execution ID is generated
        assert!(order.execution_id.starts_with("ORD-0-"));
    }

    #[test]
    fn test_batch_randomization_preserves_total() {
        let config = OrderRandomizerConfig {
            min_jitter_us: 100,
            max_jitter_us: 500,
            min_size_variation: 0.01,
            max_size_variation: 0.03,
            use_gaussian: false,
            seed: Some(123),
        };

        let mut randomizer = OrderRandomizer::new(config);
        let total_size = 10.0;
        let num_orders = 5;
        
        let orders = randomizer.randomize_batch(total_size, num_orders);
        
        assert_eq!(orders.len(), num_orders);
        
        // Total should be preserved (within floating point tolerance)
        let total: f64 = orders.iter().map(|o| o.randomized_size).sum();
        assert!((total - total_size).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_execution_delay() {
        let config = OrderRandomizerConfig {
            min_jitter_us: 1000,
            max_jitter_us: 2000,
            min_size_variation: 0.01,
            max_size_variation: 0.02,
            use_gaussian: false,
            seed: Some(456),
        };

        let mut randomizer = OrderRandomizer::new(config);
        let order = randomizer.randomize_order(1.0, 0);
        
        let start = Instant::now();
        randomizer.wait_for_execution(&order).await;
        let elapsed = start.elapsed();
        
        // Verify delay is approximately correct (with some tolerance)
        assert!(elapsed >= Duration::from_micros(800));
        assert!(elapsed <= Duration::from_micros(2500));
    }
}

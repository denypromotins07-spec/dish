//! Latency Profile Generator - Generates realistic log-normal latency distributions.
//! Based on historical cross-exchange ping times for synthetic delay injection.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Exchange latency profile
#[derive(Debug, Clone)]
pub struct ExchangeLatencyProfile {
    pub exchange_name: String,
    pub mean_latency_us: f64,    // Mean latency in microseconds
    pub std_dev_us: f64,         // Standard deviation
    pub min_latency_us: u64,     // Minimum observed latency
    pub max_latency_us: u64,     // Maximum observed latency
    pub percentile_99_us: u64,   // 99th percentile
    pub percentile_999_us: u64,  // 99.9th percentile
}

impl Default for ExchangeLatencyProfile {
    fn default() -> Self {
        Self {
            exchange_name: "unknown".to_string(),
            mean_latency_us: 5000.0,
            std_dev_us: 2000.0,
            min_latency_us: 1000,
            max_latency_us: 50000,
            percentile_99_us: 15000,
            percentile_999_us: 30000,
        }
    }
}

/// Pre-built profiles for major exchanges
pub mod exchange_profiles {
    use super::*;

    pub fn binance() -> ExchangeLatencyProfile {
        ExchangeLatencyProfile {
            exchange_name: "binance".to_string(),
            mean_latency_us: 3500.0,
            std_dev_us: 1200.0,
            min_latency_us: 800,
            max_latency_us: 25000,
            percentile_99_us: 10000,
            percentile_999_us: 18000,
        }
    }

    pub fn bybit() -> ExchangeLatencyProfile {
        ExchangeLatencyProfile {
            exchange_name: "bybit".to_string(),
            mean_latency_us: 4200.0,
            std_dev_us: 1500.0,
            min_latency_us: 1000,
            max_latency_us: 30000,
            percentile_99_us: 12000,
            percentile_999_us: 22000,
        }
    }

    pub fn okx() -> ExchangeLatencyProfile {
        ExchangeLatencyProfile {
            exchange_name: "okx".to_string(),
            mean_latency_us: 5500.0,
            std_dev_us: 2000.0,
            min_latency_us: 1200,
            max_latency_us: 40000,
            percentile_99_us: 15000,
            percentile_999_us: 28000,
        }
    }

    pub fn deribit() -> ExchangeLatencyProfile {
        ExchangeLatencyProfile {
            exchange_name: "deribit".to_string(),
            mean_latency_us: 8000.0,
            std_dev_us: 3000.0,
            min_latency_us: 2000,
            max_latency_us: 60000,
            percentile_99_us: 22000,
            percentile_999_us: 40000,
        }
    }
}

/// Log-normal distribution sampler for latency generation
pub struct LatencyProfileGenerator {
    profile: ExchangeLatencyProfile,
    rng: SmallRng,
    samples_generated: AtomicU64,
    active: AtomicBool,
}

impl LatencyProfileGenerator {
    pub fn new(profile: ExchangeLatencyProfile) -> Self {
        Self {
            profile,
            rng: SmallRng::from_entropy(),
            samples_generated: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Generate a single latency sample from log-normal distribution
    #[inline]
    pub fn generate_sample(&mut self) -> Duration {
        if !self.active.load(Ordering::Relaxed) {
            return Duration::ZERO;
        }

        // Box-Muller transform for normal distribution
        let u1: f64 = self.rng.gen::<f64>().max(1e-10);
        let u2: f64 = self.rng.gen::<f64>();
        
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        
        // Convert to log-normal
        let mean_log = self.profile.mean_latency_us.ln();
        let std_log = (1.0 + (self.profile.std_dev_us / self.profile.mean_latency_us).powi(2)).ln().sqrt();
        
        let latency_us = (mean_log + std_log * z).exp();
        
        // Clamp to observed bounds
        let latency_us = latency_us.clamp(
            self.profile.min_latency_us as f64,
            self.profile.max_latency_us as f64,
        );
        
        self.samples_generated.fetch_add(1, Ordering::Relaxed);
        
        Duration::from_micros(latency_us as u64)
    }

    /// Generate multiple samples for batch testing
    pub fn generate_samples(&mut self, count: usize) -> Vec<Duration> {
        (0..count).map(|_| self.generate_sample()).collect()
    }

    /// Generate sample with tail event probability
    #[inline]
    pub fn generate_with_tail_event(&mut self, tail_probability: f64) -> Duration {
        if self.rng.gen::<f64>() < tail_probability {
            // Generate tail event (extreme latency)
            let tail_multiplier = self.rng.gen_range(2.0..10.0);
            let base = self.generate_sample();
            Duration::from_micros((base.as_micros() as f64 * tail_multiplier) as u64)
        } else {
            self.generate_sample()
        }
    }

    /// Get statistics about generated samples
    pub fn get_stats(&self) -> GeneratorStats {
        GeneratorStats {
            profile_name: self.profile.exchange_name.clone(),
            mean_latency_us: self.profile.mean_latency_us,
            std_dev_us: self.profile.std_dev_us,
            min_latency_us: self.profile.min_latency_us,
            max_latency_us: self.profile.max_latency_us,
            samples_generated: self.samples_generated.load(Ordering::Relaxed),
        }
    }

    /// Update profile dynamically
    pub fn update_profile(&mut self, profile: ExchangeLatencyProfile) {
        self.profile = profile;
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct GeneratorStats {
    pub profile_name: String,
    pub mean_latency_us: f64,
    pub std_dev_us: f64,
    pub min_latency_us: u64,
    pub max_latency_us: u64,
    pub samples_generated: u64,
}

/// Latency injector for backtesting/shadow testing
pub struct LatencyInjector {
    generator: LatencyProfileGenerator,
    injected_count: AtomicU64,
    total_injected_us: AtomicU64,
}

impl LatencyInjector {
    pub fn new(profile: ExchangeLatencyProfile) -> Self {
        Self {
            generator: LatencyProfileGenerator::new(profile),
            injected_count: AtomicU64::new(0),
            total_injected_us: AtomicU64::new(0),
        }
    }

    /// Inject latency into an operation
    #[inline]
    pub fn inject<F, R>(&mut self, operation: F) -> R
    where
        F: FnOnce() -> R,
    {
        let latency = self.generator.generate_sample();
        
        if latency > Duration::ZERO {
            std::thread::sleep(latency);
            self.injected_count.fetch_add(1, Ordering::Relaxed);
            self.total_injected_us.fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
        }
        
        operation()
    }

    /// Get injection statistics
    pub fn get_stats(&self) -> InjectorStats {
        let injected = self.injected_count.load(Ordering::Relaxed);
        let total_us = self.total_injected_us.load(Ordering::Relaxed);
        
        InjectorStats {
            injected_count: injected,
            total_injected_us: total_us,
            avg_injected_us: if injected > 0 { total_us / injected } else { 0 },
            generator_stats: self.generator.get_stats(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InjectorStats {
    pub injected_count: u64,
    pub total_injected_us: u64,
    pub avg_injected_us: u64,
    pub generator_stats: GeneratorStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_generation() {
        let profile = exchange_profiles::binance();
        let mut generator = LatencyProfileGenerator::new(profile);

        let mut latencies = Vec::new();
        for _ in 0..1000 {
            latencies.push(generator.generate_sample().as_micros() as f64);
        }

        let mean: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;
        assert!(mean > 2000.0 && mean < 6000.0); // Should be close to Binance mean
    }

    #[test]
    fn test_tail_events() {
        let profile = exchange_profiles::binance();
        let mut generator = LatencyProfileGenerator::new(profile);

        let normal = generator.generate_sample();
        let tail = generator.generate_with_tail_event(1.0); // Force tail event

        assert!(tail > normal);
    }
}

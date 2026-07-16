// Chapter 2, File 3: MSVC Optimizer Build Configuration
// crates/hft/src/msvc_optimizer.rs
// Build configuration and compiler flags for MSVC + AMD Zen architecture

//! This module documents the build configuration required for optimal HFT performance
//! on Windows with MSVC compiler targeting AMD Ryzen AI (Zen 4) architecture.
//! 
//! Build command example:
//! ```powershell
//! $env:RUSTFLAGS = "-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort"
//! $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS = "-C target-feature=+avx2,+bmi2,+fma"
//! cargo build --release --target x86_64-pc-windows-msvc
//! ```

use std::env;
use std::path::PathBuf;

/// Compiler optimization settings for MSVC
pub struct MsvcOptimizerConfig {
    pub target_cpu: &'static str,
    pub optimization_level: u8,
    pub lto_enabled: bool,
    pub codegen_units: u32,
    pub panic_strategy: PanicStrategy,
    pub target_features: Vec<&'static str>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum PanicStrategy {
    Abort,
    Unwind,
}

impl Default for MsvcOptimizerConfig {
    fn default() -> Self {
        MsvcOptimizerConfig {
            target_cpu: "native",
            optimization_level: 3,
            lto_enabled: true,
            codegen_units: 1,
            panic_strategy: PanicStrategy::Abort,
            target_features: vec!["avx2", "bmi2", "fma", "lzcnt", "popcnt"],
        }
    }
}

impl MsvcOptimizerConfig {
    /// Generate RUSTFLAGS for optimal MSVC compilation
    pub fn generate_rustflags(&self) -> String {
        let mut flags = Vec::new();

        // Target CPU optimization
        flags.push(format!("-C target-cpu={}", self.target_cpu));

        // Optimization level
        flags.push(format!("-C opt-level={}", self.optimization_level));

        // Link Time Optimization
        if self.lto_enabled {
            flags.push("-C lto=fat".to_string());
        }

        // Codegen units (fewer = better optimization, slower compile)
        flags.push(format!("-C codegen-units={}", self.codegen_units));

        // Panic strategy (abort = smaller binaries, no unwind overhead)
        match self.panic_strategy {
            PanicStrategy::Abort => flags.push("-C panic=abort".to_string()),
            PanicStrategy::Unwind => flags.push("-C panic=unwind".to_string()),
        }

        // Target features for AMD Zen
        for feature in &self.target_features {
            flags.push(format!("-C target-feature=+{}", feature));
        }

        flags.join(" ")
    }

    /// Generate Cargo.toml profile settings
    pub fn generate_cargo_profile(&self) -> String {
        format!(
            r#"[profile.release]
opt-level = {}
lto = {}
codegen-units = {}
panic = "{}"
strip = true
debug = false
rpath = false
"#,
            self.optimization_level,
            if self.lto_enabled { "fat" } else { "thin" },
            self.codegen_units,
            match self.panic_strategy {
                PanicStrategy::Abort => "abort",
                PanicStrategy::Unwind => "unwind",
            },
        )
    }

    /// Apply environment variables for current session
    pub fn apply_to_environment(&self) {
        let rustflags = self.generate_rustflags();
        env::set_var("RUSTFLAGS", &rustflags);
        env::set_var(
            "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS",
            &rustflags,
        );
        println!("[MSVC_OPTIMIZER] Applied RUSTFLAGS: {}", rustflags);
    }

    /// Verify running on compatible hardware
    pub fn verify_hardware_compatibility() -> Result<(), String> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            unsafe {
                // Check for AVX2 support
                if !is_x86_feature_detected!("avx2") {
                    return Err("AVX2 not supported on this CPU".to_string());
                }
                
                // Check for BMI2 support
                if !is_x86_feature_detected!("bmi2") {
                    return Err("BMI2 not supported on this CPU".to_string());
                }
                
                // Check for FMA support
                if !is_x86_feature_detected!("fma") {
                    return Err("FMA not supported on this CPU".to_string());
                }
            }
            
            Ok(())
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            Err("This optimizer is only compatible with x86_64 architecture".to_string())
        }
    }
}

/// SIMD-optimized utilities for HFT calculations
pub mod simd_utils {
    use std::arch::x86_64::*;

    /// Vectorized price comparison using AVX2
    #[inline(always)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn compare_prices_avx2(prices_a: &[f64], prices_b: &[f64]) -> Vec<bool> {
        assert_eq!(prices_a.len(), prices_b.len());
        let len = prices_a.len();
        let mut results = Vec::with_capacity(len);

        let chunks_a = prices_a.chunks_exact(4);
        let chunks_b = prices_b.chunks_exact(4);
        let remainder = len % 4;

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let va = _mm256_loadu_pd(chunk_a.as_ptr());
            let vb = _mm256_loadu_pd(chunk_b.as_ptr());
            let cmp = _mm256_cmp_pd(va, vb, _CMP_LT_OQ);
            
            let mask = _mm256_movemask_pd(cmp) as u32;
            for i in 0..4 {
                results.push((mask & (1 << i)) != 0);
            }
        }

        // Handle remainder
        for i in (len - remainder)..len {
            results.push(prices_a[i] < prices_b[i]);
        }

        results
    }

    /// Vectorized sum using AVX2
    #[inline(always)]
    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_f64_avx2(values: &[f64]) -> f64 {
        let len = values.len();
        let mut sum = 0.0;

        let chunks = values.chunks_exact(4);
        let remainder = len % 4;

        let mut acc = _mm256_setzero_pd();

        for chunk in chunks {
            let v = _mm256_loadu_pd(chunk.as_ptr());
            acc = _mm256_add_pd(acc, v);
        }

        // Horizontal sum
        let hi = _mm256_permute2f128_pd(acc, acc, 0x1);
        let sum_hi = _mm256_add_pd(acc, hi);
        let lo = _mm256_unpackhi_pd(sum_hi, sum_hi);
        let sum_lo = _mm256_add_pd(sum_hi, lo);
        
        let mut temp = [0.0; 4];
        _mm256_storeu_pd(temp.as_mut_ptr(), sum_lo);
        sum += temp[0];

        // Handle remainder
        for i in (len - remainder)..len {
            sum += values[i];
        }

        sum
    }
}

/// Build script helper for generating optimized binaries
pub fn print_build_instructions() {
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:warning=Building with MSVC optimizations for AMD Zen 4");
    println!("cargo:warning=Ensure you have Visual Studio 2022 with C++ tools installed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = MsvcOptimizerConfig::default();
        assert_eq!(config.optimization_level, 3);
        assert!(config.lto_enabled);
        assert_eq!(config.codegen_units, 1);
        assert_eq!(config.panic_strategy, PanicStrategy::Abort);
    }

    #[test]
    fn test_rustflags_generation() {
        let config = MsvcOptimizerConfig::default();
        let flags = config.generate_rustflags();
        assert!(flags.contains("-C target-cpu=native"));
        assert!(flags.contains("-C opt-level=3"));
        assert!(flags.contains("-C lto=fat"));
        assert!(flags.contains("-C panic=abort"));
    }

    #[test]
    fn test_hardware_check() {
        // This will pass on CI with x86_64 and fail gracefully otherwise
        let result = MsvcOptimizerConfig::verify_hardware_compatibility();
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, we expect either success or specific feature errors
            println!("Hardware check result: {:?}", result);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_sum() {
        unsafe {
            if is_x86_feature_detected!("avx2") {
                let values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
                let sum = simd_utils::sum_f64_avx2(&values);
                assert!((sum - 5050.0).abs() < 0.001);
            }
        }
    }
}

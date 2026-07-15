//! Implied Volatility (IV) surface builder and skew tracker
//! Interpolates sparse options chain data into continuous 3D surface using cubic splines
//! Strictly bounds memory allocations for low-latency operation

use std::collections::BTreeMap;

/// Single volatility point in the surface
#[derive(Debug, Clone, Copy)]
pub struct VolPoint {
    pub strike: f64,
    pub expiry_days: u32,
    pub implied_vol: f64,
    pub bid_iv: f64,
    pub ask_iv: f64,
    pub volume: u64,
    pub open_interest: u64,
}

/// Cubic spline interpolation for smooth IV surface construction
struct CubicSpline {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSpline {
    /// Build natural cubic spline from points
    fn new(points: &[(f64, f64)]) -> Self {
        let n = points.len();
        if n < 2 {
            return Self {
                x: vec![points.first().map(|p| p.0).unwrap_or(0.0)],
                a: vec![points.first().map(|p| p.1).unwrap_or(0.0)],
                b: vec![0.0],
                c: vec![0.0],
                d: vec![0.0],
            };
        }

        let x: Vec<f64> = points.iter().map(|p| p.0).collect();
        let y: Vec<f64> = points.iter().map(|p| p.1).collect();

        // Natural spline boundary conditions
        let mut h: Vec<f64> = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            h.push(x[i + 1] - x[i]);
        }

        // Build tridiagonal system
        let mut alpha: Vec<f64> = vec![0.0];
        for i in 1..n - 1 {
            let val = 3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1]);
            alpha.push(val);
        }
        alpha.push(0.0);

        let mut l = vec![1.0];
        let mut mu = vec![0.0];
        let mut z = vec![0.0];

        for i in 1..n - 1 {
            let li = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            l.push(li);
            mu.push(h[i] / li);
            z.push((alpha[i] - h[i - 1] * z[i - 1]) / li);
        }

        l.push(1.0);
        z.push(0.0);

        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n - 1];
        let mut d = vec![0.0; n - 1];

        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        Self {
            x,
            a: y,
            b,
            c,
            d,
        }
    }

    /// Evaluate spline at given x
    #[inline]
    fn evaluate(&self, x_val: f64) -> f64 {
        if self.x.is_empty() {
            return 0.0;
        }

        // Find appropriate interval using binary search
        let idx = match self.x.binary_search_by(|&v| v.partial_cmp(&x_val).unwrap()) {
            Ok(i) => i.min(self.x.len() - 2),
            Err(i) => {
                if i == 0 {
                    0
                } else if i >= self.x.len() {
                    self.x.len() - 2
                } else {
                    i - 1
                }
            }
        };

        let dx = x_val - self.x[idx];
        self.a[idx] + self.b[idx] * dx + self.c[idx] * dx * dx + self.d[idx] * dx * dx * dx
    }
}

/// Implied Volatility Surface with bounded memory
pub struct IVSurface {
    /// Map of expiry_days -> spline for that expiry
    expiry_splines: BTreeMap<u32, CubicSpline>,
    /// Strike range bounds
    min_strike: f64,
    max_strike: f64,
    /// Memory-bounded cache size
    max_points_per_expiry: usize,
    /// Skew metrics
    atm_iv: f64,
    risk_reversal_25d: f64,
    butterfly_25d: f64,
}

impl IVSurface {
    pub fn new(max_points_per_expiry: usize) -> Self {
        Self {
            expiry_splines: BTreeMap::new(),
            min_strike: 0.0,
            max_strike: f64::MAX,
            max_points_per_expiry,
            atm_iv: 0.0,
            risk_reversal_25d: 0.0,
            butterfly_25d: 0.0,
        }
    }

    /// Build IV surface from raw volatility points
    pub fn build_surface(&mut self, mut points: Vec<VolPoint>) {
        // Group by expiry
        let mut by_expiry: BTreeMap<u32, Vec<(f64, f64)>> = BTreeMap::new();

        for point in points.iter() {
            if point.strike < self.min_strike || point.strike > self.max_strike {
                continue;
            }

            by_expiry
                .entry(point.expiry_days)
                .or_insert_with(Vec::new)
                .push((point.strike, point.implied_vol));
        }

        // Sort and limit points per expiry to bound memory
        for (_, strikes) in by_expiry.iter_mut() {
            strikes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            if strikes.len() > self.max_points_per_expiry {
                // Keep evenly spaced points
                let step = strikes.len() / self.max_points_per_expiry;
                *strikes = strikes
                    .iter()
                    .step_by(step)
                    .take(self.max_points_per_expiry)
                    .copied()
                    .collect();
            }
        }

        // Build splines for each expiry
        self.expiry_splines.clear();
        for (expiry, strikes) in by_expiry {
            self.expiry_splines.insert(expiry, CubicSpline::new(&strikes));
        }

        // Calculate skew metrics
        self.calculate_skew_metrics(&points);
    }

    /// Get interpolated IV for given strike and expiry
    #[inline]
    pub fn get_iv(&self, strike: f64, expiry_days: u32) -> Option<f64> {
        if strike < self.min_strike || strike > self.max_strike {
            return None;
        }

        // Find nearest expiries
        let expiries: Vec<u32> = self.expiry_splines.keys().copied().collect();
        if expiries.is_empty() {
            return None;
        }

        match expiries.binary_search(&expiry_days) {
            Ok(idx) => {
                // Exact match
                self.expiry_splines.get(&expiries[idx]).map(|s| s.evaluate(strike))
            }
            Err(idx) => {
                // Interpolate between expiries
                if idx == 0 {
                    self.expiry_splines.get(&expiries[0]).map(|s| s.evaluate(strike))
                } else if idx >= expiries.len() {
                    self.expiry_splines
                        .get(&expiries[expiries.len() - 1])
                        .map(|s| s.evaluate(strike))
                } else {
                    // Linear interpolation between expiries
                    let lower_exp = expiries[idx - 1];
                    let upper_exp = expiries[idx];
                    let lower_iv = self.expiry_splines.get(&lower_exp)?.evaluate(strike);
                    let upper_iv = self.expiry_splines.get(&upper_exp)?.evaluate(strike);

                    let t = (expiry_days - lower_exp) as f64 / (upper_exp - lower_exp) as f64;
                    Some(lower_iv * (1.0 - t) + upper_iv * t)
                }
            }
        }
    }

    /// Calculate skew metrics (25-delta risk reversal and butterfly)
    fn calculate_skew_metrics(&mut self, points: &[VolPoint]) {
        // Simplified skew calculation
        let mut atm_vols = Vec::new();
        let mut otm_call_vols = Vec::new();
        let mut otm_put_vols = Vec::new();

        for point in points {
            // Classify based on moneyness
            let moneyness = point.strike / 100.0; // Assuming spot = 100 for normalization
            if (0.95..=1.05).contains(&moneyness) {
                atm_vols.push(point.implied_vol);
            } else if moneyness > 1.05 {
                otm_call_vols.push(point.implied_vol);
            } else if moneyness < 0.95 {
                otm_put_vols.push(point.implied_vol);
            }
        }

        self.atm_iv = atm_vols.iter().sum::<f64>() / atm_vols.len().max(1) as f64;

        let avg_call = otm_call_vols.iter().sum::<f64>() / otm_call_vols.len().max(1) as f64;
        let avg_put = otm_put_vols.iter().sum::<f64>() / otm_put_vols.len().max(1) as f64;

        self.risk_reversal_25d = avg_call - avg_put;
        self.butterfly_25d = (avg_call + avg_put) / 2.0 - self.atm_iv;
    }

    /// Get ATM IV
    pub fn atm_iv(&self) -> f64 {
        self.atm_iv
    }

    /// Get 25-delta risk reversal (call skew - put skew)
    pub fn risk_reversal_25d(&self) -> f64 {
        self.risk_reversal_25d
    }

    /// Get 25-delta butterfly (curvature)
    pub fn butterfly_25d(&self) -> f64 {
        self.butterfly_25d
    }

    /// Set strike bounds for filtering
    pub fn set_strike_bounds(&mut self, min: f64, max: f64) {
        self.min_strike = min;
        self.max_strike = max;
    }
}

/// Real-time IV surface tracker with incremental updates
pub struct IVSurfaceTracker {
    surface: IVSurface,
    update_count: u64,
}

impl IVSurfaceTracker {
    pub fn new(max_points: usize) -> Self {
        Self {
            surface: IVSurface::new(max_points),
            update_count: 0,
        }
    }

    /// Incrementally update surface with new tick
    pub fn update_tick(&mut self, point: VolPoint) {
        self.update_count += 1;
        // In production, would merge with existing surface efficiently
        // For now, rebuild periodically
        if self.update_count % 100 == 0 {
            // Trigger rebuild logic here
        }
    }

    pub fn get_surface(&self) -> &IVSurface {
        &self.surface
    }

    pub fn rebuild(&mut self, points: Vec<VolPoint>) {
        self.surface.build_surface(points);
        self.update_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iv_surface_build() {
        let mut surface = IVSurface::new(100);
        let points = vec![
            VolPoint {
                strike: 90.0,
                expiry_days: 30,
                implied_vol: 0.75,
                bid_iv: 0.74,
                ask_iv: 0.76,
                volume: 1000,
                open_interest: 5000,
            },
            VolPoint {
                strike: 100.0,
                expiry_days: 30,
                implied_vol: 0.70,
                bid_iv: 0.69,
                ask_iv: 0.71,
                volume: 2000,
                open_interest: 8000,
            },
            VolPoint {
                strike: 110.0,
                expiry_days: 30,
                implied_vol: 0.72,
                bid_iv: 0.71,
                ask_iv: 0.73,
                volume: 1500,
                open_interest: 6000,
            },
        ];

        surface.build_surface(points);
        let iv = surface.get_iv(100.0, 30);
        assert!(iv.is_some());
    }
}

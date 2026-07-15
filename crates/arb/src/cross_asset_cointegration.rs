//! High-dimensional cointegration engine using Johansen test
//! Tracks multi-asset mean reversion for baskets of >3 assets
//! Memory-efficient implementation for strict RAM constraints

use std::collections::HashMap;

/// Rolling window data structure with fixed capacity
struct RollingMatrix {
    data: Vec<Vec<f64>>,
    max_rows: usize,
    current_row: usize,
    filled: bool,
}

impl RollingMatrix {
    fn new(max_rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![0.0; cols]; max_rows],
            max_rows,
            current_row: 0,
            filled: false,
        }
    }

    fn push(&mut self, row: &[f64]) {
        assert_eq!(row.len(), self.data[0].len());
        self.data[self.current_row].copy_from_slice(row);
        self.current_row = (self.current_row + 1) % self.max_rows;
        if self.current_row == 0 {
            self.filled = true;
        }
    }

    fn n_obs(&self) -> usize {
        if self.filled {
            self.max_rows
        } else {
            self.current_row
        }
    }

    fn n_cols(&self) -> usize {
        self.data[0].len()
    }

    fn get_data(&self) -> &[Vec<f64>] {
        if self.filled {
            &self.data
        } else {
            &self.data[..self.current_row]
        }
    }
}

/// Johansen cointegration test implementation
pub struct JohansenTest {
    /// Maximum lag for VAR model
    max_lag: usize,
    /// Critical values for trace statistic (precomputed for common significance levels)
    critical_values_95: Vec<f64>,
    critical_values_90: Vec<f64>,
    /// Minimum observations required
    min_obs: usize,
}

impl JohansenTest {
    pub fn new(max_lag: usize) -> Self {
        // Precomputed critical values for trace statistic (Osterwald-Lenum)
        // Indexed by number of cointegrating relationships
        let critical_values_95 = vec![15.41, 21.45, 27.82, 34.12, 40.24];
        let critical_values_90 = vec![13.39, 19.19, 25.32, 31.24, 37.08];

        Self {
            max_lag,
            critical_values_95,
            critical_values_90,
            min_obs: 50,
        }
    }

    /// Perform Johansen cointegration test on price series
    /// 
    /// # Arguments
    /// * `prices` - Matrix where each row is a time period and each column is an asset
    /// 
    /// # Returns
    /// CointegrationTestResult with rank and statistics
    pub fn test(&self, prices: &[Vec<f64>]) -> CointegrationResult {
        if prices.is_empty() {
            return CointegrationResult::invalid();
        }

        let n_obs = prices.len();
        let n_assets = prices[0].len();

        if n_obs < self.min_obs + self.max_lag {
            return CointegrationResult::insufficient_data(n_obs, self.min_obs + self.max_lag);
        }

        if n_assets == 0 {
            return CointegrationResult::invalid();
        }

        // Convert prices to log returns
        let log_prices: Vec<Vec<f64>> = prices
            .iter()
            .map(|row| row.iter().map(|p| p.ln()).collect())
            .collect();

        let returns = self._compute_returns(&log_prices);

        // Perform VAR and extract residuals
        let (delta_x, x_lagged) = self._prepare_var_data(&returns);

        if delta_x.len() < self.min_obs {
            return CointegrationResult::insufficient_data(delta_x.len(), self.min_obs);
        }

        // Compute moment matrices
        let s00 = self._moment_matrix(&delta_x);
        let s01 = self._cross_moment(&delta_x, &x_lagged);
        let s11 = self._moment_matrix(&x_lagged);

        // Solve eigenvalue problem: |S01 * S11^-1 * S10 - lambda * S00| = 0
        let eigenvalues = self._solve_eigenproblem(&s00, &s01, &s11, n_assets);

        // Compute trace statistics
        let trace_stats = self._compute_trace_statistics(&eigenvalues, n_obs);

        // Determine cointegration rank
        let rank = self._determine_rank(&trace_stats);

        // Extract cointegrating vectors (eigenvectors corresponding to significant eigenvalues)
        let cointegrating_vectors = self._extract_cointegrating_vectors(
            &eigenvalues,
            n_assets,
            rank,
        );

        CointegrationResult {
            valid: true,
            n_assets,
            n_observations: n_obs,
            rank,
            eigenvalues,
            trace_statistics: trace_stats,
            cointegrating_vectors,
            critical_95: self.critical_values_95.clone(),
            critical_90: self.critical_values_90.clone(),
        }
    }

    /// Rolling Johansen test with memory efficiency
    pub fn rolling_test(&self, price_buffer: &RollingMatrix) -> CointegrationResult {
        let data = price_buffer.get_data();
        
        // Convert to Vec<Vec<f64>> format
        let prices: Vec<Vec<f64>> = data.iter().map(|r| r.clone()).collect();
        
        self.test(&prices)
    }

    fn _compute_returns(&self, log_prices: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_obs = log_prices.len();
        let n_assets = log_prices[0].len();
        let mut returns = Vec::with_capacity(n_obs - 1);

        for i in 1..n_obs {
            let mut ret = Vec::with_capacity(n_assets);
            for j in 0..n_assets {
                ret.push(log_prices[i][j] - log_prices[i - 1][j]);
            }
            returns.push(ret);
        }

        returns
    }

    fn _prepare_var_data(
        &self,
        returns: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n_lag = self.max_lag;
        let n_obs = returns.len();
        let n_assets = returns[0].len();

        let mut delta_x = Vec::new();
        let mut x_lagged = Vec::new();

        for i in n_lag..n_obs {
            // Delta X_t
            delta_x.push(returns[i].clone());

            // X_{t-1} (lagged level)
            // For simplicity, use cumulative sum up to t-1
            let mut lagged_level = vec![0.0; n_assets];
            for j in 0..n_assets {
                for k in 0..i {
                    lagged_level[j] += returns[k][j];
                }
            }
            x_lagged.push(lagged_level);
        }

        (delta_x, x_lagged)
    }

    fn _moment_matrix(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = data[0].len();
        let mut result = vec![vec![0.0; n]; n];

        let n_obs = data.len() as f64;
        for row in data {
            for i in 0..n {
                for j in 0..n {
                    result[i][j] += row[i] * row[j];
                }
            }
        }

        // Normalize
        for i in 0..n {
            for j in 0..n {
                result[i][j] /= n_obs;
            }
        }

        result
    }

    fn _cross_moment(&self, x: &[Vec<f64>], y: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_rows = x[0].len();
        let n_cols = y[0].len();
        let mut result = vec![vec![0.0; n_cols]; n_rows];

        let n_obs = x.len() as f64;
        for i in 0..x.len() {
            for r in 0..n_rows {
                for c in 0..n_cols {
                    result[r][c] += x[i][r] * y[i][c];
                }
            }
        }

        for r in 0..n_rows {
            for c in 0..n_cols {
                result[r][c] /= n_obs;
            }
        }

        result
    }

    fn _solve_eigenproblem(
        &self,
        s00: &[Vec<f64>],
        s01: &[Vec<f64>],
        s11: &[Vec<f64>],
        n: usize,
    ) -> Vec<f64> {
        // Simplified eigenvalue computation using power iteration
        // In production, use a proper linear algebra library like nalgebra
        
        // Compute S11 inverse (simplified using Neumann series for well-conditioned matrices)
        let s11_inv = self._matrix_inverse_approx(s11);

        // Compute S01 * S11_inv * S10
        let s10 = self._transpose(s01);
        let temp = self._matrix_multiply(s01, &s11_inv);
        let product = self._matrix_multiply(&temp, &s10);

        // Find eigenvalues using simplified approach
        // This is a placeholder - real implementation would use QR algorithm
        self._eigenvalues_power_method(&product, n)
    }

    fn _matrix_inverse_approx(&self, m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = m.len();
        // Start with identity scaled by trace
        let trace: f64 = (0..n).map(|i| m[i][i]).sum();
        let scale = trace / n as f64;
        
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            inv[i][i] = 1.0 / (scale + 1e-10);
        }

        // Refine using Newton-Schulz iteration: X_{k+1} = X_k(2I - AX_k)
        for _ in 0..5 {
            let ax = self._matrix_multiply(m, &inv);
            let mut two_i_ax = vec![vec![0.0; n]; n];
            for i in 0..n {
                two_i_ax[i][i] = 2.0;
                for j in 0..n {
                    two_i_ax[i][j] -= ax[i][j];
                }
            }
            inv = self._matrix_multiply(&inv, &two_i_ax);
        }

        inv
    }

    fn _matrix_multiply(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = a.len();
        let n = b[0].len();
        let k = b.len();
        let mut result = vec![vec![0.0; n]; m];

        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    result[i][j] += a[i][l] * b[l][j];
                }
            }
        }

        result
    }

    fn _transpose(&self, m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = m.len();
        let cols = m[0].len();
        let mut result = vec![vec![0.0; rows]; cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j][i] = m[i][j];
            }
        }

        result
    }

    fn _eigenvalues_power_method(&self, m: &[Vec<f64>], n: usize) -> Vec<f64> {
        // Simplified: find dominant eigenvalues using deflation
        let mut eigenvalues = Vec::with_capacity(n);
        let mut work = m.to_vec();

        for _ in 0..n {
            // Power iteration
            let mut v = vec![1.0; work.len()];
            let mut eigenvalue = 0.0;

            for _ in 0..50 {
                let mut new_v = vec![0.0; work.len()];
                for i in 0..work.len() {
                    for j in 0..work[0].len() {
                        new_v[i] += work[i][j] * v[j];
                    }
                }

                let norm: f64 = new_v.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                if norm < 1e-10 {
                    break;
                }

                eigenvalue = norm;
                v = new_v;
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }

            eigenvalues.push(eigenvalue);

            // Deflate matrix
            for i in 0..work.len() {
                for j in 0..work[0].len() {
                    work[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }

    fn _compute_trace_statistics(&self, eigenvalues: &[f64], n_obs: usize) -> Vec<f64> {
        let n = eigenvalues.len();
        let mut trace_stats = Vec::with_capacity(n);

        for r in 0..n {
            // Trace statistic: -T * sum(ln(1 - lambda_i)) for i = r+1 to n
            let sum: f64 = eigenvalues[r..]
                .iter()
                .filter(|&&lam| lam < 1.0 && lam > 0.0)
                .map(|&lam| (1.0 - lam).ln())
                .sum();

            trace_stats.push(-n_obs as f64 * sum);
        }

        trace_stats
    }

    fn _determine_rank(&self, trace_stats: &[f64]) -> usize {
        for (r, &stat) in trace_stats.iter().enumerate() {
            let crit_idx = r.min(self.critical_values_95.len() - 1);
            if stat < self.critical_values_95[crit_idx] {
                return r;
            }
        }
        trace_stats.len()
    }

    fn _extract_cointegrating_vectors(
        &self,
        _eigenvalues: &[f64],
        n_assets: usize,
        rank: usize,
    ) -> Vec<Vec<f64>> {
        // Placeholder: return identity vectors for cointegrating relationships
        // Real implementation would extract eigenvectors
        (0..rank.min(n_assets))
            .map(|i| {
                let mut vec = vec![0.0; n_assets];
                vec[i % n_assets] = 1.0;
                vec
            })
            .collect()
    }
}

/// Result of cointegration test
#[derive(Debug, Clone)]
pub struct CointegrationResult {
    pub valid: bool,
    pub n_assets: usize,
    pub n_observations: usize,
    pub rank: usize,
    pub eigenvalues: Vec<f64>,
    pub trace_statistics: Vec<f64>,
    pub cointegrating_vectors: Vec<Vec<f64>>,
    pub critical_95: Vec<f64>,
    pub critical_90: Vec<f64>,
}

impl CointegrationResult {
    fn invalid() -> Self {
        Self {
            valid: false,
            n_assets: 0,
            n_observations: 0,
            rank: 0,
            eigenvalues: Vec::new(),
            trace_statistics: Vec::new(),
            cointegrating_vectors: Vec::new(),
            critical_95: Vec::new(),
            critical_90: Vec::new(),
        }
    }

    fn insufficient_data(current: usize, required: usize) -> Self {
        let mut result = Self::invalid();
        result.n_observations = current;
        result
    }

    pub fn is_cointegrated(&self) -> bool {
        self.valid && self.rank > 0
    }

    pub fn significance(&self, rank: usize) -> Option<&str> {
        if rank >= self.trace_statistics.len() {
            return None;
        }

        let stat = self.trace_statistics[rank];
        let crit_95 = self.critical_95.get(rank).copied().unwrap_or(f64::INFINITY);
        let crit_90 = self.critical_90.get(rank).copied().unwrap_or(f64::INFINITY);

        if stat > crit_95 {
            Some("significant_at_95")
        } else if stat > crit_90 {
            Some("significant_at_90")
        } else {
            Some("not_significant")
        }
    }
}

/// Multi-asset mean reversion tracker
pub struct MeanReversionTracker {
    johansen: JohansenTest,
    /// Half-life of mean reversion for each basket
    half_lives: HashMap<Vec<usize>, f64>,
    /// Current spread values
    spreads: HashMap<Vec<usize>, f64>,
    /// Lookback for half-life estimation
    halflife_lookback: usize,
}

impl MeanReversionTracker {
    pub fn new(max_lag: usize, halflife_lookback: usize) -> Self {
        Self {
            johansen: JohansenTest::new(max_lag),
            half_lives: HashMap::new(),
            spreads: HashMap::new(),
            halflife_lookback,
        }
    }

    pub fn update_basket(
        &mut self,
        basket_assets: Vec<usize>,
        prices: &[Vec<f64>],
        weights: &[f64],
    ) -> Option<f64> {
        let result = self.johansen.test(prices);

        if !result.is_cointegrated() {
            self.half_lives.remove(&basket_assets);
            return None;
        }

        // Calculate current spread
        let last_prices = prices.last()?;
        let spread: f64 = weights
            .iter()
            .zip(last_prices.iter())
            .map(|(&w, &p)| w * p.ln())
            .sum();

        self.spreads.insert(basket_assets.clone(), spread);

        // Estimate half-life from recent spread history
        let half_life = self._estimate_half_life(&basket_assets);
        self.half_lives.insert(basket_assets.clone(), half_life);

        Some(half_life)
    }

    fn _estimate_half_life(&self, _basket: &[usize]) -> f64 {
        // Simplified: use Ornstein-Uhlenbeck process fitting
        // In production, fit AR(1) to spread series
        10.0 // Default half-life estimate
    }

    pub fn get_spread(&self, basket: &[usize]) -> Option<f64> {
        self.spreads.get(basket).copied()
    }

    pub fn get_half_life(&self, basket: &[usize]) -> Option<f64> {
        self.half_lives.get(basket).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_johansen_basic() {
        let test = JohansenTest::new(2);
        
        // Generate synthetic cointegrated series
        let n_obs = 100;
        let mut prices = vec![vec![0.0; 3]; n_obs];
        
        for i in 0..n_obs {
            prices[i][0] = 100.0 + (i as f64 * 0.1).sin() * 10.0;
            prices[i][1] = prices[i][0] * 1.5 + (i as f64 * 0.2).sin() * 5.0;
            prices[i][2] = prices[i][0] * 0.8 + (i as f64 * 0.15).sin() * 8.0;
        }
        
        let result = test.test(&prices);
        assert!(result.valid);
    }

    #[test]
    fn test_rolling_matrix() {
        let mut mat = RollingMatrix::new(5, 3);
        
        for i in 0..10 {
            mat.push(&[i as f64, (i + 1) as f64, (i + 2) as f64]);
        }
        
        assert_eq!(mat.n_obs(), 5);
        assert!(mat.filled);
    }
}

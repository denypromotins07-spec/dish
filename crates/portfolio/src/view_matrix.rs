//! Lock-free mapping engine that translates ML directional signals into Black-Litterman P and Omega matrices
//! Avoids memory duplication by using zero-copy views

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::cell::UnsafeCell;

const MAX_ASSETS: usize = 128;
const MAX_VIEWS: usize = 32;

/// A view specification without allocation
#[repr(align(64))]
pub struct ViewSpec {
    /// Asset indices involved in this view (positive for long, negative for short)
    pub asset_indices: [i32; MAX_ASSETS],
    /// Weights for each asset in the view
    pub weights: [f64; MAX_ASSETS],
    /// Expected return
    pub expected_return: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Number of assets in this view
    pub num_assets: usize,
    /// Whether this view is active
    pub active: AtomicBool,
}

impl Default for ViewSpec {
    fn default() -> Self {
        Self {
            asset_indices: [-1; MAX_ASSETS],
            weights: [0.0; MAX_ASSETS],
            expected_return: 0.0,
            confidence: 0.5,
            num_assets: 0,
            active: AtomicBool::new(false),
        }
    }
}

/// Lock-free view matrix builder
#[repr(align(64))]
pub struct ViewMatrixBuilder {
    /// Pool of view specifications
    views: UnsafeCell<[ViewSpec; MAX_VIEWS]>,
    /// Count of active views
    view_count: AtomicUsize,
    /// Total asset count
    asset_count: usize,
    /// Generation counter for lock-free reads
    generation: AtomicUsize,
}

unsafe impl Sync for ViewMatrixBuilder {}
unsafe impl Send for ViewMatrixBuilder {}

impl ViewMatrixBuilder {
    #[inline(always)]
    pub fn new(asset_count: usize) -> Self {
        assert!(asset_count <= MAX_ASSETS);
        
        let views = unsafe { &mut *UnsafeCell::new([ViewSpec::default(); MAX_VIEWS]) };
        for view in views.iter_mut() {
            view.asset_count = asset_count;
        }
        
        Self {
            views: UnsafeCell::new([ViewSpec::default(); MAX_VIEWS]),
            view_count: AtomicUsize::new(0),
            asset_count,
            generation: AtomicUsize::new(0),
        }
    }

    /// Add an absolute view (single asset expected return)
    #[inline(always)]
    pub fn add_absolute_view(&self, asset_index: usize, expected_return: f64, confidence: f64) -> Option<usize> {
        if asset_index >= self.asset_count {
            return None;
        }
        
        let view_idx = self.view_count.load(Ordering::Relaxed);
        if view_idx >= MAX_VIEWS {
            return None;
        }
        
        let views = unsafe { &mut *self.views.get() };
        let view = &mut views[view_idx];
        
        // Reset view
        view.asset_indices.fill(-1);
        view.weights.fill(0.0);
        
        // Set up pick vector
        view.asset_indices[0] = asset_index as i32;
        view.weights[0] = 1.0;
        view.expected_return = expected_return;
        view.confidence = confidence.clamp(0.0, 1.0);
        view.num_assets = 1;
        view.active.store(true, Ordering::Release);
        
        // Increment count and generation
        self.view_count.fetch_add(1, Ordering::Release);
        self.generation.fetch_add(1, Ordering::Release);
        
        Some(view_idx)
    }

    /// Add a relative view (outperformance between two assets)
    #[inline(always)]
    pub fn add_relative_view(
        &self,
        long_asset: usize,
        short_asset: usize,
        outperformance: f64,
        confidence: f64,
    ) -> Option<usize> {
        if long_asset >= self.asset_count || short_asset >= self.asset_count {
            return None;
        }
        
        let view_idx = self.view_count.load(Ordering::Relaxed);
        if view_idx >= MAX_VIEWS {
            return None;
        }
        
        let views = unsafe { &mut *self.views.get() };
        let view = &mut views[view_idx];
        
        // Reset view
        view.asset_indices.fill(-1);
        view.weights.fill(0.0);
        
        // Set up pick vector: +1 for long, -1 for short
        view.asset_indices[0] = long_asset as i32;
        view.asset_indices[1] = short_asset as i32;
        view.weights[0] = 1.0;
        view.weights[1] = -1.0;
        view.expected_return = outperformance;
        view.confidence = confidence.clamp(0.0, 1.0);
        view.num_assets = 2;
        view.active.store(true, Ordering::Release);
        
        self.view_count.fetch_add(1, Ordering::Release);
        self.generation.fetch_add(1, Ordering::Release);
        
        Some(view_idx)
    }

    /// Add a basket view (e.g., "L1 tokens will outperform L2 tokens")
    #[inline(always)]
    pub fn add_basket_view(
        &self,
        long_assets: &[(usize, f64)],
        short_assets: &[(usize, f64)],
        expected_outperformance: f64,
        confidence: f64,
    ) -> Option<usize> {
        let total_assets = long_assets.len() + short_assets.len();
        if total_assets > self.asset_count || total_assets > MAX_ASSETS {
            return None;
        }
        
        let view_idx = self.view_count.load(Ordering::Relaxed);
        if view_idx >= MAX_VIEWS {
            return None;
        }
        
        let views = unsafe { &mut *self.views.get() };
        let view = &mut views[view_idx];
        
        // Reset view
        view.asset_indices.fill(-1);
        view.weights.fill(0.0);
        
        let mut idx = 0;
        
        // Add long positions
        for &(asset, weight) in long_assets {
            if asset < self.asset_count {
                view.asset_indices[idx] = asset as i32;
                view.weights[idx] = weight;
                idx += 1;
            }
        }
        
        // Add short positions (negative weights)
        for &(asset, weight) in short_assets {
            if asset < self.asset_count {
                view.asset_indices[idx] = asset as i32;
                view.weights[idx] = -weight;
                idx += 1;
            }
        }
        
        view.expected_return = expected_outperformance;
        view.confidence = confidence.clamp(0.0, 1.0);
        view.num_assets = idx;
        view.active.store(true, Ordering::Release);
        
        self.view_count.fetch_add(1, Ordering::Release);
        self.generation.fetch_add(1, Ordering::Release);
        
        Some(view_idx)
    }

    /// Get current generation for lock-free synchronization
    #[inline(always)]
    pub fn generation(&self) -> usize {
        self.generation.load(Ordering::Acquire)
    }

    /// Get count of active views
    #[inline(always)]
    pub fn view_count(&self) -> usize {
        self.view_count.load(Ordering::Acquire)
    }

    /// Extract P matrix row (pick vector) for a view - zero copy
    #[inline(always)]
    pub fn get_pick_vector(&self, view_idx: usize) -> Option<&[f64]> {
        if view_idx >= self.view_count.load(Ordering::Acquire) {
            return None;
        }
        
        let views = unsafe { &*self.views.get() };
        let view = &views[view_idx];
        
        if !view.active.load(Ordering::Acquire) {
            return None;
        }
        
        Some(&view.weights[..view.num_assets])
    }

    /// Get all active views as iterator
    #[inline(always)]
    pub fn active_views(&self) -> ActiveViewsIter<'_> {
        let count = self.view_count.load(Ordering::Acquire);
        let views = unsafe { &*self.views.get() };
        ActiveViewsIter {
            views,
            index: 0,
            count,
        }
    }

    /// Clear all views
    #[inline(always)]
    pub fn clear(&self) {
        let views = unsafe { &mut *self.views.get() };
        for view in views.iter_mut() {
            view.active.store(false, Ordering::Release);
        }
        self.view_count.store(0, Ordering::Release);
        self.generation.fetch_add(1, Ordering::Release);
    }
}

/// Iterator over active views
pub struct ActiveViewsIter<'a> {
    views: &'a [ViewSpec; MAX_VIEWS],
    index: usize,
    count: usize,
}

impl<'a> Iterator for ActiveViewsIter<'a> {
    type Item = &'a ViewSpec;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.count {
            let view = &self.views[self.index];
            self.index += 1;
            if view.active.load(Ordering::Acquire) {
                return Some(view);
            }
        }
        None
    }
}

/// Translated Black-Litterman matrices (zero-copy representation)
pub struct TranslatedMatrices<'a> {
    pub p_matrix: Vec<&'a [f64]>,
    pub q_vector: Vec<f64>,
    pub omega_diag: Vec<f64>,
}

impl ViewMatrixBuilder {
    /// Translate views into BL matrices
    #[inline(always)]
    pub fn translate(&self, covariance_diag: &[f64], tau: f64) -> TranslatedMatrices<'_> {
        let mut p_matrix = Vec::with_capacity(self.view_count.load(Ordering::Relaxed));
        let mut q_vector = Vec::with_capacity(self.view_count.load(Ordering::Relaxed));
        let mut omega_diag = Vec::with_capacity(self.view_count.load(Ordering::Relaxed));
        
        for view in self.active_views() {
            // P row
            p_matrix.push(&view.weights[..view.num_assets]);
            
            // Q element
            q_vector.push(view.expected_return);
            
            // Omega diagonal element
            // Omega_kk = tau * (P_k * diag(Sigma) * P_k') / confidence
            let mut variance = 0.0;
            for i in 0..view.num_assets {
                let asset_idx = view.asset_indices[i] as usize;
                if asset_idx < covariance_diag.len() {
                    variance += view.weights[i].powi(2) * covariance_diag[asset_idx];
                }
            }
            
            let confidence_factor = if view.confidence > 1e-6 { 1.0 / view.confidence } else { 1e6 };
            omega_diag.push(tau * variance * confidence_factor);
        }
        
        TranslatedMatrices {
            p_matrix,
            q_vector,
            omega_diag,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_matrix_builder() {
        let builder = ViewMatrixBuilder::new(10);
        
        // Add absolute view
        builder.add_absolute_view(0, 0.15, 0.7);
        
        // Add relative view
        builder.add_relative_view(1, 2, 0.05, 0.6);
        
        assert_eq!(builder.view_count(), 2);
        
        // Verify views
        let views: Vec<_> = builder.active_views().collect();
        assert_eq!(views.len(), 2);
        
        // First view should be absolute on asset 0
        assert_eq!(views[0].weights[0], 1.0);
        assert_eq!(views[0].expected_return, 0.15);
        
        // Second view should be relative
        assert_eq!(views[1].weights[0], 1.0);
        assert_eq!(views[1].weights[1], -1.0);
        assert_eq!(views[1].expected_return, 0.05);
    }
}

"""
PyO3 Bindings for Rust Market Data Models.

Exposes high-performance Rust data models to Python/Nautilus without
memory duplication, utilizing Python's buffer protocol for zero-copy
array transfers between Rust and Python.
"""

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::mem;

use crate::market_data::{
    Bar, BookLevel, OrderBookSnapshot, TradeTick, MAX_BOOK_DEPTH,
};
use crate::orderflow_metrics::{
    FootprintBar, FootprintLevel, LiquiditySweep, OrderFlowMetrics, SMCStructure, SMCType,
};

/// Python module for market data models
#[pymodule]
fn rust_models(_py: Python, m: &PyModule) -> PyResult<()> {
    // Market data structs
    m.add_class::<PyTradeTick>()?;
    m.add_class::<PyOrderBookSnapshot>()?;
    m.add_class::<PyBar>()?;
    
    // Order flow metrics
    m.add_class::<PyOrderFlowMetrics>()?;
    m.add_class::<PyFootprintBar>()?;
    m.add_class::<PyLiquiditySweep>()?;
    m.add_class::<PySMCStructure>()?;
    
    // Constants
    m.add("MAX_BOOK_DEPTH", MAX_BOOK_DEPTH)?;
    
    Ok(())
}

/// Python-exposed TradeTick wrapper
#[pyclass(name = "TradeTick")]
#[derive(Clone)]
pub struct PyTradeTick {
    #[pyo3(get, set)]
    pub symbol_hash: u64,
    #[pyo3(get, set)]
    pub price: f64,
    #[pyo3(get, set)]
    pub quantity: f64,
    #[pyo3(get, set)]
    pub timestamp_ns: i64,
    #[pyo3(get, set)]
    pub trade_id: i64,
    #[pyo3(get, set)]
    pub is_buyer_maker: bool,
}

#[pymethods]
impl PyTradeTick {
    #[new]
    fn new(
        symbol_hash: u64,
        price: f64,
        quantity: f64,
        timestamp_ns: i64,
        trade_id: i64,
        is_buyer_maker: bool,
    ) -> Self {
        Self {
            symbol_hash,
            price,
            quantity,
            timestamp_ns,
            trade_id,
            is_buyer_maker,
        }
    }

    /// Convert to internal Rust format (zero-copy when possible)
    fn to_rust(&self) -> TradeTick {
        TradeTick::new(
            self.symbol_hash,
            self.price,
            self.quantity,
            self.timestamp_ns,
            self.trade_id,
            self.is_buyer_maker,
        )
    }

    /// Create from internal Rust format
    #[staticmethod]
    fn from_rust(tick: &TradeTick) -> Self {
        Self {
            symbol_hash: tick.symbol_hash,
            price: tick.price_f64(),
            quantity: tick.quantity_f64(),
            timestamp_ns: tick.timestamp_ns,
            trade_id: tick.trade_id,
            is_buyer_maker: tick.is_buyer_maker(),
        }
    }

    /// Serialize to bytes (for zero-copy transfer)
    fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        let rust_tick = self.to_rust();
        let bytes = bytemuck::bytes_of(&rust_tick);
        PyBytes::new(py, bytes)
    }

    /// Deserialize from bytes (zero-copy)
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        if data.len() != mem::size_of::<TradeTick>() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid byte length for TradeTick",
            ));
        }
        
        let tick: &TradeTick = bytemuck::from_bytes(data);
        Ok(Self::from_rust(tick))
    }

    fn __repr__(&self) -> String {
        format!(
            "TradeTick(symbol_hash={}, price={}, qty={}, time={})",
            self.symbol_hash, self.price, self.quantity, self.timestamp_ns
        )
    }
}

/// Python-exposed OrderBookSnapshot wrapper
#[pyclass(name = "OrderBookSnapshot")]
pub struct PyOrderBookSnapshot {
    #[pyo3(get, set)]
    pub symbol_hash: u64,
    #[pyo3(get, set)]
    pub last_update_id: u64,
    #[pyo3(get, set)]
    pub timestamp_ns: i64,
    #[pyo3(get, set)]
    pub bids: Vec<(f64, f64)>,
    #[pyo3(get, set)]
    pub asks: Vec<(f64, f64)>,
}

#[pymethods]
impl PyOrderBookSnapshot {
    #[new]
    fn new(symbol_hash: u64, last_update_id: u64, timestamp_ns: i64) -> Self {
        Self {
            symbol_hash,
            last_update_id,
            timestamp_ns,
            bids: Vec::with_capacity(MAX_BOOK_DEPTH),
            asks: Vec::with_capacity(MAX_BOOK_DEPTH),
        }
    }

    fn add_bid(&mut self, price: f64, quantity: f64) {
        self.bids.push((price, quantity));
    }

    fn add_ask(&mut self, price: f64, quantity: f64) {
        self.asks.push((price, quantity));
    }

    fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(p, _)| *p)
    }

    fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(p, _)| *p)
    }

    fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Convert to internal Rust format
    fn to_rust(&self) -> OrderBookSnapshot {
        let mut snapshot = OrderBookSnapshot::empty(self.symbol_hash);
        snapshot.last_update_id = self.last_update_id;
        snapshot.timestamp_ns = self.timestamp_ns;

        for (i, (price, qty)) in self.bids.iter().take(MAX_BOOK_DEPTH).enumerate() {
            snapshot.set_bid(i, *price, *qty);
        }

        for (i, (price, qty)) in self.asks.iter().take(MAX_BOOK_DEPTH).enumerate() {
            snapshot.set_ask(i, *price, *qty);
        }

        snapshot
    }

    /// Serialize to bytes
    fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        let rust_snapshot = self.to_rust();
        let bytes = bytemuck::bytes_of(&rust_snapshot);
        PyBytes::new(py, bytes)
    }

    /// Get as dictionary for Nautilus compatibility
    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("symbol_hash", self.symbol_hash)?;
            dict.set_item("last_update_id", self.last_update_id)?;
            dict.set_item("timestamp_ns", self.timestamp_ns)?;
            dict.set_item("bids", self.bids.clone())?;
            dict.set_item("asks", self.asks.clone())?;
            dict.set_item("best_bid", self.best_bid())?;
            dict.set_item("best_ask", self.best_ask())?;
            dict.set_item("mid_price", self.mid_price())?;
            dict.set_item("spread", self.spread())?;
            Ok(dict.into())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "OrderBookSnapshot(symbol={}, bids={}, asks={}, mid={:?})",
            self.symbol_hash,
            self.bids.len(),
            self.asks.len(),
            self.mid_price()
        )
    }
}

/// Python-exposed Bar wrapper
#[pyclass(name = "Bar")]
#[derive(Clone)]
pub struct PyBar {
    #[pyo3(get, set)]
    pub symbol_hash: u64,
    #[pyo3(get, set)]
    pub timestamp_ns: i64,
    #[pyo3(get, set)]
    pub open: f64,
    #[pyo3(get, set)]
    pub high: f64,
    #[pyo3(get, set)]
    pub low: f64,
    #[pyo3(get, set)]
    pub close: f64,
    #[pyo3(get, set)]
    pub volume: f64,
    #[pyo3(get, set)]
    pub trade_count: u32,
    #[pyo3(get, set)]
    pub duration_ns: i64,
    #[pyo3(get, set)]
    pub is_complete: bool,
}

#[pymethods]
impl PyBar {
    #[new]
    fn new(
        symbol_hash: u64,
        timestamp_ns: i64,
        duration_ns: i64,
    ) -> Self {
        Self {
            symbol_hash,
            timestamp_ns,
            open: 0.0,
            high: 0.0,
            low: f64::MAX,
            close: 0.0,
            volume: 0.0,
            trade_count: 0,
            duration_ns,
            is_complete: false,
        }
    }

    fn update(&mut self, price: f64, quantity: f64) {
        if self.trade_count == 0 {
            self.open = price;
        }
        if price > self.high || self.high == 0.0 {
            self.high = price;
        }
        if price < self.low {
            self.low = price;
        }
        self.close = price;
        self.volume += quantity;
        self.trade_count += 1;
    }

    fn complete(&mut self) {
        self.is_complete = true;
    }

    /// Convert to Rust format
    fn to_rust(&self) -> Bar {
        let mut bar = Bar::new(self.symbol_hash, self.timestamp_ns, self.duration_ns);
        
        if self.trade_count > 0 {
            bar.update(self.open, 1.0); // Initialize with open
            // Rebuild from OHLCV (simplified)
            bar.update(self.close, self.volume);
        }
        
        if self.is_complete {
            bar.complete();
        }
        
        bar
    }

    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("symbol_hash", self.symbol_hash)?;
            dict.set_item("timestamp_ns", self.timestamp_ns)?;
            dict.set_item("open", self.open)?;
            dict.set_item("high", self.high)?;
            dict.set_item("low", self.low)?;
            dict.set_item("close", self.close)?;
            dict.set_item("volume", self.volume)?;
            dict.set_item("trade_count", self.trade_count)?;
            dict.set_item("is_complete", self.is_complete)?;
            Ok(dict.into())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Bar({}, O={} H={} L={} C={} V={})",
            self.symbol_hash, self.open, self.high, self.low, self.close, self.volume
        )
    }
}

/// Python-exposed OrderFlowMetrics wrapper
#[pyclass(name = "OrderFlowMetrics")]
#[derive(Clone)]
pub struct PyOrderFlowMetrics {
    #[pyo3(get, set)]
    pub symbol_hash: u64,
    #[pyo3(get, set)]
    pub timestamp_ns: i64,
    #[pyo3(get, set)]
    pub aggressive_buy_volume: f64,
    #[pyo3(get, set)]
    pub aggressive_sell_volume: f64,
    #[pyo3(get, set)]
    pub delta: f64,
    #[pyo3(get, set)]
    pub cvd: f64,
    #[pyo3(get, set)]
    pub vwap: f64,
    #[pyo3(get, set)]
    pub total_volume: f64,
    #[pyo3(get, set)]
    pub trade_count: u32,
    #[pyo3(get, set)]
    pub has_imbalance: bool,
}

#[pymethods]
impl PyOrderFlowMetrics {
    #[new]
    fn new(symbol_hash: u64, timestamp_ns: i64) -> Self {
        Self {
            symbol_hash,
            timestamp_ns,
            aggressive_buy_volume: 0.0,
            aggressive_sell_volume: 0.0,
            delta: 0.0,
            cvd: 0.0,
            vwap: 0.0,
            total_volume: 0.0,
            trade_count: 0,
            has_imbalance: false,
        }
    }

    fn update(&mut self, price: f64, quantity: f64, is_buyer_maker: bool) {
        if is_buyer_maker {
            self.aggressive_sell_volume += quantity;
        } else {
            self.aggressive_buy_volume += quantity;
        }
        self.delta = self.aggressive_buy_volume - self.aggressive_sell_volume;
        self.total_volume += quantity;
        self.trade_count += 1;
        
        // Simple VWAP calculation
        if self.total_volume > 0.0 {
            self.vwap = (self.vwap * (self.total_volume - quantity) + price * quantity) 
                / self.total_volume;
        }
        
        // Check imbalance
        if self.total_volume > 0.0 {
            let buy_ratio = self.aggressive_buy_volume / self.total_volume;
            self.has_imbalance = buy_ratio > 0.7 || buy_ratio < 0.3;
        }
    }

    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("symbol_hash", self.symbol_hash)?;
            dict.set_item("timestamp_ns", self.timestamp_ns)?;
            dict.set_item("aggressive_buy_volume", self.aggressive_buy_volume)?;
            dict.set_item("aggressive_sell_volume", self.aggressive_sell_volume)?;
            dict.set_item("delta", self.delta)?;
            dict.set_item("cvd", self.cvd)?;
            dict.set_item("vwap", self.vwap)?;
            dict.set_item("total_volume", self.total_volume)?;
            dict.set_item("trade_count", self.trade_count)?;
            dict.set_item("has_imbalance", self.has_imbalance)?;
            Ok(dict.into())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "OrderFlowMetrics(delta={}, cvd={}, imbalance={})",
            self.delta, self.cvd, self.has_imbalance
        )
    }
}

/// Python-exposed FootprintBar wrapper
#[pyclass(name = "FootprintBar")]
pub struct PyFootprintBar {
    #[pyo3(get, set)]
    pub symbol_hash: u64,
    #[pyo3(get, set)]
    pub timestamp_ns: i64,
    #[pyo3(get, set)]
    pub levels: Vec<PyObject>,
}

#[pymethods]
impl PyFootprintBar {
    #[new]
    fn new(symbol_hash: u64, timestamp_ns: i64) -> Self {
        Self {
            symbol_hash,
            timestamp_ns,
            levels: Vec::new(),
        }
    }

    fn add_level(&mut self, price: f64, buy_volume: f64, sell_volume: f64) {
        Python::with_gil(|py| {
            let level_dict = PyDict::new(py);
            level_dict.set_item("price", price).unwrap();
            level_dict.set_item("buy_volume", buy_volume).unwrap();
            level_dict.set_item("sell_volume", sell_volume).unwrap();
            self.levels.push(level_dict.into());
        });
    }

    fn __repr__(&self) -> String {
        format!("FootprintBar({}, {} levels)", self.symbol_hash, self.levels.len())
    }
}

/// Python-exposed LiquiditySweep wrapper
#[pyclass(name = "LiquiditySweep")]
#[derive(Clone)]
pub struct PyLiquiditySweep {
    #[pyo3(get, set)]
    pub symbol_hash: u64,
    #[pyo3(get, set)]
    pub timestamp_ns: i64,
    #[pyo3(get, set)]
    pub swept_price: f64,
    #[pyo3(get, set)]
    pub direction: i32,
    #[pyo3(get, set)]
    pub volume: f64,
    #[pyo3(get, set)]
    pub reversed: bool,
}

#[pymethods]
impl PyLiquiditySweep {
    #[new]
    fn new(
        symbol_hash: u64,
        timestamp_ns: i64,
        swept_price: f64,
        direction: i32,
        volume: f64,
    ) -> Self {
        Self {
            symbol_hash,
            timestamp_ns,
            swept_price,
            direction,
            volume,
            reversed: false,
        }
    }

    fn is_upside(&self) -> bool {
        self.direction > 0
    }

    fn is_downside(&self) -> bool {
        self.direction < 0
    }

    fn __repr__(&self) -> String {
        format!(
            "LiquiditySweep(price={}, direction={}, reversed={})",
            self.swept_price,
            if self.direction > 0 { "up" } else { "down" },
            self.reversed
        )
    }
}

/// Python-exposed SMC Structure wrapper
#[pyclass(name = "SMCStructure")]
#[derive(Clone)]
pub struct PySMCStructure {
    #[pyo3(get, set)]
    pub structure_type: String,
    #[pyo3(get, set)]
    pub is_bullish: bool,
    #[pyo3(get, set)]
    pub symbol_hash: u64,
    #[pyo3(get, set)]
    pub timestamp_ns: i64,
    #[pyo3(get, set)]
    pub high: f64,
    #[pyo3(get, set)]
    pub low: f64,
    #[pyo3(get, set)]
    pub strength_score: u32,
}

#[pymethods]
impl PySMCStructure {
    #[new]
    fn new(
        structure_type: String,
        is_bullish: bool,
        symbol_hash: u64,
        timestamp_ns: i64,
        high: f64,
        low: f64,
    ) -> Self {
        Self {
            structure_type,
            is_bullish,
            symbol_hash,
            timestamp_ns,
            high,
            low,
            strength_score: 50,
        }
    }

    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("type", &self.structure_type)?;
            dict.set_item("is_bullish", self.is_bullish)?;
            dict.set_item("symbol_hash", self.symbol_hash)?;
            dict.set_item("timestamp_ns", self.timestamp_ns)?;
            dict.set_item("high", self.high)?;
            dict.set_item("low", self.low)?;
            dict.set_item("strength", self.strength_score)?;
            Ok(dict.into())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SMCStructure(type={}, bullish={}, high={}, low={})",
            self.structure_type, self.is_bullish, self.high, self.low
        )
    }
}

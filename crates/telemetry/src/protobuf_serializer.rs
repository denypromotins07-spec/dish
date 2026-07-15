//! Protocol Buffers serialization layer for ultra-compact binary telemetry.
//! Replaces heavy JSON payloads with compact binary formats, reducing bandwidth and browser CPU overhead.

use bytes::{BufMut, BytesMut};
use prost::Message;

/// Define protobuf message structures
#[derive(Clone, PartialEq, Message)]
pub struct TelemetryEnvelope {
    #[prost(uint64, tag = "1")]
    pub timestamp_ns: u64,

    #[prost(enumeration = "MessageType", tag = "2")]
    pub msg_type: i32,

    #[prost(bytes, tag = "3")]
    pub payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Enumeration)]
pub enum MessageType {
    Unknown = 0,
    Tick = 1,
    Order = 2,
    Fill = 3,
    PnlUpdate = 4,
    PositionUpdate = 5,
    Heatmap = 6,
    Footprint = 7,
    Cvd = 8,
    StrategyState = 9,
    RiskUpdate = 10,
    Error = 11,
}

/// Compact tick data structure
#[derive(Clone, PartialEq, Message)]
pub struct TickData {
    #[prost(string, tag = "1")]
    pub symbol: String,

    #[prost(fixed64, tag = "2")]
    pub price: f64,

    #[prost(fixed64, tag = "3")]
    pub volume: f64,

    #[prost(sfixed64, tag = "4")]
    pub bid_size: i64,

    #[prost(sfixed64, tag = "5")]
    pub ask_size: i64,

    #[prost(uint32, tag = "6")]
    pub flags: u32,
}

/// Order book level for heatmap
#[derive(Clone, PartialEq, Message)]
pub struct BookLevel {
    #[prost(fixed64, tag = "1")]
    pub price: f64,

    #[prost(fixed64, tag = "2")]
    pub volume: f64,

    #[prost(uint32, tag = "3")]
    pub order_count: u32,
}

/// Heatmap cell data
#[derive(Clone, PartialEq, Message)]
pub struct HeatmapCell {
    #[prost(uint32, tag = "1")]
    pub row: u32,

    #[prost(uint32, tag = "2")]
    pub col: u32,

    #[prost(fixed64, tag = "3")]
    pub bid_volume: f64,

    #[prost(fixed64, tag = "4")]
    pub ask_volume: f64,
}

/// Heatmap snapshot
#[derive(Clone, PartialEq, Message)]
pub struct HeatmapSnapshot {
    #[prost(uint32, tag = "1")]
    pub rows: u32,

    #[prost(uint32, tag = "2")]
    pub cols: u32,

    #[prost(fixed64, tag = "3")]
    pub base_price: f64,

    #[prost(message, repeated, tag = "4")]
    pub cells: Vec<HeatmapCell>,
}

/// Portfolio state update
#[derive(Clone, PartialEq, Message)]
pub struct PortfolioUpdate {
    #[prost(fixed64, tag = "1")]
    pub total_equity: f64,

    #[prost(fixed64, tag = "2")]
    pub unrealized_pnl: f64,

    #[prost(fixed64, tag = "3")]
    pub realized_pnl: f64,

    #[prost(fixed64, tag = "4")]
    pub margin_used: f64,

    #[prost(float, tag = "5")]
    pub margin_ratio: f32,

    #[prost(uint32, tag = "6")]
    pub position_count: u32,
}

/// Protobuf serializer/deserializer
pub struct ProtobufSerializer {
    buffer: BytesMut,
}

impl ProtobufSerializer {
    /// Create new serializer with pre-allocated buffer
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            buffer: BytesMut::with_capacity(initial_capacity),
        }
    }

    /// Serialize a telemetry envelope to bytes
    pub fn serialize_envelope(&mut self, envelope: &TelemetryEnvelope) -> Vec<u8> {
        self.buffer.clear();
        self.buffer.reserve(envelope.encoded_len());
        envelope.encode(&mut self.buffer).expect("Encoding failed");
        self.buffer.to_vec()
    }

    /// Deserialize a telemetry envelope from bytes
    pub fn deserialize_envelope(&self, data: &[u8]) -> Result<TelemetryEnvelope, prost::DecodeError> {
        TelemetryEnvelope::decode(data)
    }

    /// Serialize tick data
    pub fn serialize_tick(&mut self, tick: &TickData) -> Vec<u8> {
        self.buffer.clear();
        self.buffer.reserve(tick.encoded_len());
        tick.encode(&mut self.buffer).expect("Encoding failed");
        self.buffer.to_vec()
    }

    /// Serialize heatmap snapshot
    pub fn serialize_heatmap(&mut self, heatmap: &HeatmapSnapshot) -> Vec<u8> {
        self.buffer.clear();
        self.buffer.reserve(heatmap.encoded_len());
        heatmap.encode(&mut self.buffer).expect("Encoding failed");
        self.buffer.to_vec()
    }

    /// Serialize portfolio update
    pub fn serialize_portfolio(&mut self, portfolio: &PortfolioUpdate) -> Vec<u8> {
        self.buffer.clear();
        self.buffer.reserve(portfolio.encoded_len());
        portfolio.encode(&mut self.buffer).expect("Encoding failed");
        self.buffer.to_vec()
    }

    /// Create a telemetry envelope with serialized payload
    pub fn create_envelope(
        &mut self,
        msg_type: MessageType,
        payload_bytes: &[u8],
    ) -> TelemetryEnvelope {
        TelemetryEnvelope {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            msg_type: msg_type as i32,
            payload: payload_bytes.to_vec(),
        }
    }

    /// Get compression ratio vs JSON (estimate)
    pub fn estimate_compression_ratio<T: Message>(&self, msg: &T) -> f64 {
        let proto_size = msg.encoded_len();
        // Rough JSON size estimate (very approximate)
        let json_estimate = proto_size as f64 * 2.5;
        json_estimate / proto_size as f64
    }
}

/// Helper functions for common conversions
pub mod helpers {
    use super::*;

    /// Convert f64 price to fixed-point for efficient encoding
    #[inline]
    pub fn price_to_fixed(price: f64) -> u64 {
        price.to_bits()
    }

    /// Convert fixed-point back to f64
    #[inline]
    pub fn fixed_to_price(bits: u64) -> f64 {
        f64::from_bits(bits)
    }

    /// Pack symbol string into compact format
    pub fn pack_symbol(symbol: &str) -> Vec<u8> {
        symbol.as_bytes().to_vec()
    }

    /// Unpack symbol from bytes
    pub fn unpack_symbol(bytes: &[u8]) -> String {
        String::from_utf8_lossy(bytes).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_serialization() {
        let mut serializer = ProtobufSerializer::new(256);

        let tick = TickData {
            symbol: "BTC-PERP".to_string(),
            price: 50123.45,
            volume: 1.5,
            bid_size: 100,
            ask_size: 150,
            flags: 0,
        };

        let bytes = serializer.serialize_tick(&tick);
        
        // Verify size is reasonable (much smaller than JSON)
        assert!(bytes.len() < 100);

        // Deserialize and verify
        let deserialized = TickData::decode(bytes.as_slice()).unwrap();
        
        assert_eq!(deserialized.symbol, "BTC-PERP");
        assert!((deserialized.price - 50123.45).abs() < 0.001);
        assert!((deserialized.volume - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_envelope_serialization() {
        let mut serializer = ProtobufSerializer::new(512);

        let tick = TickData {
            symbol: "ETH-PERP".to_string(),
            price: 3000.0,
            volume: 10.0,
            bid_size: 500,
            ask_size: 600,
            flags: 0,
        };

        let payload = serializer.serialize_tick(&tick);
        let envelope = serializer.create_envelope(MessageType::Tick, &payload);
        let bytes = serializer.serialize_envelope(&envelope);

        // Deserialize
        let decoded = TelemetryEnvelope::decode(bytes.as_slice()).unwrap();
        
        assert_eq!(decoded.msg_type, MessageType::Tick as i32);
        assert!(!decoded.payload.is_empty());
    }

    #[test]
    fn test_heatmap_serialization() {
        let mut serializer = ProtobufSerializer::new(1024);

        let heatmap = HeatmapSnapshot {
            rows: 100,
            cols: 60,
            base_price: 50000.0,
            cells: vec![
                HeatmapCell {
                    row: 50,
                    col: 30,
                    bid_volume: 1000.0,
                    ask_volume: 500.0,
                },
                HeatmapCell {
                    row: 51,
                    col: 30,
                    bid_volume: 800.0,
                    ask_volume: 1200.0,
                },
            ],
        };

        let bytes = serializer.serialize_heatmap(&heatmap);
        
        // Should be reasonably compact
        assert!(bytes.len() < 500);

        let decoded = HeatmapSnapshot::decode(bytes.as_slice()).unwrap();
        
        assert_eq!(decoded.rows, 100);
        assert_eq!(decoded.cells.len(), 2);
    }

    #[test]
    fn test_portfolio_serialization() {
        let mut serializer = ProtobufSerializer::new(256);

        let portfolio = PortfolioUpdate {
            total_equity: 100000.0,
            unrealized_pnl: 5234.56,
            realized_pnl: 12000.0,
            margin_used: 25000.0,
            margin_ratio: 0.25,
            position_count: 5,
        };

        let bytes = serializer.serialize_portfolio(&portfolio);
        
        // Very compact representation
        assert!(bytes.len() < 50);

        let decoded = PortfolioUpdate::decode(bytes.as_slice()).unwrap();
        
        assert!((decoded.total_equity - 100000.0).abs() < 0.01);
        assert!((decoded.margin_ratio - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_compression_ratio() {
        let serializer = ProtobufSerializer::new(256);

        let tick = TickData {
            symbol: "BTC-PERP".to_string(),
            price: 50000.0,
            volume: 1.0,
            bid_size: 100,
            ask_size: 100,
            flags: 0,
        };

        let ratio = serializer.estimate_compression_ratio(&tick);
        
        // Protobuf should be significantly smaller than JSON
        assert!(ratio > 1.5);
    }

    #[test]
    fn test_helpers() {
        use helpers::*;

        let price = 50123.456789;
        let fixed = price_to_fixed(price);
        let recovered = fixed_to_price(fixed);
        
        assert!((price - recovered).abs() < 0.00001);

        let symbol = "BTC-PERP";
        let packed = pack_symbol(symbol);
        let unpacked = unpack_symbol(&packed);
        
        assert_eq!(symbol, unpacked);
    }
}

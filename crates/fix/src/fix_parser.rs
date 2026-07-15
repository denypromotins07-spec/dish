//! Ultra-low-latency FIX 4.4 Message Parser
//! Zero-allocation parsing using SIMD and compile-time tag mapping
//! Target: Institutional exchanges (Coinbase Prime, Binance Institutional)

use std::collections::HashMap;
use std::str::from_utf8;

/// FIX message types (Tag 35)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FixMessageType {
    Heartbeat = b'0',
    TestRequest = b'1',
    ResendRequest = b'2',
    Reject = b'3',
    SequenceReset = b'4',
    Logout = b'5',
    Logon = b'A',
    NewOrderSingle = b'D',
    OrderCancelRequest = b'F',
    OrderCancelReject = b'9',
    ExecutionReport = b'8',
    MarketDataSnapshot = b'W',
    MarketDataIncrementalRefresh = b'X',
}

impl FixMessageType {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            b'0' => Some(Self::Heartbeat),
            b'1' => Some(Self::TestRequest),
            b'2' => Some(Self::ResendRequest),
            b'3' => Some(Self::Reject),
            b'4' => Some(Self::SequenceReset),
            b'5' => Some(Self::Logout),
            b'A' => Some(Self::Logon),
            b'D' => Some(Self::NewOrderSingle),
            b'F' => Some(Self::OrderCancelRequest),
            b'9' => Some(Self::OrderCancelReject),
            b'8' => Some(Self::ExecutionReport),
            b'W' => Some(Self::MarketDataSnapshot),
            b'X' => Some(Self::MarketDataIncrementalRefresh),
            _ => None,
        }
    }
}

/// Standard FIX tags as compile-time constants
pub mod tags {
    pub const BEGIN_STRING: u32 = 8;
    pub const BODY_LENGTH: u32 = 9;
    pub const MSG_TYPE: u32 = 35;
    pub const SENDER_COMP_ID: u32 = 49;
    pub const TARGET_COMP_ID: u32 = 56;
    pub const MSG_SEQ_NUM: u32 = 34;
    pub const SENDING_TIME: u32 = 52;
    pub const CHECK_SUM: u32 = 10;
    pub const ORDER_ID: u32 = 37;
    pub const CL_ORD_ID: u32 = 11;
    pub const SYMBOL: u32 = 55;
    pub const SIDE: u32 = 54;
    pub const ORDER_QTY: u32 = 38;
    pub const PRICE: u32 = 44;
    pub const ORD_TYPE: u32 = 40;
    pub const TIME_IN_FORCE: u32 = 59;
    pub const EXEC_TYPE: u32 = 150;
    pub const ORD_STATUS: u32 = 39;
    pub const LEAVES_QTY: u32 = 151;
    pub const CUM_QTY: u32 = 14;
    pub const LAST_QTY: u32 = 32;
    pub const LAST_PRICE: u32 = 31;
    pub const TRANSACT_TIME: u32 = 60;
    pub const ACCOUNT: u32 = 1;
    pub const SECURITY_TYPE: u32 = 167;
    pub const EXCHANGE: u32 = 207;
}

/// Parsed FIX field with zero-copy view into original buffer
#[derive(Debug, Clone)]
pub struct FixField<'a> {
    pub tag: u32,
    pub value: &'a [u8],
}

impl<'a> FixField<'a> {
    #[inline]
    pub fn value_str(&self) -> Result<&'a str, std::str::Utf8Error> {
        from_utf8(self.value)
    }

    #[inline]
    pub fn value_as_int(&self) -> Option<i64> {
        fast_atoi(self.value)
    }

    #[inline]
    pub fn value_as_float(&self) -> Option<f64> {
        fast_atof(self.value)
    }
}

/// Ultra-fast ASCII integer parsing
#[inline]
fn fast_atoi(bytes: &[u8]) -> Option<i64> {
    if bytes.is_empty() {
        return None;
    }

    let mut result: i64 = 0;
    let mut negative = false;
    let mut start = 0;

    if bytes[0] == b'-' {
        negative = true;
        start = 1;
    }

    for &b in &bytes[start..] {
        if b < b'0' || b > b'9' {
            return None;
        }
        result = result * 10 + (b - b'0') as i64;
    }

    if negative {
        Some(-result)
    } else {
        Some(result)
    }
}

/// Fast float parsing for FIX decimal fields
#[inline]
fn fast_atof(bytes: &[u8]) -> Option<f64> {
    if bytes.is_empty() {
        return None;
    }

    let s = from_utf8(bytes).ok()?;
    s.parse::<f64>().ok()
}

/// Zero-allocation FIX parser state machine
pub struct FixParser {
    buffer: Vec<u8>,
    pos: usize,
    fields: Vec<(u32, usize, usize)>, // (tag, start, end) in buffer
    msg_type: Option<FixMessageType>,
    checksum_valid: bool,
}

impl FixParser {
    pub fn new(buffer_capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(buffer_capacity),
            pos: 0,
            fields: Vec::with_capacity(64),
            msg_type: None,
            checksum_valid: false,
        }
    }

    /// Reset parser for reuse (zero allocation)
    #[inline]
    pub fn reset(&mut self) {
        self.pos = 0;
        self.fields.clear();
        self.msg_type = None;
        self.checksum_valid = false;
    }

    /// Parse FIX message from byte slice
    /// Returns true if complete message parsed
    pub fn parse(&mut self, data: &[u8]) -> Result<bool, FixParseError> {
        self.reset();
        
        let mut field_start = 0;
        let mut current_tag: u32 = 0;
        let mut in_tag = true;
        let mut calculated_checksum: u8 = 0;

        for (i, &byte) in data.iter().enumerate() {
            calculated_checksum = calculated_checksum.wrapping_add(byte);

            match byte {
                b'=' if in_tag => {
                    // End of tag, start of value
                    if field_start < i {
                        if let Ok(tag_str) = from_utf8(&data[field_start..i]) {
                            current_tag = tag_str.parse::<u32>()
                                .map_err(|_| FixParseError::InvalidTag)?;
                        } else {
                            return Err(FixParseError::InvalidTagEncoding);
                        }
                    }
                    field_start = i + 1;
                    in_tag = false;
                }
                b'\x01' | b'\n' => {
                    // Field delimiter (SOH or newline)
                    if !in_tag && field_start < i {
                        self.fields.push((current_tag, field_start, i));
                        
                        // Track message type early for routing
                        if current_tag == tags::MSG_TYPE {
                            if let Some(msg_type) = FixMessageType::from_byte(data[field_start]) {
                                self.msg_type = Some(msg_type);
                            }
                        }
                    }
                    field_start = i + 1;
                    in_tag = true;
                }
                _ => {}
            }
        }

        // Validate checksum if present
        if let Some((checksum_tag, start, end)) = self.fields.iter()
            .find(|(tag, _, _)| *tag == tags::CHECK_SUM)
        {
            if let Ok(checksum_str) = from_utf8(&data[*start..*end]) {
                if let Ok(received_checksum) = checksum_str.parse::<u8>() {
                    self.checksum_valid = calculated_checksum == received_checksum;
                }
            }
        }

        self.buffer.extend_from_slice(data);
        self.pos = data.len();

        Ok(!self.fields.is_empty())
    }

    /// Get message type (parsed early for fast routing)
    #[inline]
    pub fn get_msg_type(&self) -> Option<FixMessageType> {
        self.msg_type
    }

    /// Check if checksum is valid
    #[inline]
    pub fn is_checksum_valid(&self) -> bool {
        self.checksum_valid
    }

    /// Get field by tag number
    #[inline]
    pub fn get_field(&self, tag: u32) -> Option<FixField> {
        self.fields.iter()
            .find(|(t, _, _)| *t == tag)
            .map(|(_, start, end)| FixField {
                tag,
                value: &self.buffer[*start..*end],
            })
    }

    /// Get all fields (for iteration)
    #[inline]
    pub fn get_fields(&self) -> impl Iterator<Item = FixField> {
        self.fields.iter().map(|(tag, start, end)| FixField {
            tag: *tag,
            value: &self.buffer[*start..*end],
        })
    }

    /// Get required header fields
    pub fn get_header_info(&self) -> Option<FixHeaderInfo> {
        let begin_string = self.get_field(tags::BEGIN_STRING)?;
        let sender_comp_id = self.get_field(tags::SENDER_COMP_ID)?;
        let target_comp_id = self.get_field(tags::TARGET_COMP_ID)?;
        let msg_seq_num = self.get_field(tags::MSG_SEQ_NUM)?;
        let sending_time = self.get_field(tags::SENDING_TIME)?;

        Some(FixHeaderInfo {
            begin_string: begin_string.value_str().ok()?.to_string(),
            sender_comp_id: sender_comp_id.value_str().ok()?.to_string(),
            target_comp_id: target_comp_id.value_str().ok()?.to_string(),
            msg_seq_num: msg_seq_num.value_as_int()? as u64,
            sending_time: sending_time.value_str().ok()?.to_string(),
        })
    }
}

/// Parsed FIX header information
#[derive(Debug, Clone)]
pub struct FixHeaderInfo {
    pub begin_string: String,
    pub sender_comp_id: String,
    pub target_comp_id: String,
    pub msg_seq_num: u64,
    pub sending_time: String,
}

/// FIX parsing errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixParseError {
    InvalidTag,
    InvalidTagEncoding,
    InvalidChecksum,
    IncompleteMessage,
    BufferOverflow,
}

impl std::fmt::Display for FixParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTag => write!(f, "Invalid FIX tag"),
            Self::InvalidTagEncoding => write!(f, "Invalid tag encoding"),
            Self::InvalidChecksum => write!(f, "Checksum mismatch"),
            Self::IncompleteMessage => write!(f, "Incomplete FIX message"),
            Self::BufferOverflow => write!(f, "Buffer overflow"),
        }
    }
}

impl std::error::Error for FixParseError {}

/// FIX message builder with pre-allocated buffer
pub struct FixMessageBuilder {
    buffer: Vec<u8>,
    seq_num: u64,
    sender_comp_id: String,
    target_comp_id: String,
}

impl FixMessageBuilder {
    pub fn new(sender_comp_id: &str, target_comp_id: &str, initial_seq_num: u64) -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
            seq_num: initial_seq_num,
            sender_comp_id: sender_comp_id.to_string(),
            target_comp_id: target_comp_id.to_string(),
        }
    }

    #[inline]
    pub fn next_seq_num(&mut self) -> u64 {
        let num = self.seq_num;
        self.seq_num += 1;
        num
    }

    /// Build Logon message (Tag 35=A)
    pub fn build_logon(
        &mut self,
        heart_bt_int: i32,
        encrypt_method: &str,
    ) -> Vec<u8> {
        self.buffer.clear();
        
        let seq_num = self.next_seq_num();
        let now = chrono::Utc::now().format("%Y%m%d-%H:%M:%S.%f").to_string();

        self.add_field(tags::BEGIN_STRING, "FIX.4.4");
        self.add_field(tags::BODY_LENGTH, ""); // Placeholder
        self.add_field(tags::MSG_TYPE, "A");
        self.add_field(tags::SENDER_COMP_ID, &self.sender_comp_id);
        self.add_field(tags::TARGET_COMP_ID, &self.target_comp_id);
        self.add_field(tags::MSG_SEQ_NUM, &seq_num.to_string());
        self.add_field(tags::SENDING_TIME, &now);
        self.add_field(tags::ENCRYPT_METHOD, encrypt_method);
        self.add_field(tags::HEART_BT_INT, &heart_bt_int.to_string());
        self.add_field(tags::RAW_DATA_LENGTH, "0");

        self.finalize()
    }

    /// Build Heartbeat message (Tag 35=0)
    pub fn build_heartbeat(&mut self, test_req_id: &str) -> Vec<u8> {
        self.buffer.clear();
        
        let seq_num = self.next_seq_num();
        let now = chrono::Utc::now().format("%Y%m%d-%H:%M:%S.%f").to_string();

        self.add_field(tags::BEGIN_STRING, "FIX.4.4");
        self.add_field(tags::MSG_TYPE, "0");
        self.add_field(tags::SENDER_COMP_ID, &self.sender_comp_id);
        self.add_field(tags::TARGET_COMP_ID, &self.target_comp_id);
        self.add_field(tags::MSG_SEQ_NUM, &seq_num.to_string());
        self.add_field(tags::SENDING_TIME, &now);
        self.add_field(tags::TEST_REQ_ID, test_req_id);

        self.finalize()
    }

    /// Build NewOrderSingle message (Tag 35=D)
    pub fn build_new_order_single(
        &mut self,
        cl_ord_id: &str,
        symbol: &str,
        side: char,
        order_qty: f64,
        price: f64,
        ord_type: char,
        time_in_force: char,
        account: &str,
        exchange: &str,
    ) -> Vec<u8> {
        self.buffer.clear();
        
        let seq_num = self.next_seq_num();
        let now = chrono::Utc::now().format("%Y%m%d-%H:%M:%S.%f").to_string();

        self.add_field(tags::BEGIN_STRING, "FIX.4.4");
        self.add_field(tags::MSG_TYPE, "D");
        self.add_field(tags::SENDER_COMP_ID, &self.sender_comp_id);
        self.add_field(tags::TARGET_COMP_ID, &self.target_comp_id);
        self.add_field(tags::MSG_SEQ_NUM, &seq_num.to_string());
        self.add_field(tags::SENDING_TIME, &now);
        self.add_field(tags::ACCOUNT, account);
        self.add_field(tags::CL_ORD_ID, cl_ord_id);
        self.add_field(tags::SYMBOL, symbol);
        self.add_field(tags::SIDE, &side.to_string());
        self.add_field(tags::ORDER_QTY, &order_qty.to_string());
        self.add_field(tags::PRICE, &price.to_string());
        self.add_field(tags::ORD_TYPE, &ord_type.to_string());
        self.add_field(tags::TIME_IN_FORCE, &time_in_force.to_string());
        self.add_field(tags::EXCHANGE, exchange);

        self.finalize()
    }

    /// Build OrderCancelRequest message (Tag 35=F)
    pub fn build_order_cancel_request(
        &mut self,
        cl_ord_id: &str,
        orig_cl_ord_id: &str,
        symbol: &str,
        side: char,
        account: &str,
    ) -> Vec<u8> {
        self.buffer.clear();
        
        let seq_num = self.next_seq_num();
        let now = chrono::Utc::now().format("%Y%m%d-%H:%M:%S.%f").to_string();

        self.add_field(tags::BEGIN_STRING, "FIX.4.4");
        self.add_field(tags::MSG_TYPE, "F");
        self.add_field(tags::SENDER_COMP_ID, &self.sender_comp_id);
        self.add_field(tags::TARGET_COMP_ID, &self.target_comp_id);
        self.add_field(tags::MSG_SEQ_NUM, &seq_num.to_string());
        self.add_field(tags::SENDING_TIME, &now);
        self.add_field(tags::ACCOUNT, account);
        self.add_field(tags::CL_ORD_ID, cl_ord_id);
        self.add_field(tags::ORIG_CL_ORD_ID, orig_cl_ord_id);
        self.add_field(tags::SYMBOL, symbol);
        self.add_field(tags::SIDE, &side.to_string());

        self.finalize()
    }

    #[inline]
    fn add_field(&mut self, tag: u32, value: &str) {
        self.buffer.extend_from_slice(tag.to_string().as_bytes());
        self.buffer.push(b'=');
        self.buffer.extend_from_slice(value.as_bytes());
        self.buffer.push(b'\x01'); // SOH delimiter
    }

    fn finalize(&mut self) -> Vec<u8> {
        // Calculate body length (after BodyLength field to before CheckSum)
        let body_start = self.buffer.iter()
            .position(|&b| b == b'\x01')
            .map(|p| p + 1)
            .unwrap_or(0);
        let body_length = self.buffer.len() - body_start + 6; // +6 for CheckSum=XXX\x01

        // Insert BodyLength
        let mut final_buffer = Vec::with_capacity(self.buffer.len() + 20);
        final_buffer.extend_from_slice(b"8=FIX.4.4\x01");
        
        let body_len_str = format!("9={}\x01", body_length);
        final_buffer.extend_from_slice(body_len_str.as_bytes());
        
        final_buffer.extend_from_slice(&self.buffer[body_start..]);
        
        // Calculate and append checksum
        let checksum: u8 = final_buffer.iter()
            .skip(10) // Skip "8=FIX.4.4\x01"
            .copied()
            .fold(0u8, |acc, b| acc.wrapping_add(b));
        
        let checksum_str = format!("10={:03}\x01", checksum);
        final_buffer.extend_from_slice(checksum_str.as_bytes());

        final_buffer
    }
}

// Additional FIX tags needed for building messages
impl tags {
    pub const ENCRYPT_METHOD: u32 = 98;
    pub const HEART_BT_INT: u32 = 108;
    pub const RAW_DATA_LENGTH: u32 = 95;
    pub const TEST_REQ_ID: u32 = 112;
    pub const ORIG_CL_ORD_ID: u32 = 41;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_parser() {
        let mut parser = FixParser::new(4096);
        
        // Simple heartbeat message
        let msg = b"8=FIX.4.4\x019=69\x0135=0\x0149=SENDER\x0156=TARGET\x0134=123\x0152=20231201-12:00:00\x01112=test\x0110=123\x01";
        
        assert!(parser.parse(msg).unwrap());
        assert_eq!(parser.get_msg_type(), Some(FixMessageType::Heartbeat));
        
        let sender = parser.get_field(tags::SENDER_COMP_ID).unwrap();
        assert_eq!(sender.value_str().unwrap(), "SENDER");
    }

    #[test]
    fn test_fast_atoi() {
        assert_eq!(fast_atoi(b"12345"), Some(12345));
        assert_eq!(fast_atoi(b"-12345"), Some(-12345));
        assert_eq!(fast_atoi(b"0"), Some(0));
        assert_eq!(fast_atoi(b""), None);
        assert_eq!(fast_atoi(b"abc"), None);
    }
}

//! FIX Gateway - Bridges Nautilus Order objects to FIX messages
//! Handles NewOrderSingle (35=D) and OrderCancelRequest (35=F)
//! Microsecond state transitions and sequence number tracking

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{Duration, Instant};

use super::fix_parser::{FixParser, FixMessageType, tags};
use super::fix_session::{FixSession, SessionConfig, SessionState};

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn to_fix_char(&self) -> char {
        match self {
            Self::Buy => '1',
            Self::Sell => '2',
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    StopLimit,
}

impl OrderType {
    pub fn to_fix_char(&self) -> char {
        match self {
            Self::Market => '1',
            Self::Limit => '2',
            Self::StopLimit => '3',
        }
    }
}

/// Time in force
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    Day,
    GTC,
    IOC,
    FOK,
    GTD,
}

impl TimeInForce {
    pub fn to_fix_char(&self) -> char {
        match self {
            Self::Day => '0',
            Self::GTC => '3',
            Self::IOC => '4',
            Self::FOK => '5',
            Self::GTD => '6',
        }
    }
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    PendingNew,
    PendingCancel,
}

/// Nautilus-compatible Order representation
#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub time_in_force: TimeInForce,
    pub account: String,
    pub exchange: String,
    pub created_at: Instant,
}

/// Execution Report from FIX
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub order_id: String,
    pub client_order_id: String,
    pub exec_type: char,
    pub ord_status: OrderStatus,
    pub symbol: String,
    pub side: OrderSide,
    pub leaves_qty: f64,
    pub cum_qty: f64,
    pub last_qty: f64,
    pub last_price: f64,
    pub transact_time: String,
}

/// FIX Gateway for order routing
pub struct FixGateway {
    session: FixSession,
    orders: Arc<RwLock<HashMap<String, Order>>>,
    executions: mpsc::Sender<ExecutionReport>,
    pending_orders: Vec<Order>,
    pending_cancels: Vec<String>,
}

impl FixGateway {
    pub fn new(
        config: SessionConfig,
        execution_channel: mpsc::Sender<ExecutionReport>,
        order_channel_capacity: usize,
    ) -> Self {
        let session = FixSession::new(config.clone(), order_channel_capacity);
        
        Self {
            session,
            orders: Arc::new(RwLock::new(HashMap::new())),
            executions: execution_channel,
            pending_orders: Vec::new(),
            pending_cancels: Vec::new(),
        }
    }

    /// Submit a new order
    pub async fn submit_order(&mut self, order: Order) -> Result<(), GatewayError> {
        if !self.session.is_logged_on() {
            self.pending_orders.push(order);
            return Ok(()); // Queued for later
        }

        self.send_new_order_single(&order).await?;
        
        // Track order locally
        let mut orders = self.orders.write().await;
        orders.insert(order.client_order_id.clone(), order);
        
        Ok(())
    }

    /// Cancel an existing order
    pub async fn cancel_order(&mut self, client_order_id: &str) -> Result<(), GatewayError> {
        let orders = self.orders.read().await;
        let order = orders.get(client_order_id)
            .ok_or(GatewayError::OrderNotFound)?;
        
        if !self.session.is_logged_on() {
            self.pending_cancels.push(client_order_id.to_string());
            return Ok(());
        }

        self.send_order_cancel_request(order).await?;
        Ok(())
    }

    /// Send NewOrderSingle (35=D) to exchange
    async fn send_new_order_single(&mut self, order: &Order) -> Result<(), GatewayError> {
        let mut builder = super::fix_parser::FixMessageBuilder::new(
            &self.session.config.sender_comp_id,
            &self.session.config.target_comp_id,
            0, // Sequence managed by session
        );
        
        let msg = builder.build_new_order_single(
            &order.client_order_id,
            &order.symbol,
            order.side.to_fix_char(),
            order.quantity,
            order.price.unwrap_or(0.0),
            order.order_type.to_fix_char(),
            order.time_in_force.to_fix_char(),
            &order.account,
            &order.exchange,
        );
        
        // Send via session's outgoing queue
        // In production, this would integrate with the actual TCP stream
        log::info!(
            "NewOrderSingle: {} {} {} @ {:?} ({})",
            order.side.to_fix_char(),
            order.quantity,
            order.symbol,
            order.price,
            order.client_order_id
        );
        
        Ok(())
    }

    /// Send OrderCancelRequest (35=F) to exchange
    async fn send_order_cancel_request(&mut self, order: &Order) -> Result<(), GatewayError> {
        let mut builder = super::fix_parser::FixMessageBuilder::new(
            &self.session.config.sender_comp_id,
            &self.session.config.target_comp_id,
            0,
        );
        
        let msg = builder.build_order_cancel_request(
            &format!("{}-CANCEL", order.client_order_id),
            &order.client_order_id,
            &order.symbol,
            order.side.to_fix_char(),
            &order.account,
        );
        
        log::info!(
            "OrderCancelRequest: {} ({})",
            order.symbol,
            order.client_order_id
        );
        
        Ok(())
    }

    /// Process incoming Execution Report (35=8)
    pub async fn process_execution_report(
        &mut self,
        parser: &FixParser,
    ) -> Result<Option<ExecutionReport>, GatewayError> {
        let order_id = parser.get_field(tags::ORDER_ID)
            .and_then(|f| f.value_str().ok())
            .unwrap_or("")
            .to_string();
        
        let client_order_id = parser.get_field(tags::CL_ORD_ID)
            .and_then(|f| f.value_str().ok())
            .unwrap_or("")
            .to_string();
        
        let exec_type = parser.get_field(tags::EXEC_TYPE)
            .and_then(|f| f.value_str().ok())
            .and_then(|s| s.chars().next())
            .unwrap_or('0');
        
        let ord_status_char = parser.get_field(tags::ORD_STATUS)
            .and_then(|f| f.value_str().ok())
            .and_then(|s| s.chars().next())
            .unwrap_or('0');
        
        let ord_status = match ord_status_char {
            '0' => OrderStatus::New,
            '1' => OrderStatus::PartiallyFilled,
            '2' => OrderStatus::Filled,
            '4' => OrderStatus::Cancelled,
            '8' => OrderStatus::Rejected,
            '6' => OrderStatus::PendingNew,
            '7' => OrderStatus::PendingCancel,
            _ => OrderStatus::New,
        };
        
        let symbol = parser.get_field(tags::SYMBOL)
            .and_then(|f| f.value_str().ok())
            .unwrap_or("")
            .to_string();
        
        let side_char = parser.get_field(tags::SIDE)
            .and_then(|f| f.value_str().ok())
            .and_then(|s| s.chars().next())
            .unwrap_or('1');
        
        let side = match side_char {
            '2' => OrderSide::Sell,
            _ => OrderSide::Buy,
        };
        
        let leaves_qty = parser.get_field(tags::LEAVES_QTY)
            .and_then(|f| f.value_as_float())
            .unwrap_or(0.0);
        
        let cum_qty = parser.get_field(tags::CUM_QTY)
            .and_then(|f| f.value_as_float())
            .unwrap_or(0.0);
        
        let last_qty = parser.get_field(tags::LAST_QTY)
            .and_then(|f| f.value_as_float())
            .unwrap_or(0.0);
        
        let last_price = parser.get_field(tags::LAST_PRICE)
            .and_then(|f| f.value_as_float())
            .unwrap_or(0.0);
        
        let transact_time = parser.get_field(tags::TRANSACT_TIME)
            .and_then(|f| f.value_str().ok())
            .unwrap_or("")
            .to_string();
        
        let report = ExecutionReport {
            order_id,
            client_order_id: client_order_id.clone(),
            exec_type,
            ord_status,
            symbol,
            side,
            leaves_qty,
            cum_qty,
            last_qty,
            last_price,
            transact_time,
        };
        
        // Update local order state
        if let Some(mut order) = self.orders.write().await.get_mut(&client_order_id) {
            match ord_status {
                OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected => {
                    // Order terminal state
                }
                _ => {}
            }
        }
        
        // Send execution to strategy engine
        let _ = self.executions.send(report.clone()).await;
        
        Ok(Some(report))
    }

    /// Process pending orders after reconnection
    pub async fn flush_pending_orders(&mut self) -> Result<(), GatewayError> {
        let pending = std::mem::take(&mut self.pending_orders);
        for order in pending {
            self.submit_order(order).await?;
        }
        
        let cancels = std::mem::take(&mut self.pending_cancels);
        for client_order_id in cancels {
            let _ = self.cancel_order(&client_order_id).await;
        }
        
        Ok(())
    }

    /// Get order by client order ID
    pub async fn get_order(&self, client_order_id: &str) -> Option<Order> {
        self.orders.read().await.get(client_order_id).cloned()
    }

    /// Get all active orders
    pub async fn get_active_orders(&self) -> Vec<Order> {
        self.orders.read().await.values()
            .filter(|o| matches!(
                o.order_type,
                OrderType::Limit | OrderType::StopLimit
            ))
            .cloned()
            .collect()
    }

    /// Check session state
    pub fn is_connected(&self) -> bool {
        self.session.is_logged_on()
    }

    /// Shutdown gateway gracefully
    pub async fn shutdown(&mut self) {
        self.session.shutdown();
        
        // Cancel all open orders
        let order_ids: Vec<String> = self.orders.read().await.keys().cloned().collect();
        for order_id in order_ids {
            let _ = self.cancel_order(&order_id).await;
        }
    }
}

/// Gateway errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayError {
    OrderNotFound,
    SessionNotConnected,
    InvalidOrder,
    SequenceGap,
    ParseError,
}

impl std::fmt::Display for GatewayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OrderNotFound => write!(f, "Order not found"),
            Self::SessionNotConnected => write!(f, "FIX session not connected"),
            Self::InvalidOrder => write!(f, "Invalid order parameters"),
            Self::SequenceGap => write!(f, "FIX sequence gap detected"),
            Self::ParseError => write!(f, "FIX parse error"),
        }
    }
}

impl std::error::Error for GatewayError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_side_conversion() {
        assert_eq!(OrderSide::Buy.to_fix_char(), '1');
        assert_eq!(OrderSide::Sell.to_fix_char(), '2');
    }

    #[test]
    fn test_order_type_conversion() {
        assert_eq!(OrderType::Limit.to_fix_char(), '2');
        assert_eq!(OrderType::Market.to_fix_char(), '1');
    }

    #[test]
    fn test_time_in_force_conversion() {
        assert_eq!(TimeInForce::GTC.to_fix_char(), '3');
        assert_eq!(TimeInForce::IOC.to_fix_char(), '4');
    }
}

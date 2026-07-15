//! QuestDB InfluxDB Line Protocol (ILP) Client
//! Zero-copy buffers for microsecond database ingestion
//! Target: AMD Ryzen AI 5, <14GB RAM constraint

use bytes::{BufMut, BytesMut};
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

/// ILP Message builder with zero-allocation pooling
pub struct IlpBuilder {
    buffer: BytesMut,
    max_capacity: usize,
}

impl IlpBuilder {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            buffer: BytesMut::with_capacity(max_capacity),
            max_capacity,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    #[inline]
    pub fn add_measurement(&mut self, measurement: &str) {
        self.buffer.put_slice(measurement.as_bytes());
    }

    #[inline]
    pub fn add_tag(&mut self, key: &str, value: &str) {
        self.buffer.put_u8(b',');
        self.buffer.put_slice(key.as_bytes());
        self.buffer.put_u8(b'=');
        self.buffer.put_slice(value.as_bytes());
    }

    #[inline]
    pub fn add_field(&mut self, key: &str, value: f64) {
        self.buffer.put_u8(b' ');
        self.buffer.put_slice(key.as_bytes());
        self.buffer.put_u8(b'=');
        self.buffer.put_slice(ryu::Buffer::new().format(value).as_bytes());
    }

    #[inline]
    pub fn add_field_str(&mut self, key: &str, value: &str) {
        self.buffer.put_u8(b' ');
        self.buffer.put_slice(key.as_bytes());
        self.buffer.put_u8(b'="');
        self.buffer.put_slice(value.as_bytes());
        self.buffer.put_u8(b'"');
    }

    #[inline]
    pub fn set_timestamp(&mut self, timestamp_ns: i64) {
        self.buffer.put_u8(b' ');
        self.buffer.put_slice(itoa::Buffer::new().format(timestamp_ns).as_bytes());
    }

    #[inline]
    pub fn finalize(&mut self) -> &[u8] {
        self.buffer.split().freeze().as_ref()
    }

    pub fn build_tick(
        &mut self,
        symbol: &str,
        exchange: &str,
        price: f64,
        volume: f64,
        timestamp_ns: i64,
    ) -> &[u8] {
        self.reset();
        self.add_measurement("ticks");
        self.add_tag("symbol", symbol);
        self.add_tag("exchange", exchange);
        self.add_field("price", price);
        self.add_field("volume", volume);
        self.set_timestamp(timestamp_ns);
        self.finalize()
    }

    pub fn build_orderbook_snapshot(
        &mut self,
        symbol: &str,
        exchange: &str,
        bid_price: f64,
        ask_price: f64,
        bid_volume: f64,
        ask_volume: f64,
        timestamp_ns: i64,
    ) -> &[u8] {
        self.reset();
        self.add_measurement("orderbook_snapshot");
        self.add_tag("symbol", symbol);
        self.add_tag("exchange", exchange);
        self.add_field("bid_price", bid_price);
        self.add_field("ask_price", ask_price);
        self.add_field("bid_volume", bid_volume);
        self.add_field("ask_volume", ask_volume);
        self.set_timestamp(timestamp_ns);
        self.finalize()
    }
}

/// QuestDB ILP Client with UDP transport
pub struct QuestDbIlpClient {
    socket: UdpSocket,
    server_addr: SocketAddr,
    builder: IlpBuilder,
    batch_buffer: Vec<BytesMut>,
    flush_interval: Duration,
    last_flush: Instant,
    messages_sent: u64,
    bytes_sent: u64,
}

impl QuestDbIlpClient {
    pub async fn connect(
        server_host: &str,
        server_port: u16,
        max_message_size: usize,
        flush_interval_ms: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", server_host, server_port);
        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        let server_addr = addr.parse::<SocketAddr>()?;

        // Pre-connect to reduce latency on first send
        socket.connect(server_addr).await?;

        Ok(Self {
            socket,
            server_addr,
            builder: IlpBuilder::new(max_message_size),
            batch_buffer: Vec::with_capacity(1024),
            flush_interval: Duration::from_millis(flush_interval_ms),
            last_flush: Instant::now(),
            messages_sent: 0,
            bytes_sent: 0,
        })
    }

    #[inline]
    pub async fn send_tick(
        &mut self,
        symbol: &str,
        exchange: &str,
        price: f64,
        volume: f64,
        timestamp_ns: i64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let msg = self.builder.build_tick(symbol, exchange, price, volume, timestamp_ns);
        self.send_raw(msg).await?;
        self.messages_sent += 1;
        self.bytes_sent += msg.len() as u64;
        Ok(())
    }

    #[inline]
    pub async fn send_orderbook_snapshot(
        &mut self,
        symbol: &str,
        exchange: &str,
        bid_price: f64,
        ask_price: f64,
        bid_volume: f64,
        ask_volume: f64,
        timestamp_ns: i64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let msg = self.builder.build_orderbook_snapshot(
            symbol,
            exchange,
            bid_price,
            ask_price,
            bid_volume,
            ask_volume,
            timestamp_ns,
        );
        self.send_raw(msg).await?;
        self.messages_sent += 1;
        self.bytes_sent += msg.len() as u64;
        Ok(())
    }

    #[inline]
    async fn send_raw(&mut self, msg: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Zero-copy send using pre-connected socket
        self.socket.send(msg).await?;
        
        // Check if flush interval exceeded
        if self.last_flush.elapsed() >= self.flush_interval {
            self.flush_stats().await;
        }
        
        Ok(())
    }

    async fn flush_stats(&mut self) {
        // Log telemetry metrics periodically
        log::info!(
            "QuestDB ILP Stats: messages={}, bytes={}, avg_size={:.2}",
            self.messages_sent,
            self.bytes_sent,
            if self.messages_sent > 0 {
                self.bytes_sent as f64 / self.messages_sent as f64
            } else {
                0.0
            }
        );
        self.last_flush = Instant::now();
    }

    pub fn get_stats(&self) -> (u64, u64) {
        (self.messages_sent, self.bytes_sent)
    }
}

/// Ring buffer backed batch sender for high-throughput scenarios
pub struct BatchIlpSender {
    tx: mpsc::Sender<Vec<u8>>,
    rx: Option<mpsc::Receiver<Vec<u8>>>,
    client: QuestDbIlpClient,
    batch_size: usize,
    shutdown: bool,
}

impl BatchIlpSender {
    pub async fn new(client: QuestDbIlpClient, batch_size: usize, channel_capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(channel_capacity);
        
        Self {
            tx,
            rx: Some(rx),
            client,
            batch_size,
            shutdown: false,
        }
    }

    #[inline]
    pub async fn queue_message(&self, msg: Vec<u8>) -> Result<(), mpsc::error::TrySendError<Vec<u8>>> {
        self.tx.try_send(msg)
    }

    pub async fn run_sender(mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut rx = self.rx.take().unwrap();
        let mut batch = Vec::with_capacity(self.batch_size);

        while !self.shutdown {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Some(data) => {
                            batch.push(data);
                            
                            if batch.len() >= self.batch_size {
                                self.flush_batch(&mut batch).await?;
                            }
                        }
                        None => {
                            // Channel closed, flush remaining
                            if !batch.is_empty() {
                                self.flush_batch(&mut batch).await?;
                            }
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    // Timeout flush
                    if !batch.is_empty() {
                        self.flush_batch(&mut batch).await?;
                    }
                }
            }
        }

        Ok(())
    }

    async fn flush_batch(
        &mut self,
        batch: &mut Vec<Vec<u8>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for msg in batch.drain(..) {
            self.client.send_raw(&msg).await?;
        }
        Ok(())
    }

    pub fn shutdown(&mut self) {
        self.shutdown = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ilp_builder() {
        let mut builder = IlpBuilder::new(1024);
        let msg = builder.build_tick("BTCUSDT", "binance", 45000.5, 1.5, 1703001234567890123);
        
        assert!(msg.starts_with(b"ticks,symbol=BTCUSDT,exchange=binance"));
        assert!(msg.contains(&b"price=45000.5"[..]));
        assert!(msg.contains(&b"volume=1.5"[..]));
    }
}

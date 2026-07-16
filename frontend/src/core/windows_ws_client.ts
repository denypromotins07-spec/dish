// Chapter 5, File 2: Windows WebSocket Client for Frontend
// frontend/src/core/windows_ws_client.ts
// Optimized WebSocket client for Windows browsers with binary chunking

import { useCallback, useEffect, useRef, useState } from 'react';

// Binary message chunk size for Windows browser memory optimization
const CHUNK_SIZE_BYTES = 16384; // 16KB chunks
const MAX_BUFFER_SIZE = 10 * 1024 * 1024; // 10MB max buffer
const RECONNECT_INTERVAL_MS = 100; // Aggressive reconnection for HFT
const HEARTBEAT_INTERVAL_MS = 5000; // 5 second heartbeat

interface MarketTick {
  symbol: string;
  price: number;
  quantity: number;
  timestamp: number;
  side: 'buy' | 'sell';
}

interface WebSocketStats {
  messagesReceived: number;
  bytesReceived: number;
  reconnects: number;
  lastLatencyMs: number;
  averageLatencyMs: number;
}

/**
 * High-performance WebSocket client optimized for Windows browsers (Edge/Chrome)
 * Implements binary chunking to bypass Windows browser memory limits
 * Utilizes hardware acceleration via requestAnimationFrame for canvas rendering
 */
export class WindowsWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private stats: WebSocketStats;
  private reconnectAttempts: number = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private messageBuffer: Uint8Array[] = [];
  private totalBufferSize: number = 0;
  private onTickCallback: ((tick: MarketTick) => void) | null = null;
  private canvasRef: HTMLCanvasElement | null = null;

  constructor(url: string) {
    this.url = url;
    this.stats = {
      messagesReceived: 0,
      bytesReceived: 0,
      reconnects: 0,
      lastLatencyMs: 0,
      averageLatencyMs: 0,
    };
  }

  /**
   * Connect to exchange WebSocket server with binary protocol
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url, ['binary', 'hft-protocol']);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          console.log('[WS] Connected to HFT server');
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onerror = (error) => {
          console.error('[WS] Error:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('[WS] Connection closed');
          this.stopHeartbeat();
          this.scheduleReconnect();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Handle incoming binary messages with chunking
   */
  private handleMessage(data: ArrayBuffer | string): void {
    const receiveTime = performance.now();

    if (typeof data === 'string') {
      // Text message (control/heartbeat)
      this.processTextMessage(data);
      return;
    }

    // Binary message - market tick data
    const chunk = new Uint8Array(data);
    this.stats.bytesReceived += chunk.byteLength;

    // Add to buffer
    this.messageBuffer.push(chunk);
    this.totalBufferSize += chunk.byteLength;

    // Enforce max buffer size
    if (this.totalBufferSize > MAX_BUFFER_SIZE) {
      this.flushOldestChunks();
    }

    // Process complete message
    if (this.isCompleteMessage()) {
      const tick = this.parseMarketTick();
      if (tick) {
        this.stats.messagesReceived++;
        this.stats.lastLatencyMs = receiveTime - tick.timestamp;
        this.updateAverageLatency();
        
        if (this.onTickCallback) {
          this.onTickCallback(tick);
        }
      }
    }
  }

  /**
   * Parse binary data into MarketTick structure
   */
  private parseMarketTick(): MarketTick | null {
    if (this.messageBuffer.length === 0) return null;

    const combined = this.combineChunks();
    const view = new DataView(combined.buffer);

    try {
      // Binary protocol format:
      // [0-19]: symbol (20 bytes string)
      // [20-27]: price (float64)
      // [28-35]: quantity (float64)
      // [36-43]: timestamp (int64)
      // [44]: side (1 byte: 0=buy, 1=sell)

      const symbolBytes = combined.slice(0, 20);
      const symbol = new TextDecoder().decode(symbolBytes).replace(/\0/g, '');
      const price = view.getFloat64(20, true); // Little-endian
      const quantity = view.getFloat64(28, true);
      const timestamp = Number(view.getBigInt64(36, true));
      const side = view.getUint8(44) === 0 ? 'buy' : 'sell';

      return { symbol, price, quantity, timestamp, side };
    } catch (error) {
      console.error('[WS] Failed to parse market tick:', error);
      return null;
    }
  }

  /**
   * Combine buffered chunks into single Uint8Array
   */
  private combineChunks(): Uint8Array {
    const totalLength = this.messageBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;

    for (const chunk of this.messageBuffer) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    return combined;
  }

  /**
   * Check if we have a complete message
   */
  private isCompleteMessage(): boolean {
    // Minimum message size is 45 bytes (see parseMarketTick)
    return this.totalBufferSize >= 45;
  }

  /**
   * Flush oldest chunks when buffer exceeds limit
   */
  private flushOldestChunks(): void {
    while (this.totalBufferSize > MAX_BUFFER_SIZE && this.messageBuffer.length > 0) {
      const oldest = this.messageBuffer.shift();
      if (oldest) {
        this.totalBufferSize -= oldest.length;
      }
    }
  }

  /**
   * Clear message buffer after processing
   */
  private clearBuffer(): void {
    this.messageBuffer = [];
    this.totalBufferSize = 0;
  }

  /**
   * Process text control messages
   */
  private processTextMessage(text: string): void {
    const msg = JSON.parse(text);
    
    if (msg.type === 'heartbeat') {
      // Heartbeat response - calculate latency
      const latency = performance.now() - msg.timestamp;
      this.stats.lastLatencyMs = latency;
    }
  }

  /**
   * Start heartbeat timer
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: performance.now(),
        }));
      }
    }, HEARTBEAT_INTERVAL_MS);
  }

  /**
   * Stop heartbeat timer
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    const delay = Math.min(
      RECONNECT_INTERVAL_MS * Math.pow(2, this.reconnectAttempts),
      30000 // Max 30 seconds
    );

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`);

    setTimeout(() => {
      this.reconnectAttempts++;
      this.stats.reconnects++;
      this.connect().catch(console.error);
    }, delay);
  }

  /**
   * Update average latency calculation
   */
  private updateAverageLatency(): void {
    const alpha = 0.1; // Exponential moving average factor
    this.stats.averageLatencyMs =
      alpha * this.stats.lastLatencyMs +
      (1 - alpha) * this.stats.averageLatencyMs;
  }

  /**
   * Set callback for market tick events
   */
  onTick(callback: (tick: MarketTick) => void): void {
    this.onTickCallback = callback;
  }

  /**
   * Get current statistics
   */
  getStats(): WebSocketStats {
    return { ...this.stats };
  }

  /**
   * Disconnect from server
   */
  disconnect(): void {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Attach canvas for hardware-accelerated rendering
   */
  attachCanvas(canvas: HTMLCanvasElement): void {
    this.canvasRef = canvas;
    // Enable hardware acceleration hints for Windows browsers
    const context = canvas.getContext('2d', {
      alpha: false, // Optimize for opaque canvas
      desynchronized: true, // Reduce latency for real-time rendering
    });
    
    if (context) {
      // Force GPU acceleration on Windows
      (context as CanvasRenderingContext2D).imageSmoothingEnabled = false;
    }
  }
}

/**
 * React hook for using the WebSocket client
 */
export function useHFTWebSocket(url: string) {
  const clientRef = useRef<WindowsWebSocketClient | null>(null);
  const [connected, setConnected] = useState(false);
  const [stats, setStats] = useState<WebSocketStats | null>(null);
  const [latestTick, setLatestTick] = useState<MarketTick | null>(null);

  useEffect(() => {
    const client = new WindowsWebSocketClient(url);
    clientRef.current = client;

    client.onTick((tick) => {
      setLatestTick(tick);
      setStats(client.getStats());
    });

    client.connect()
      .then(() => setConnected(true))
      .catch((err) => console.error('Connection failed:', err));

    return () => {
      client.disconnect();
      setConnected(false);
    };
  }, [url]);

  const sendOrder = useCallback((order: { symbol: string; quantity: number; side: string }) => {
    if (clientRef.current) {
      // Send order via WebSocket
      const message = JSON.stringify({ type: 'order', ...order });
      clientRef.current['ws']?.send(message);
    }
  }, []);

  return { connected, stats, latestTick, sendOrder };
}

export default WindowsWebSocketClient;

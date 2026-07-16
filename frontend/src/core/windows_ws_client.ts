/**
 * =============================================================================
 * WINDOWS WEBSOCKET CLIENT - OPTIMIZED FOR EDGE/CHROME ON WINDOWS 11
 * Binary chunking to bypass browser memory limits, GPU-accelerated Canvas
 * =============================================================================
 */

import { useEffect, useRef, useCallback, useState } from 'react';

// =============================================================================
// CONFIGURATION - WINDOWS-SPECIFIC OPTIMIZATIONS
// =============================================================================
const WS_CONFIG = {
  url: import.meta.env.VITE_WS_URL || 'wss://localhost:8081',
  reconnectInterval: 1000,
  maxReconnectAttempts: 10,
  // Windows browser memory optimization
  binaryChunkSize: 65536, // 64KB chunks to avoid GC pressure
  maxBufferLength: 1000,
  // GPU acceleration hints for AMD Radeon
  enableGPUAcceleration: true,
};

// =============================================================================
// BINARY CHUNKING DECODER - BYPASSES WINDOWS BROWSER MEMORY LIMITS
// =============================================================================
class BinaryChunkDecoder {
  private buffer: Uint8Array[] = [];
  private expectedLength: number = 0;
  private receivedLength: number = 0;

  start(expectedLength: number): void {
    this.expectedLength = expectedLength;
    this.receivedLength = 0;
    this.buffer = [];
  }

  addChunk(chunk: ArrayBuffer): boolean {
    const uint8 = new Uint8Array(chunk);
    this.buffer.push(uint8);
    this.receivedLength += uint8.length;

    if (this.receivedLength >= this.expectedLength) {
      return true; // Message complete
    }
    return false;
  }

  decode(): Uint8Array {
    const result = new Uint8Array(this.receivedLength);
    let offset = 0;

    for (const chunk of this.buffer) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return result;
  }

  reset(): void {
    this.buffer = [];
    this.expectedLength = 0;
    this.receivedLength = 0;
  }
}

// =============================================================================
// WEBSOCKET HOOK - WINDOWS-OPTIMIZED CONNECTION MANAGEMENT
// =============================================================================
export function useWindowsWebSocket(
  onMessage: (data: any) => void,
  onError?: (error: Error) => void
) {
  const wsRef = useRef<WebSocket | null>(null);
  const decoderRef = useRef<BinaryChunkDecoder>(new BinaryChunkDecoder());
  const reconnectCountRef = useRef(0);
  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState<number>(0);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    console.log('[WS] Connecting to:', WS_CONFIG.url);

    try {
      // Enable binaryType for efficient ArrayBuffer handling
      wsRef.current = new WebSocket(WS_CONFIG.url);
      wsRef.current.binaryType = 'arraybuffer';

      const connectTime = performance.now();

      wsRef.current.onopen = () => {
        console.log('[WS] Connected');
        setIsConnected(true);
        reconnectCountRef.current = 0;
        setLatency(performance.now() - connectTime);
      };

      wsRef.current.onclose = (event) => {
        console.log('[WS] Closed:', event.code, event.reason);
        setIsConnected(false);

        // Auto-reconnect with exponential backoff
        if (reconnectCountRef.current < WS_CONFIG.maxReconnectAttempts) {
          const delay = WS_CONFIG.reconnectInterval * Math.pow(2, reconnectCountRef.current);
          console.log(`[WS] Reconnecting in ${delay}ms...`);
          reconnectCountRef.current++;
          setTimeout(connect, delay);
        } else {
          onError?.(new Error('Max reconnection attempts reached'));
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('[WS] Error:', error);
        onError?.(new Error('WebSocket connection error'));
      };

      wsRef.current.onmessage = (event) => {
        const receiveTime = performance.now();

        if (event.data instanceof ArrayBuffer) {
          // Handle binary chunked messages
          handleBinaryMessage(event.data, onMessage);
        } else {
          // Handle JSON text messages
          try {
            const data = JSON.parse(event.data);
            data._receiveTime = receiveTime;
            onMessage(data);
          } catch (e) {
            console.error('[WS] Failed to parse JSON:', e);
          }
        }
      };
    } catch (error) {
      console.error('[WS] Connection failed:', error);
      onError?.(error as Error);
    }
  }, [onMessage, onError]);

  const handleBinaryMessage = useCallback(
    (data: ArrayBuffer, callback: (data: any) => void) => {
      const view = new DataView(data);
      
      // Protocol: First 4 bytes = total message length
      if (view.byteLength >= 4) {
        const totalLength = view.getUint32(0, false); // Big-endian
        
        if (decoderRef.current.expectedLength === 0) {
          // Start new message
          decoderRef.current.start(totalLength);
        }
        
        // Add chunk (skip length header if it's the first chunk)
        const chunk = decoderRef.current.expectedLength === 0 
          ? data.slice(4) 
          : data;
        
        const isComplete = decoderRef.current.addChunk(chunk);
        
        if (isComplete) {
          try {
            const decoded = decoderRef.current.decode();
            const jsonStr = new TextDecoder().decode(decoded);
            const parsed = JSON.parse(jsonStr);
            parsed._receiveTime = performance.now();
            parsed._isBinary = true;
            callback(parsed);
          } catch (e) {
            console.error('[WS] Binary decode failed:', e);
          }
          decoderRef.current.reset();
        }
      }
    },
    []
  );

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('[WS] Cannot send - not connected');
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { isConnected, latency, sendMessage, disconnect, reconnect: connect };
}

// =============================================================================
// GPU-ACCELERATED CANVAS RENDERER - AMD RADEON OPTIMIZED
// =============================================================================
export class TradingCanvasRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D | null;
  private animationFrameId: number | null = null;
  private dataQueue: any[] = [];
  private maxDisplayItems = 100;

  constructor(canvasId: string) {
    this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    
    // Enable hardware acceleration hints for Windows browsers
    this.canvas.style.willChange = 'transform';
    this.canvas.style.transform = 'translateZ(0)';
    
    // Force GPU compositing in Edge/Chrome
    Object.assign(this.canvas.style, {
      contain: 'layout paint',
      contentVisibility: 'visible',
    });

    this.ctx = this.canvas.getContext('2d', {
      alpha: false, // Optimize for opaque background
      desynchronized: true, // Reduce latency for real-time rendering
    });

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  private resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    
    if (this.ctx) {
      this.ctx.scale(dpr, dpr);
    }
  }

  public pushData(data: any): void {
    this.dataQueue.push(data);
    
    // Trim queue to prevent memory buildup
    if (this.dataQueue.length > this.maxDisplayItems) {
      this.dataQueue.shift();
    }

    // Request render on next frame
    if (this.animationFrameId === null) {
      this.animationFrameId = requestAnimationFrame(() => this.render());
    }
  }

  private render(): void {
    if (!this.ctx || this.dataQueue.length === 0) {
      this.animationFrameId = null;
      return;
    }

    const width = this.canvas.width / (window.devicePixelRatio || 1);
    const height = this.canvas.height / (window.devicePixelRatio || 1);

    // Clear with optimized fill
    this.ctx.fillStyle = '#0a0a0f';
    this.ctx.fillRect(0, 0, width, height);

    // Draw order book depth visualization
    this.drawOrderBook(width, height);

    // Draw recent trades
    this.drawTrades(width, height);

    // Draw latency heatmap
    this.drawLatencyHeatmap(width, height);

    this.animationFrameId = null;
    
    // Continue animation if there's more data
    if (this.dataQueue.length > 0) {
      this.animationFrameId = requestAnimationFrame(() => this.render());
    }
  }

  private drawOrderBook(width: number, height: number): void {
    const latest = this.dataQueue[this.dataQueue.length - 1];
    if (!latest?.orderbook) return;

    const { bids, asks } = latest.orderbook;
    const barHeight = height / 4;
    const centerY = height / 2;

    // Draw bids (green, bottom)
    if (bids?.length) {
      const maxBid = Math.max(...bids.map((b: any) => b.size));
      
      bids.forEach((bid: any, i: number) => {
        const barWidth = (bid.size / maxBid) * (width * 0.4);
        const y = centerY + (i / bids.length) * barHeight;
        
        this.ctx.fillStyle = `rgba(0, 255, 100, ${1 - i / bids.length})`;
        this.ctx.fillRect(0, y, barWidth, barHeight / bids.length - 1);
      });
    }

    // Draw asks (red, top)
    if (asks?.length) {
      const maxAsk = Math.max(...asks.map((a: any) => a.size));
      
      asks.forEach((ask: any, i: number) => {
        const barWidth = (ask.size / maxAsk) * (width * 0.4);
        const y = centerY - ((i + 1) / asks.length) * barHeight;
        
        this.ctx.fillStyle = `rgba(255, 50, 50, ${1 - i / asks.length})`;
        this.ctx.fillRect(width - barWidth, y, barWidth, barHeight / asks.length - 1);
      });
    }
  }

  private drawTrades(width: number, height: number): void {
    const trades = this.dataQueue
      .filter(d => d.trade)
      .slice(-50)
      .map(d => d.trade);

    if (!trades.length) return;

    const dotRadius = 3;
    const tradeWidth = width * 0.3;
    const startX = width * 0.35;

    trades.forEach((trade: any, i: number) => {
      const x = startX + (i / trades.length) * tradeWidth;
      const y = height / 2 + (trade.side === 'buy' ? -20 : 20);
      
      this.ctx.beginPath();
      this.ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
      this.ctx.fillStyle = trade.side === 'buy' ? '#00ff66' : '#ff3333';
      this.ctx.fill();
    });
  }

  private drawLatencyHeatmap(width: number, height: number): void {
    const latencies = this.dataQueue
      .filter(d => d._receiveTime && d._sendTime)
      .map(d => d._receiveTime - d._sendTime)
      .slice(-100);

    if (!latencies.length) return;

    const maxLatency = Math.max(...latencies, 100);
    const barHeight = 20;
    const y = height - barHeight - 10;

    latencies.forEach((lat: number, i: number) => {
      const barWidth = (width * 0.9) / latencies.length;
      const hue = 120 - (lat / maxLatency) * 120; // Green to Red
      
      this.ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
      this.ctx.fillRect(i * barWidth, y, barWidth - 1, barHeight);
    });

    // Label
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '12px Consolas, monospace';
    this.ctx.fillText(`Latency: ${Math.round(latencies[latencies.length - 1])}ms`, 10, y - 5);
  }

  public destroy(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }
}

// =============================================================================
// PERFORMANCE MONITOR - WINDOWS BROWSER METRICS
// =============================================================================
export function monitorBrowserPerformance() {
  const metrics = {
    memory: (performance as any).memory ? {
      usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
      totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
    } : null,
    fps: 0,
    latency: [],
  };

  // Track FPS using requestAnimationFrame
  let lastTime = performance.now();
  let frames = 0;

  const measureFPS = () => {
    frames++;
    const now = performance.now();
    
    if (now - lastTime >= 1000) {
      metrics.fps = frames;
      frames = 0;
      lastTime = now;
    }
    
    requestAnimationFrame(measureFPS);
  };

  measureFPS();

  return metrics;
}

export default useWindowsWebSocket;

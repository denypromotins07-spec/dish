import { EventEmitter } from 'events';

interface WSConfig {
  url: string;
  maxReconnectDelay: number;
  initialReconnectDelay: number;
  backoffMultiplier: number;
  binaryType: BinaryType;
}

interface BackpressureSignal {
  type: 'backpressure';
  pauseMs: number;
}

type MessageHandler = (data: ArrayBuffer) => void;

export class WSClient extends EventEmitter {
  private ws: WebSocket | null = null;
  private config: WSConfig;
  private reconnectDelay: number;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private isPaused: boolean = false;
  private pauseTimeout: ReturnType<typeof setTimeout> | null = null;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private lastPongTime: number = 0;

  constructor(config: Partial<WSConfig> = {}) {
    super();
    this.config = {
      url: config.url || `ws://${window.location.host}/ws`,
      maxReconnectDelay: config.maxReconnectDelay || 30000,
      initialReconnectDelay: config.initialReconnectDelay || 1000,
      backoffMultiplier: config.backoffMultiplier || 2,
      binaryType: config.binaryType || 'arraybuffer',
    };
    this.reconnectDelay = this.config.initialReconnectDelay;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(this.config.url);
      this.ws.binaryType = this.config.binaryType;

      this.ws.onopen = () => {
        console.log('[WS] Connected to backend');
        this.reconnectDelay = this.config.initialReconnectDelay;
        this.emit('connected');
        this.startPingInterval();
      };

      this.ws.onclose = (event) => {
        console.log(`[WS] Disconnected: ${event.code} ${event.reason}`);
        this.stopPingInterval();
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('[WS] Error:', error);
        this.emit('error', error);
      };

      this.ws.onmessage = (event) => {
        if (this.isPaused) return;

        if (event.data instanceof ArrayBuffer) {
          const view = new DataView(event.data);
          
          // Check for backpressure signal (first byte = 0xFF)
          if (view.getUint8(0) === 0xFF) {
            const pauseMs = view.getUint32(1, true);
            this.handleBackpressure(pauseMs);
            return;
          }

          this.messageHandlers.forEach(handler => handler(event.data));
          this.emit('message', event.data);
        }
      };
    } catch (error) {
      console.error('[WS] Connection failed:', error);
      this.scheduleReconnect();
    }
  }

  private handleBackpressure(pauseMs: number): void {
    console.log(`[WS] Backpressure signal received, pausing for ${pauseMs}ms`);
    this.isPaused = true;
    
    if (this.pauseTimeout) clearTimeout(this.pauseTimeout);
    this.pauseTimeout = setTimeout(() => {
      this.isPaused = false;
      this.emit('resumed');
    }, pauseMs);
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);

    this.reconnectTimer = setTimeout(() => {
      console.log(`[WS] Reconnecting in ${this.reconnectDelay}ms...`);
      this.reconnectDelay = Math.min(
        this.reconnectDelay * this.config.backoffMultiplier,
        this.config.maxReconnectDelay
      );
      this.connect();
    }, this.reconnectDelay);
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopPingInterval();
    
    if (this.pauseTimeout) {
      clearTimeout(this.pauseTimeout);
      this.pauseTimeout = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  subscribe(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  send(data: ArrayBuffer | string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      console.warn('[WS] Cannot send, connection not open');
    }
  }

  private startPingInterval(): void {
    this.pingInterval = setInterval(() => {
      if (Date.now() - this.lastPongTime > 10000) {
        console.warn('[WS] Ping timeout, reconnecting...');
        this.ws?.close();
        return;
      }
      this.send(new Uint8Array([0x01])); // Ping opcode
    }, 5000);
    this.lastPongTime = Date.now();
  }

  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  getPing(): number {
    return Date.now() - this.lastPongTime;
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const wsClient = new WSClient();

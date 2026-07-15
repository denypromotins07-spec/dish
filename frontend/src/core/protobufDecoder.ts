// Zero-copy binary decoder for Protobuf/MessagePack streams from Rust backend
// Avoids JSON.parse() GC pauses by directly parsing into TypedArrays

export interface TickData {
  timestamp: number;
  price: number;
  volume: number;
  side: number; // 0 = bid, 1 = ask
}

export interface OrderBookLevel {
  price: number;
  size: number;
  orderCount: number;
}

export interface OrderBookSnapshot {
  bids: Float64Array;
  asks: Float64Array;
  timestamp: number;
}

export interface LatencySample {
  value: number; // microseconds
  timestamp: number;
}

const MESSAGE_TYPES = {
  TICK: 0x01,
  ORDERBOOK_SNAPSHOT: 0x02,
  ORDERBOOK_DELTA: 0x03,
  TELEMETRY: 0x04,
  LATENCY: 0x05,
} as const;

export class ProtobufDecoder {
  private tickBuffer: Float64Array;
  private orderBookBuffer: Float64Array;
  private latencyBuffer: Uint32Array;
  private maxTicks: number = 10000;
  private maxLevels: number = 50;
  private maxLatencySamples: number = 1000;

  constructor() {
    this.tickBuffer = new Float64Array(this.maxTicks * 4); // [timestamp, price, volume, side]
    this.orderBookBuffer = new Float64Array(this.maxLevels * 3); // [price, size, count]
    this.latencyBuffer = new Uint32Array(this.maxLatencySamples);
  }

  decode(buffer: ArrayBuffer): DecodedMessage | null {
    const view = new DataView(buffer);
    const messageType = view.getUint8(0);

    switch (messageType) {
      case MESSAGE_TYPES.TICK:
        return this.decodeTick(view, 1);
      case MESSAGE_TYPES.ORDERBOOK_SNAPSHOT:
        return this.decodeOrderBookSnapshot(view, 1);
      case MESSAGE_TYPES.ORDERBOOK_DELTA:
        return this.decodeOrderBookDelta(view, 1);
      case MESSAGE_TYPES.TELEMETRY:
        return this.decodeTelemetry(view, 1);
      case MESSAGE_TYPES.LATENCY:
        return this.decodeLatency(view, 1);
      default:
        console.warn(`[Decoder] Unknown message type: ${messageType}`);
        return null;
    }
  }

  private decodeTick(view: DataView, offset: number): TickMessage {
    const timestamp = Number(view.getBigInt64(offset, true));
    offset += 8;

    const price = view.getFloat64(offset, true);
    offset += 8;

    const volume = view.getFloat64(offset, true);
    offset += 8;

    const side = view.getUint8(offset);

    return {
      type: 'tick',
      data: { timestamp, price, volume, side },
    };
  }

  private decodeOrderBookSnapshot(view: DataView, offset: number): OrderBookMessage {
    const timestamp = Number(view.getBigInt64(offset, true));
    offset += 8;

    const bidCount = view.getUint16(offset, true);
    offset += 2;

    const askCount = view.getUint16(offset, true);
    offset += 2;

    const bids = new Float64Array(bidCount * 3);
    for (let i = 0; i < bidCount * 3; i++) {
      bids[i] = view.getFloat64(offset + i * 8, true);
    }
    offset += bidCount * 3 * 8;

    const asks = new Float64Array(askCount * 3);
    for (let i = 0; i < askCount * 3; i++) {
      asks[i] = view.getFloat64(offset + i * 8, true);
    }

    return {
      type: 'orderbook_snapshot',
      data: { bids, asks, timestamp },
    };
  }

  private decodeOrderBookDelta(view: DataView, offset: number): OrderBookDeltaMessage {
    const timestamp = Number(view.getBigInt64(offset, true));
    offset += 8;

    const levelCount = view.getUint8(offset);
    offset += 1;

    const deltas: Array<{ side: number; price: number; size: number }> = [];
    for (let i = 0; i < levelCount; i++) {
      const side = view.getUint8(offset);
      offset += 1;
      const price = view.getFloat64(offset, true);
      offset += 8;
      const size = view.getFloat64(offset, true);
      offset += 8;
      deltas.push({ side, price, size });
    }

    return {
      type: 'orderbook_delta',
      data: { deltas, timestamp },
    };
  }

  private decodeTelemetry(view: DataView, offset: number): TelemetryMessage {
    const cpuTemp = view.getFloat32(offset, true);
    offset += 4;

    const ramUsage = view.getFloat32(offset, true);
    offset += 4;

    const activeThreads = view.getUint16(offset, true);
    offset += 2;

    const fpsThrottler = view.getUint8(offset);
    offset += 1;

    const gpuLoad = view.getFloat32(offset, true);

    return {
      type: 'telemetry',
      data: { cpuTemp, ramUsage, activeThreads, fpsThrottler, gpuLoad },
    };
  }

  private decodeLatency(view: DataView, offset: number): LatencyMessage {
    const sampleCount = view.getUint16(offset, true);
    offset += 2;

    const samples = new Uint32Array(sampleCount);
    for (let i = 0; i < sampleCount; i++) {
      samples[i] = view.getUint32(offset + i * 4, true);
    }

    return {
      type: 'latency',
      data: { samples },
    };
  }

  // Direct buffer access for zero-copy rendering
  getTickBuffer(): Float64Array {
    return this.tickBuffer;
  }

  getOrderBookBuffer(): Float64Array {
    return this.orderBookBuffer;
  }

  getLatencyBuffer(): Uint32Array {
    return this.latencyBuffer;
  }
}

export type DecodedMessage =
  | TickMessage
  | OrderBookMessage
  | OrderBookDeltaMessage
  | TelemetryMessage
  | LatencyMessage;

interface TickMessage {
  type: 'tick';
  data: TickData;
}

interface OrderBookMessage {
  type: 'orderbook_snapshot';
  data: OrderBookSnapshot;
}

interface OrderBookDeltaMessage {
  type: 'orderbook_delta';
  data: {
    deltas: Array<{ side: number; price: number; size: number }>;
    timestamp: number;
  };
}

interface TelemetryMessage {
  type: 'telemetry';
  data: {
    cpuTemp: number;
    ramUsage: number;
    activeThreads: number;
    fpsThrottler: number;
    gpuLoad: number;
  };
}

interface LatencyMessage {
  type: 'latency';
  data: {
    samples: Uint32Array;
  };
}

export const decoder = new ProtobufDecoder();

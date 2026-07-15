import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// High-frequency tick state - bypasses React re-renders, goes to Canvas refs
interface HFState {
  // Direct TypedArray references for zero-copy rendering
  tickBuffer: Float64Array | null;
  orderBookBids: Float64Array | null;
  orderBookAsks: Float64Array | null;
  latencySamples: Uint32Array | null;
  
  // Metadata (low-frequency, safe to expose to React)
  lastTickTime: number;
  tickCount: number;
  spread: number;
  midPrice: number;
}

// Low-frequency UI state - triggers React re-renders
interface UIState {
  // System telemetry
  cpuTemp: number;
  ramUsage: number;
  activeThreads: number;
  gpuLoad: number;
  fpsThrottler: number;
  
  // Connection status
  wsConnected: boolean;
  wsPing: number;
  packetLoss: number;
  
  // UI preferences
  activeTab: string;
  theme: 'dark' | 'light';
  showTelemetry: boolean;
}

interface HFActions {
  updateTickBuffer: (buffer: Float64Array) => void;
  updateOrderBook: (bids: Float64Array, asks: Float64Array, spread: number, midPrice: number) => void;
  updateLatencySamples: (samples: Uint32Array) => void;
  incrementTickCount: () => void;
  setLastTickTime: (time: number) => void;
}

interface UIActions {
  updateTelemetry: (data: { cpuTemp: number; ramUsage: number; activeThreads: number; gpuLoad: number; fpsThrottler: number }) => void;
  updateConnectionStatus: (connected: boolean, ping: number, packetLoss: number) => void;
  setActiveTab: (tab: string) => void;
  toggleTelemetry: () => void;
}

const initialHFState: HFState = {
  tickBuffer: null,
  orderBookBids: null,
  orderBookAsks: null,
  latencySamples: null,
  lastTickTime: 0,
  tickCount: 0,
  spread: 0,
  midPrice: 0,
};

const initialUIState: UIState = {
  cpuTemp: 0,
  ramUsage: 0,
  activeThreads: 0,
  gpuLoad: 0,
  fpsThrottler: 60,
  wsConnected: false,
  wsPing: 0,
  packetLoss: 0,
  activeTab: 'dashboard',
  theme: 'dark',
  showTelemetry: true,
};

export const useHFStore = create<HFState & HFActions>()(
  subscribeWithSelector((set, get) => ({
    ...initialHFState,
    
    updateTickBuffer: (buffer) => {
      // Direct mutation - no React re-render
      const state = get();
      if (state.tickBuffer !== buffer) {
        // We store reference only, actual data stays in decoder
        set({ tickBuffer: buffer }, false, 'tickBuffer');
      }
    },
    
    updateOrderBook: (bids, asks, spread, midPrice) => {
      set({ 
        orderBookBids: bids, 
        orderBookAsks: asks, 
        spread, 
        midPrice 
      }, false, 'orderbook');
    },
    
    updateLatencySamples: (samples) => {
      set({ latencySamples: samples }, false, 'latency');
    },
    
    incrementTickCount: () => {
      // Only update count every 100 ticks to reduce re-renders
      const current = get().tickCount;
      if (current % 100 === 0) {
        set({ tickCount: current + 1 });
      }
    },
    
    setLastTickTime: (time) => {
      set({ lastTickTime: time }, false, 'lastTickTime');
    },
  }))
);

export const useUIStore = create<UIState & UIActions>()(
  subscribeWithSelector((set) => ({
    ...initialUIState,
    
    updateTelemetry: (data) => {
      set(data);
    },
    
    updateConnectionStatus: (connected, ping, packetLoss) => {
      set({ wsConnected: connected, wsPing: ping, packetLoss });
    },
    
    setActiveTab: (tab) => {
      set({ activeTab: tab });
    },
    
    toggleTelemetry: () => {
      set((state) => ({ showTelemetry: !state.showTelemetry }));
    },
  }))
);

// Selector helpers for performance-critical components
export const selectMidPrice = (state: HFState) => state.midPrice;
export const selectSpread = (state: HFState) => state.spread;
export const selectTickCount = (state: HFState) => state.tickCount;
export const selectWSPing = (state: UIState) => state.wsPing;
export const selectWSConnected = (state: UIState) => state.wsConnected;

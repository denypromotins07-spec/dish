import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface ReplayEvent {
  time: number;
  price: number;
  type: number; // 0=candle, 1=order, 2=fill
  color: [number, number, number, number];
}

interface ReplayState {
  // Core state
  isPlaying: boolean;
  currentTime: number;
  totalDuration: number;
  speed: number;
  
  // Ring buffer for replay data (fixed size to prevent GC)
  replayData: ReplayEvent[];
  maxBufferSize: number;
  
  // Actions
  setIsPlaying: (playing: boolean) => void;
  setCurrentTime: (time: number) => void;
  setTotalDuration: (duration: number) => void;
  setSpeed: (speed: number) => void;
  seekTo: (time: number) => void;
  addEvent: (event: ReplayEvent) => void;
  clearBuffer: () => void;
  reset: () => void;
}

const MAX_BUFFER_SIZE = 50000; // Fixed size ring buffer

export const useReplayStore = create<ReplayState>()(
  subscribeWithSelector((set, get) => ({
    isPlaying: false,
    currentTime: 0,
    totalDuration: 0,
    speed: 1,
    replayData: [],
    maxBufferSize: MAX_BUFFER_SIZE,

    setIsPlaying: (playing: boolean) => set({ isPlaying: playing }),
    
    setCurrentTime: (time: number) => set({ currentTime: Math.max(0, time) }),
    
    setTotalDuration: (duration: number) => set({ totalDuration: duration }),
    
    setSpeed: (speed: number) => set({ speed }),
    
    seekTo: (time: number) => {
      set({ currentTime: Math.max(0, Math.min(time, get().totalDuration)) });
    },
    
    addEvent: (event: ReplayEvent) => {
      const currentData = get().replayData;
      
      // Ring buffer logic: remove oldest if at capacity
      if (currentData.length >= MAX_BUFFER_SIZE) {
        const newData = currentData.slice(1);
        newData.push(event);
        set({ replayData: newData });
      } else {
        set({ replayData: [...currentData, event] });
      }
    },
    
    clearBuffer: () => set({ replayData: [] }),
    
    reset: () => set({
      isPlaying: false,
      currentTime: 0,
      replayData: [],
    }),
  }))
);

// Pre-allocate TypedArray for high-frequency updates
export const replayVertexPool = new Float32Array(MAX_BUFFER_SIZE * 7);
export let replayVertexCount = 0;

export const resetVertexPool = () => {
  replayVertexCount = 0;
};

export const addToVertexPool = (event: ReplayEvent) => {
  if (replayVertexCount >= MAX_BUFFER_SIZE) return;
  
  const idx = replayVertexCount * 7;
  replayVertexPool[idx] = event.time;
  replayVertexPool[idx + 1] = event.price;
  replayVertexPool[idx + 2] = event.type;
  replayVertexPool[idx + 3] = event.color[0];
  replayVertexPool[idx + 4] = event.color[1];
  replayVertexPool[idx + 5] = event.color[2];
  replayVertexPool[idx + 6] = event.color[3];
  
  replayVertexCount++;
};

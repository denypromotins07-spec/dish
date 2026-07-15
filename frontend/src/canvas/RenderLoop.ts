// Master render loop manager
// Dynamically throttles frontend FPS based on backend telemetry
// to save AMD laptop battery and GPU resources during extreme volatility

import { useUIStore } from '@/core/store';

export interface RenderLoopConfig {
  targetFPS: number;
  minFPS: number;
  maxFPS: number;
  throttleThreshold: number; // CPU temp threshold for throttling
  adaptiveThrottling: boolean;
}

const defaultConfig: RenderLoopConfig = {
  targetFPS: 60,
  minFPS: 15,
  maxFPS: 60,
  throttleThreshold: 75, // Celsius
  adaptiveThrottling: true,
};

type RenderCallback = (deltaTime: number, timestamp: number) => void;

export class RenderLoop {
  private config: RenderLoopConfig;
  private callbacks: Set<RenderCallback> = new Set();
  private animationFrameId: number | null = null;
  private lastTimestamp: number = 0;
  private frameInterval: number = 1000 / 60;
  private accumulatedTime: number = 0;
  private isRunning: boolean = false;
  private currentFPS: number = 60;
  private frameCount: number = 0;
  private fpsUpdateTime: number = 0;

  constructor(config: Partial<RenderLoopConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
    this.frameInterval = 1000 / this.config.targetFPS;
  }

  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.lastTimestamp = performance.now();
    this.fpsUpdateTime = this.lastTimestamp;
    
    const loop = (timestamp: number) => {
      if (!this.isRunning) return;

      const deltaTime = timestamp - this.lastTimestamp;
      this.accumulatedTime += deltaTime;
      this.lastTimestamp = timestamp;

      // FPS calculation and throttling
      this.frameCount++;
      if (timestamp - this.fpsUpdateTime >= 1000) {
        this.currentFPS = this.frameCount;
        this.frameCount = 0;
        this.fpsUpdateTime = timestamp;
        
        // Adaptive throttling based on backend telemetry
        if (this.config.adaptiveThrottling) {
          this.adjustFPSBasedOnTelemetry();
        }
      }

      // Fixed timestep rendering
      while (this.accumulatedTime >= this.frameInterval) {
        this.executeCallbacks(this.frameInterval);
        this.accumulatedTime -= this.frameInterval;
      }

      this.animationFrameId = requestAnimationFrame(loop);
    };

    this.animationFrameId = requestAnimationFrame(loop);
    console.log('[RenderLoop] Started at', this.config.targetFPS, 'FPS');
  }

  stop(): void {
    this.isRunning = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    console.log('[RenderLoop] Stopped');
  }

  private executeCallbacks(deltaTime: number): void {
    const timestamp = performance.now();
    this.callbacks.forEach((callback) => {
      try {
        callback(deltaTime, timestamp);
      } catch (error) {
        console.error('[RenderLoop] Callback error:', error);
      }
    });
  }

  subscribe(callback: RenderCallback): () => void {
    this.callbacks.add(callback);
    return () => this.callbacks.delete(callback);
  }

  setFPS(fps: number): void {
    const clampedFPS = Math.max(this.config.minFPS, Math.min(this.config.maxFPS, fps));
    this.config.targetFPS = clampedFPS;
    this.frameInterval = 1000 / clampedFPS;
    console.log(`[RenderLoop] FPS set to ${clampedFPS}`);
  }

  getFPS(): number {
    return this.currentFPS;
  }

  getTargetFPS(): number {
    return this.config.targetFPS;
  }

  private adjustFPSBasedOnTelemetry(): void {
    const state = useUIStore.getState();
    const { cpuTemp, fpsThrottler, gpuLoad } = state;

    let newFPS = this.config.targetFPS;

    // Throttle based on CPU temperature
    if (cpuTemp > this.config.throttleThreshold + 15) {
      newFPS = this.config.minFPS;
    } else if (cpuTemp > this.config.throttleThreshold + 5) {
      newFPS = Math.max(this.config.minFPS, this.config.targetFPS * 0.5);
    } else if (cpuTemp > this.config.throttleThreshold) {
      newFPS = Math.max(this.config.minFPS, this.config.targetFPS * 0.75);
    }

    // Respect backend fps_throttler signal
    if (fpsThrottler > 0 && fpsThrottler < newFPS) {
      newFPS = fpsThrottler;
    }

    // Throttle based on GPU load
    if (gpuLoad > 90) {
      newFPS = Math.max(this.config.minFPS, newFPS * 0.75);
    } else if (gpuLoad > 75) {
      newFPS = Math.max(this.config.minFPS, newFPS * 0.9);
    }

    // Apply throttling if needed
    if (newFPS !== this.config.targetFPS) {
      this.setFPS(newFPS);
    }
  }

  // Manual override for user-controlled FPS
  setConfig(config: Partial<RenderLoopConfig>): void {
    this.config = { ...this.config, ...config };
    
    if (config.targetFPS !== undefined) {
      this.setFPS(config.targetFPS);
    }
    
    if (config.adaptiveThrottling !== undefined) {
      console.log(`[RenderLoop] Adaptive throttling ${config.adaptiveThrottling ? 'enabled' : 'disabled'}`);
    }
  }

  getConfig(): RenderLoopConfig {
    return { ...this.config };
  }

  isAdaptiveThrottlingEnabled(): boolean {
    return this.config.adaptiveThrottling;
  }

  destroy(): void {
    this.stop();
    this.callbacks.clear();
  }
}

// Singleton instance for global render loop management
export const renderLoop = new RenderLoop();

// Auto-start render loop with adaptive throttling
if (typeof window !== 'undefined') {
  // Defer start until DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      renderLoop.start();
    });
  } else {
    renderLoop.start();
  }
}

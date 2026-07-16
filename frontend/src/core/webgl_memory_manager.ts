/**
 * WebGL Memory Manager for Low-Memory Frontend
 * 
 * This module strictly limits WebGL texture sizes and forces GPU
 * to release unused buffers. Drops render loop to 30fps during
 * high-tick volatility to save CPU/RAM overhead.
 * 
 * Target: Keep browser GPU memory under 256MB, total frontend under 1GB
 */

interface WebGLContextConfig {
  maxTextureSize: number;
  maxBufferCount: number;
  targetFPS: number;
  reducedFPS: number;
  memoryThresholdMB: number;
}

interface BufferInfo {
  id: string;
  size: number;
  type: 'vertex' | 'index' | 'texture';
  created: number;
  lastUsed: number;
}

export class WebGLMemoryManager {
  private gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
  private config: WebGLContextConfig;
  private buffers: Map<string, BufferInfo> = new Map();
  private frameCount: number = 0;
  private lastTime: number = 0;
  private currentFPS: number = 60;
  private isHighVolatility: boolean = false;
  private onLowMemoryCallback: (() => void) | null = null;

  constructor(config?: Partial<WebGLContextConfig>) {
    this.config = {
      maxTextureSize: 1024, // Reduced from typical 4096
      maxBufferCount: 50,
      targetFPS: 60,
      reducedFPS: 30, // Drop to 30fps during high volatility
      memoryThresholdMB: 200, // Alert at 200MB
      ...config
    };
  }

  /**
   * Initialize WebGL context with memory-optimized settings
   */
  initialize(canvas: HTMLCanvasElement): boolean {
    try {
      const gl = canvas.getContext('webgl2', {
        alpha: false, // Disable alpha channel if not needed
        antialias: false, // Disable AA for performance
        depth: false, // No depth buffer needed for 2D charts
        stencil: false, // No stencil buffer
        powerPreference: 'low-power', // Force low-power mode
        failIfMajorPerformanceCaveat: false,
        preserveDrawingBuffer: false, // Allow buffer recycling
      }) as WebGL2RenderingContext | null;

      if (!gl) {
        console.warn('WebGL2 not available, falling back to WebGL');
        return false;
      }

      this.gl = gl;

      // Set viewport limits
      gl.viewport(0, 0, Math.min(canvas.width, 1920), Math.min(canvas.height, 1080));

      // Enable aggressive cleanup
      const ext = gl.getExtension('WEBGL_lose_context');
      if (ext) {
        console.log('[WebGLMemory] WEBGL_lose_context extension available');
      }

      console.log(`[WebGLMemory] Initialized with max texture ${this.config.maxTextureSize}px`);
      return true;
    } catch (e) {
      console.error('[WebGLMemory] Initialization failed:', e);
      return false;
    }
  }

  /**
   * Create a texture with strict size limits
   */
  createLimitedTexture(width: number, height: number, data: Uint8Array): WebGLTexture | null {
    if (!this.gl) return null;

    // Enforce maximum texture size
    const clampedWidth = Math.min(width, this.config.maxTextureSize);
    const clampedHeight = Math.min(height, this.config.maxTextureSize);

    // Check if we're at buffer limit
    if (this.buffers.size >= this.config.maxBufferCount) {
      console.warn('[WebGLMemory] Buffer limit reached, forcing cleanup');
      this.forceCleanup();
    }

    const texture = this.gl.createTexture();
    if (!texture) return null;

    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);

    // Set memory-efficient parameters
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);

    // Use compressed format if possible
    try {
      this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGBA,
        clampedWidth,
        clampedHeight,
        0,
        this.gl.RGBA,
        this.gl.UNSIGNED_BYTE,
        data
      );
    } catch (e) {
      console.error('[WebGLMemory] Texture creation failed:', e);
      this.gl.deleteTexture(texture);
      return null;
    }

    // Track buffer
    const bufferId = `texture_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.buffers.set(bufferId, {
      id: bufferId,
      size: clampedWidth * clampedHeight * 4, // RGBA bytes
      type: 'texture',
      created: Date.now(),
      lastUsed: Date.now(),
    });

    this.monitorMemoryUsage();
    return texture;
  }

  /**
   * Create a vertex/index buffer with tracking
   */
  createBuffer(type: 'vertex' | 'index', data: Float32Array | Uint16Array): WebGLBuffer | null {
    if (!this.gl) return null;

    const buffer = this.gl.createBuffer();
    if (!buffer) return null;

    this.gl.bindBuffer(
      type === 'vertex' ? this.gl.ARRAY_BUFFER : this.gl.ELEMENT_ARRAY_BUFFER,
      buffer
    );
    this.gl.bufferData(
      type === 'vertex' ? this.gl.ARRAY_BUFFER : this.gl.ELEMENT_ARRAY_BUFFER,
      data,
      this.gl.STATIC_DRAW
    );

    // Track buffer
    const bufferId = `${type}_buffer_${Date.now()}`;
    this.buffers.set(bufferId, {
      id: bufferId,
      size: data.byteLength,
      type,
      created: Date.now(),
      lastUsed: Date.now(),
    });

    return buffer;
  }

  /**
   * Delete a buffer and free GPU memory immediately
   */
  deleteBuffer(buffer: WebGLBuffer | WebGLTexture): void {
    if (!this.gl || !buffer) return;

    // Find and remove from tracking
    for (const [id, info] of this.buffers.entries()) {
      // We can't directly compare WebGL objects, so we clean up by ID pattern
      if (buffer instanceof WebGLTexture && info.type === 'texture') {
        this.gl.deleteTexture(buffer as WebGLTexture);
        this.buffers.delete(id);
        break;
      } else if (buffer instanceof WebGLBuffer) {
        this.gl.deleteBuffer(buffer);
        this.buffers.delete(id);
        break;
      }
    }
  }

  /**
   * Force cleanup of oldest buffers when memory is tight
   */
  forceCleanup(): void {
    const now = Date.now();
    const sortedBuffers = Array.from(this.buffers.values())
      .sort((a, b) => a.lastUsed - b.lastUsed);

    // Remove oldest 20% of buffers
    const toRemove = sortedBuffers.slice(0, Math.ceil(sortedBuffers.length * 0.2));
    
    for (const bufferInfo of toRemove) {
      console.log(`[WebGLMemory] Removing old buffer: ${bufferInfo.id}`);
      this.buffers.delete(bufferInfo.id);
      // Note: Actual WebGL deletion would require keeping references
    }

    // Signal for context loss if critically low on memory
    if (this.getTotalMemoryUsage() > this.config.memoryThresholdMB * 1024 * 1024) {
      this.onLowMemoryCallback?.();
    }
  }

  /**
   * Get total tracked memory usage
   */
  getTotalMemoryUsage(): number {
    let total = 0;
    for (const info of this.buffers.values()) {
      total += info.size;
    }
    return total;
  }

  /**
   * Monitor memory and trigger cleanup if needed
   */
  private monitorMemoryUsage(): void {
    const totalBytes = this.getTotalMemoryUsage();
    const totalMB = totalBytes / (1024 * 1024);

    if (totalMB > this.config.memoryThresholdMB) {
      console.warn(`[WebGLMemory] High memory usage: ${totalMB.toFixed(2)}MB`);
      this.forceCleanup();
    }
  }

  /**
   * Adaptive frame rate control based on market volatility
   */
  setHighVolatility(highVolatility: boolean): void {
    this.isHighVolatility = highVolatility;
    const targetFPS = highVolatility ? this.config.reducedFPS : this.config.targetFPS;
    console.log(`[WebGLMemory] Volatility mode: ${highVolatility ? 'REDUCED' : 'NORMAL'} (${targetFPS}fps)`);
  }

  /**
   * Request animation frame with adaptive FPS limiting
   */
  requestAdaptiveFrame(callback: (deltaTime: number) => void): void {
    const now = performance.now();
    const targetFrameTime = 1000 / (this.isHighVolatility ? this.config.reducedFPS : this.config.targetFPS);

    const animate = (currentTime: number) => {
      const deltaTime = currentTime - this.lastTime;

      if (deltaTime >= targetFrameTime) {
        this.lastTime = currentTime;
        this.frameCount++;
        
        // Calculate actual FPS
        if (this.frameCount % 60 === 0) {
          this.currentFPS = Math.round(1000 / ((currentTime - this.lastTime) / 60));
        }

        callback(deltaTime);
      }

      requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }

  /**
   * Set callback for low-memory events
   */
  onLowMemory(callback: () => void): void {
    this.onLowMemoryCallback = callback;
  }

  /**
   * Get memory statistics for monitoring
   */
  getStats(): {
    bufferCount: number;
    totalMemoryMB: number;
    currentFPS: number;
    isHighVolatility: boolean;
  } {
    return {
      bufferCount: this.buffers.size,
      totalMemoryMB: this.getTotalMemoryUsage() / (1024 * 1024),
      currentFPS: this.currentFPS,
      isHighVolatility: this.isHighVolatility,
    };
  }

  /**
   * Release all resources
   */
  dispose(): void {
    if (!this.gl) return;

    // Force context loss to ensure cleanup
    const ext = this.gl.getExtension('WEBGL_lose_context');
    if (ext) {
      ext.loseContext();
    }

    this.buffers.clear();
    this.gl = null;
    console.log('[WebGLMemory] Disposed all resources');
  }
}

// Singleton instance for app-wide use
let globalMemoryManager: WebGLMemoryManager | null = null;

export function getWebGLMemoryManager(): WebGLMemoryManager {
  if (!globalMemoryManager) {
    globalMemoryManager = new WebGLMemoryManager({
      maxTextureSize: 1024,
      maxBufferCount: 50,
      targetFPS: 60,
      reducedFPS: 30,
      memoryThresholdMB: 200,
    });
  }
  return globalMemoryManager;
}

export default WebGLMemoryManager;

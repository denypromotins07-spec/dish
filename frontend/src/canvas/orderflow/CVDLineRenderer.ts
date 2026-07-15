/**
 * CVDLineRenderer - High-Speed Cumulative Volume Delta Line Chart
 * Uses fixed-size circular buffer (Ring Buffer) to prevent array resizing
 * Zero-GC rendering for long trading sessions
 */

interface CVDPoint {
  timestamp: number;
  cvdValue: number;
}

export class CVDRingBuffer {
  private buffer: Float64Array;
  private timestamps: Float64Array;
  private head: number = 0;
  private tail: number = 0;
  private size: number = 0;
  private capacity: number;
  
  // Running CVD calculation
  private cumulativeDelta: number = 0;

  constructor(capacity: number = 100000) {
    this.capacity = capacity;
    this.buffer = new Float64Array(capacity);
    this.timestamps = new Float64Array(capacity);
  }

  /**
   * Add new CVD point (circular, overwrites oldest)
   */
  push(timestamp: number, delta: number): void {
    this.cumulativeDelta += delta;
    
    this.buffer[this.head] = this.cumulativeDelta;
    this.timestamps[this.head] = timestamp;
    
    this.head = (this.head + 1) % this.capacity;
    
    if (this.size === this.capacity) {
      // Buffer full, move tail
      this.tail = (this.tail + 1) % this.capacity;
    } else {
      this.size++;
    }
  }

  /**
   * Get all points in chronological order (zero-copy view)
   */
  getPoints(): { timestamps: Float64Array; values: Float64Array; start: number; end: number; size: number } {
    return {
      timestamps: this.timestamps,
      values: this.buffer,
      start: this.tail,
      end: this.head,
      size: this.size
    };
  }

  /**
   * Get current CVD value
   */
  getCurrentCVD(): number {
    if (this.size === 0) return 0;
    const lastIdx = (this.head - 1 + this.capacity) % this.capacity;
    return this.buffer[lastIdx];
  }

  /**
   * Get CVD change over last N points
   */
  getCVDChange(n: number): number {
    if (this.size < n) return 0;
    
    const currentIdx = (this.head - 1 + this.capacity) % this.capacity;
    const oldIdx = (this.head - n + this.capacity) % this.capacity;
    
    return this.buffer[currentIdx] - this.buffer[oldIdx];
  }

  /**
   * Reset buffer
   */
  reset(): void {
    this.head = 0;
    this.tail = 0;
    this.size = 0;
    this.cumulativeDelta = 0;
    this.buffer.fill(0);
    this.timestamps.fill(0);
  }

  /**
   * Get buffer statistics
   */
  getStats(): { size: number; capacity: number; isFull: boolean } {
    return {
      size: this.size,
      capacity: this.capacity,
      isFull: this.size === this.capacity
    };
  }
}

export class CVDLineRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private ringBuffer: CVDRingBuffer;
  
  // Rendering config
  private lineWidth: number = 2;
  private positiveColor: string = '#00ff00';
  private negativeColor: string = '#ff0000';
  private zeroLineColor: string = 'rgba(100, 100, 100, 0.5)';
  private gridColor: string = 'rgba(50, 50, 50, 0.3)';
  
  // Viewport
  private visibleStart: number = 0;
  private visibleEnd: number = 0;
  private minValue: number = 0;
  private maxValue: number = 0;

  constructor(canvas: HTMLCanvasElement, bufferSize: number = 100000) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) throw new Error('Failed to create 2D context');
    this.ctx = ctx;
    
    this.ringBuffer = new CVDRingBuffer(bufferSize);
  }

  /**
   * Add tick data
   */
  addTick(timestamp: number, aggressorSide: number): void {
    // aggressorSide: 1 = buy, -1 = sell
    const delta = aggressorSide;
    this.ringBuffer.push(timestamp, delta);
  }

  /**
   * Render CVD line
   */
  render(): void {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    const points = this.ringBuffer.getPoints();
    if (points.size < 2) return;
    
    // Calculate min/max for scaling
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    const { timestamps, values, start, end, size } = points;
    
    // Iterate through circular buffer
    for (let i = 0; i < size; i++) {
      const idx = (start + i) % size;
      const val = values[idx];
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }
    
    // Add padding
    const range = maxVal - minVal || 1;
    minVal -= range * 0.1;
    maxVal += range * 0.1;
    
    this.minValue = minVal;
    this.maxValue = maxVal;
    
    // Draw zero line
    const zeroY = this.valueToY(0, height);
    ctx.strokeStyle = this.zeroLineColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(width, zeroY);
    ctx.stroke();
    
    // Draw grid lines
    ctx.strokeStyle = this.gridColor;
    ctx.lineWidth = 1;
    const gridLines = 5;
    for (let i = 1; i < gridLines; i++) {
      const y = (height / gridLines) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Draw CVD line
    ctx.strokeStyle = this.getCurrentCVD() >= 0 ? this.positiveColor : this.negativeColor;
    ctx.lineWidth = this.lineWidth;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    
    const visiblePoints = Math.min(size, 10000); // Limit visible points for performance
    const startIndex = Math.max(0, size - visiblePoints);
    
    for (let i = startIndex; i < size; i++) {
      const idx = (start + i) % size;
      const x = ((i - startIndex) / (visiblePoints - 1)) * width;
      const y = this.valueToY(values[idx], height);
      
      if (i === startIndex) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    // Fill area between line and zero
    ctx.fillStyle = this.getCurrentCVD() >= 0 
      ? 'rgba(0, 255, 0, 0.1)' 
      : 'rgba(255, 0, 0, 0.1)';
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    
    for (let i = startIndex; i < size; i++) {
      const idx = (start + i) % size;
      const x = ((i - startIndex) / (visiblePoints - 1)) * width;
      const y = this.valueToY(values[idx], height);
      ctx.lineTo(x, y);
    }
    
    ctx.lineTo(width, zeroY);
    ctx.closePath();
    ctx.fill();
  }

  /**
   * Convert CVD value to Y coordinate
   */
  private valueToY(value: number, height: number): number {
    const normalized = (value - this.minValue) / (this.maxValue - this.minValue);
    return height - (normalized * height);
  }

  /**
   * Get current CVD value
   */
  getCurrentCVD(): number {
    return this.ringBuffer.getCurrentCVD();
  }

  /**
   * Get CVD change
   */
  getCVDChange(n: number): number {
    return this.ringBuffer.getCVDChange(n);
  }

  /**
   * Resize canvas
   */
  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  /**
   * Reset renderer
   */
  reset(): void {
    this.ringBuffer.reset();
  }
}

/**
 * Crosshair - Hardware-Accelerated Crosshair Overlay
 * Drawn on separate top-layer transparent canvas
 * Uses requestPointerLock for precision tracking
 * Zero impact on main chart render loop
 */

interface CrosshairState {
  x: number;
  y: number;
  visible: boolean;
  price: number;
  timestamp: number;
}

export class Crosshair {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  
  // Current state
  private state: CrosshairState = {
    x: 0,
    y: 0,
    visible: false,
    price: 0,
    timestamp: 0
  };
  
  // Configuration
  private lineColor: string = 'rgba(255, 255, 255, 0.5)';
  private lineWidth: number = 1;
  private labelBackgroundColor: string = 'rgba(0, 0, 0, 0.8)';
  private labelTextColor: string = '#ffffff';
  private fontSize: number = 11;
  private fontFamily: string = "'JetBrains Mono', monospace";
  
  // Viewport bindings (set by parent)
  private minPrice: number = 0;
  private maxPrice: number = 0;
  private minTime: number = 0;
  private maxTime: number = 0;
  private width: number = 0;
  private height: number = 0;
  
  // Dirty flag for optimized rendering
  private isDirty: boolean = false;
  private lastX: number = -1;
  private lastY: number = -1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) throw new Error('Failed to create 2D context');
    this.ctx = ctx;
  }

  /**
   * Set viewport bindings for price/time calculation
   */
  setViewport(minPrice: number, maxPrice: number, minTime: number, maxTime: number): void {
    this.minPrice = minPrice;
    this.maxPrice = maxPrice;
    this.minTime = minTime;
    this.maxTime = maxTime;
  }

  /**
   * Update crosshair position
   */
  setPosition(x: number, y: number): void {
    if (x !== this.lastX || y !== this.lastY) {
      this.state.x = x;
      this.state.y = y;
      this.state.visible = true;
      
      // Calculate price and timestamp from coordinates
      const priceRange = this.maxPrice - this.minPrice;
      const timeRange = this.maxTime - this.minTime;
      
      this.state.price = this.maxPrice - (y / this.height) * priceRange;
      this.state.timestamp = this.minTime + (x / this.width) * timeRange;
      
      this.isDirty = true;
      this.lastX = x;
      this.lastY = y;
    }
  }

  /**
   * Hide crosshair
   */
  hide(): void {
    if (this.state.visible) {
      this.state.visible = false;
      this.isDirty = true;
      this.clear();
    }
  }

  /**
   * Clear canvas
   */
  private clear(): void {
    this.ctx.clearRect(0, 0, this.width, this.height);
  }

  /**
   * Render crosshair (only if dirty)
   */
  render(): void {
    if (!this.isDirty || !this.state.visible) return;
    
    const ctx = this.ctx;
    const { x, y, price, timestamp } = this.state;
    
    // Clear previous frame
    this.clear();
    
    // Draw vertical line
    ctx.strokeStyle = this.lineColor;
    ctx.lineWidth = this.lineWidth;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, this.height);
    ctx.stroke();
    
    // Draw horizontal line
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(this.width, y);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw price label
    const priceLabel = this.formatPrice(price);
    const priceTextWidth = ctx.measureText(priceLabel).width;
    const priceLabelHeight = this.fontSize + 8;
    
    // Label background
    ctx.fillStyle = this.labelBackgroundColor;
    ctx.fillRect(this.width - priceTextWidth - 10, y - priceLabelHeight / 2, priceTextWidth + 16, priceLabelHeight);
    
    // Label text
    ctx.fillStyle = this.labelTextColor;
    ctx.font = `${this.fontSize}px ${this.fontFamily}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(priceLabel, this.width - 4, y);
    
    // Draw timestamp label
    const timeLabel = this.formatTimestamp(timestamp);
    const timeTextWidth = ctx.measureText(timeLabel).width;
    
    // Label background
    ctx.fillStyle = this.labelBackgroundColor;
    ctx.fillRect(x - timeTextWidth / 2 - 8, 0, timeTextWidth + 16, priceLabelHeight);
    
    // Label text
    ctx.fillStyle = this.labelTextColor;
    ctx.textAlign = 'center';
    ctx.fillText(timeLabel, x, priceLabelHeight / 2);
    
    this.isDirty = false;
  }

  /**
   * Format price based on magnitude
   */
  private formatPrice(price: number): string {
    if (price >= 10000) return price.toFixed(0);
    if (price >= 100) return price.toFixed(2);
    if (price >= 1) return price.toFixed(4);
    return price.toFixed(6);
  }

  /**
   * Format timestamp
   */
  private formatTimestamp(timestamp: number): string {
    const date = new Date(timestamp);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');
    return `${hours}:${minutes}:${seconds}`;
  }

  /**
   * Resize canvas
   */
  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    this.width = width;
    this.height = height;
    this.isDirty = true;
  }

  /**
   * Enable pointer lock for precision tracking
   */
  enablePointerLock(): void {
    this.canvas.requestPointerLock?.();
  }

  /**
   * Disable pointer lock
   */
  disablePointerLock(): void {
    document.exitPointerLock?.();
  }

  /**
   * Get current crosshair state
   */
  getState(): CrosshairState {
    return { ...this.state };
  }

  /**
   * Mark as dirty for next render
   */
  markDirty(): void {
    this.isDirty = true;
  }
}

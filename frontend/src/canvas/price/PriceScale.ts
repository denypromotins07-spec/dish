/**
 * PriceScale - Dynamic Y-Axis Price Scale Calculator and Renderer
 * Renders on separate Canvas 2D layer, only redraws on viewport min/max change
 * Completely isolated from high-frequency candle render loop
 */

interface PriceLevel {
  price: number;
  y: number;
  label: string;
}

export class PriceScale {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  
  // Pre-allocated levels buffer
  private levels: PriceLevel[] = [];
  private maxLevels: number = 20;
  
  // Current scale state
  private minPrice: number = 0;
  private maxPrice: number = 0;
  private height: number = 0;
  private width: number = 60;
  
  // Configuration
  private fontSize: number = 11;
  private fontFamily: string = "'JetBrains Mono', monospace";
  private textColor: string = '#888888';
  private gridColor: string = 'rgba(50, 50, 50, 0.3)';
  private highlightColor: string = '#ffffff';
  
  // Dirty flag for optimized rendering
  private isDirty: boolean = true;
  private lastMinPrice: number = 0;
  private lastMaxPrice: number = 0;

  constructor(canvas: HTMLCanvasElement, width: number = 60) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) throw new Error('Failed to create 2D context');
    this.ctx = ctx;
    
    this.width = width;
    
    // Pre-allocate levels
    for (let i = 0; i < this.maxLevels; i++) {
      this.levels.push({ price: 0, y: 0, label: '' });
    }
  }

  /**
   * Update price range (sets dirty flag)
   */
  setPriceRange(minPrice: number, maxPrice: number): void {
    // Only mark dirty if values actually changed
    if (minPrice !== this.lastMinPrice || maxPrice !== this.lastMaxPrice) {
      this.minPrice = minPrice;
      this.maxPrice = maxPrice;
      this.lastMinPrice = minPrice;
      this.lastMaxPrice = maxPrice;
      this.isDirty = true;
    }
  }

  /**
   * Calculate optimal price levels
   */
  private calculateLevels(): void {
    const range = this.maxPrice - this.minPrice;
    if (range <= 0) return;
    
    // Calculate optimal step size
    const targetLevels = 10;
    const rawStep = range / targetLevels;
    
    // Round to "nice" numbers (1, 2, 5, 10, etc.)
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const normalizedStep = rawStep / magnitude;
    
    let step: number;
    if (normalizedStep < 1.5) step = 1 * magnitude;
    else if (normalizedStep < 3) step = 2 * magnitude;
    else if (normalizedStep < 7) step = 5 * magnitude;
    else step = 10 * magnitude;
    
    // Generate levels
    const startPrice = Math.ceil(this.minPrice / step) * step;
    let levelIndex = 0;
    
    for (let price = startPrice; price <= this.maxPrice && levelIndex < this.maxLevels; price += step) {
      const normalizedPrice = (price - this.minPrice) / range;
      const y = this.height - (normalizedPrice * this.height);
      
      this.levels[levelIndex].price = price;
      this.levels[levelIndex].y = y;
      this.levels[levelIndex].label = this.formatPrice(price, step);
      levelIndex++;
    }
    
    // Fill remaining with empty
    for (let i = levelIndex; i < this.maxLevels; i++) {
      this.levels[i].price = 0;
      this.levels[i].y = 0;
      this.levels[i].label = '';
    }
  }

  /**
   * Format price based on step size
   */
  private formatPrice(price: number, step: number): string {
    const decimals = step < 1 ? 4 : step < 10 ? 2 : 0;
    return price.toFixed(decimals);
  }

  /**
   * Render price scale (only if dirty)
   */
  render(): void {
    if (!this.isDirty) return;
    
    const ctx = this.ctx;
    const width = this.width;
    const height = this.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Calculate levels
    this.calculateLevels();
    
    // Draw grid lines and labels
    ctx.font = `${this.fontSize}px ${this.fontFamily}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    for (const level of this.levels) {
      if (level.label === '') continue;
      
      // Draw grid line
      ctx.strokeStyle = this.gridColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, level.y);
      ctx.lineTo(width - 5, level.y);
      ctx.stroke();
      
      // Draw price label
      ctx.fillStyle = this.textColor;
      ctx.fillText(level.label, width - 8, level.y);
    }
    
    // Draw current price highlight (if available)
    // This would be updated separately for real-time price
    
    this.isDirty = false;
  }

  /**
   * Update current price highlight (without full re-render)
   */
  drawCurrentPrice(price: number): void {
    if (price < this.minPrice || price > this.maxPrice) return;
    
    const ctx = this.ctx;
    const width = this.width;
    const range = this.maxPrice - this.minPrice;
    const normalizedPrice = (price - this.minPrice) / range;
    const y = this.height - (normalizedPrice * this.height);
    
    // Clear previous highlight area
    ctx.clearRect(0, y - 10, width, 20);
    
    // Draw highlight line
    ctx.strokeStyle = this.highlightColor;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw price label with highlight background
    const label = this.formatPrice(price, 0.01);
    const textWidth = ctx.measureText(label).width;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(width - textWidth - 12, y - 10, textWidth + 10, 20);
    
    ctx.fillStyle = this.highlightColor;
    ctx.fillText(label, width - 8, y);
  }

  /**
   * Set canvas dimensions
   */
  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    this.width = width;
    this.height = height;
    this.isDirty = true;
  }

  /**
   * Force re-render on next frame
   */
  markDirty(): void {
    this.isDirty = true;
  }

  /**
   * Get current price range
   */
  getPriceRange(): { min: number; max: number } {
    return { min: this.minPrice, max: this.maxPrice };
  }
}

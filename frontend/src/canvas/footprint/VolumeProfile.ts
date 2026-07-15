/**
 * VolumeProfile - Horizontal Histogram Renderer for POC, VAH, VAL
 * Renders to offscreen canvas buffer, composites only on viewport change
 * Pure math pixel mapping for zero-layout-shift performance
 */

interface VolumeLevel {
  price: number;
  volume: number;
  isPOC: boolean;
  isVAH: boolean;
  isVAL: boolean;
}

export class VolumeProfile {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private offscreenCanvas: HTMLCanvasElement;
  private offscreenCtx: CanvasRenderingContext2D;
  
  // Pre-allocated level buffer
  private levels: VolumeLevel[] = [];
  private maxLevels: number = 500;
  
  // Profile calculations
  private totalVolume: number = 0;
  private pocPrice: number = 0;
  private vahPrice: number = 0;
  private valPrice: number = 0;
  private valueAreaPercent: number = 0.70;
  
  // Rendering config
  private barHeight: number = 4;
  private maxBarWidth: number = 200;
  private profileWidth: number = 150;
  
  // Colors
  private pocColor: string = '#ffff00';
  private vahColor: string = '#00ff00';
  private valColor: string = '#ff0000';
  private normalColor: string = 'rgba(100, 100, 100, 0.5)';
  private bidColor: string = 'rgba(0, 255, 255, 0.4)';
  private askColor: string = 'rgba(255, 0, 255, 0.4)';

  constructor(canvas: HTMLCanvasElement, width: number, height: number) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) throw new Error('Failed to create 2D context');
    this.ctx = ctx;
    
    // Offscreen canvas for buffering
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = width;
    this.offscreenCanvas.height = height;
    const offCtx = this.offscreenCanvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!offCtx) throw new Error('Failed to create offscreen context');
    this.offscreenCtx = offCtx;
    
    // Pre-allocate level buffer
    for (let i = 0; i < this.maxLevels; i++) {
      this.levels.push({
        price: 0,
        volume: 0,
        isPOC: false,
        isVAH: false,
        isVAL: false
      });
    }
  }

  /**
   * Add volume at price level
   */
  addVolume(price: number, volume: number): void {
    // Find existing level or use next available slot
    let levelIndex = -1;
    for (let i = 0; i < this.levels.length; i++) {
      if (this.levels[i].price === price) {
        levelIndex = i;
        break;
      }
    }
    
    if (levelIndex === -1) {
      for (let i = 0; i < this.levels.length; i++) {
        if (this.levels[i].volume === 0) {
          levelIndex = i;
          this.levels[i].price = price;
          break;
        }
      }
    }
    
    if (levelIndex !== -1) {
      this.levels[levelIndex].volume += volume;
      this.totalVolume += volume;
    }
  }

  /**
   * Calculate POC, VAH, VAL from current volume distribution
   */
  calculateValueArea(): void {
    if (this.totalVolume === 0) return;
    
    // Find POC (Point of Control - highest volume)
    let maxVolume = 0;
    for (const level of this.levels) {
      if (level.volume > maxVolume) {
        maxVolume = level.volume;
        this.pocPrice = level.price;
      }
    }
    
    // Sort levels by price for VAH/VAL calculation
    const sortedLevels = [...this.levels]
      .filter(l => l.volume > 0)
      .sort((a, b) => a.price - b.price);
    
    if (sortedLevels.length === 0) return;
    
    // Calculate value area (70% of total volume around POC)
    const targetVolume = this.totalVolume * this.valueAreaPercent;
    let accumulatedVolume = 0;
    let pocIndex = -1;
    
    // Find POC index in sorted array
    for (let i = 0; i < sortedLevels.length; i++) {
      if (sortedLevels[i].price === this.pocPrice) {
        pocIndex = i;
        break;
      }
    }
    
    if (pocIndex === -1) return;
    
    // Expand from POC to find VAH and VAL
    let leftIndex = pocIndex;
    let rightIndex = pocIndex;
    accumulatedVolume = sortedLevels[pocIndex].volume;
    
    while (accumulatedVolume < targetVolume && 
           (leftIndex > 0 || rightIndex < sortedLevels.length - 1)) {
      const leftVol = leftIndex > 0 ? sortedLevels[leftIndex - 1].volume : 0;
      const rightVol = rightIndex < sortedLevels.length - 1 ? sortedLevels[rightIndex + 1].volume : 0;
      
      if (leftVol >= rightVol && leftIndex > 0) {
        leftIndex--;
        accumulatedVolume += leftVol;
      } else if (rightIndex < sortedLevels.length - 1) {
        rightIndex++;
        accumulatedVolume += rightVol;
      } else {
        break;
      }
    }
    
    this.vahPrice = sortedLevels[rightIndex].price;
    this.valPrice = sortedLevels[leftIndex].price;
    
    // Mark levels
    for (const level of this.levels) {
      level.isPOC = level.price === this.pocPrice;
      level.isVAH = level.price === this.vahPrice;
      level.isVAL = level.price === this.valPrice;
    }
  }

  /**
   * Render volume profile to offscreen buffer
   */
  render(priceMin: number, priceMax: number, canvasHeight: number): void {
    const ctx = this.offscreenCtx;
    const width = this.profileWidth;
    
    // Clear buffer
    ctx.clearRect(0, 0, width, canvasHeight);
    
    if (this.totalVolume === 0 || priceMax <= priceMin) return;
    
    const priceRange = priceMax - priceMin;
    const maxVolume = Math.max(...this.levels.filter(l => l.volume > 0).map(l => l.volume), 1);
    
    // Draw each level
    for (const level of this.levels) {
      if (level.volume === 0) continue;
      
      // Calculate Y position
      const normalizedPrice = (level.price - priceMin) / priceRange;
      const y = canvasHeight - (normalizedPrice * canvasHeight) - this.barHeight / 2;
      
      // Calculate bar width
      const barWidth = (level.volume / maxVolume) * this.maxBarWidth;
      
      // Determine color
      let color = this.normalColor;
      if (level.isPOC) color = this.pocColor;
      else if (level.isVAH) color = this.vahColor;
      else if (level.isVAL) color = this.valColor;
      
      // Draw bar
      ctx.fillStyle = color;
      ctx.fillRect(0, y, barWidth, this.barHeight);
      
      // Draw POC/VAH/VAL labels
      if (level.isPOC || level.isVAH || level.isVAL) {
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'left';
        const label = level.isPOC ? 'POC' : level.isVAH ? 'VAH' : 'VAL';
        ctx.fillText(`${label} ${level.price.toFixed(2)}`, barWidth + 5, y + this.barHeight - 1);
      }
    }
    
    // Draw value area background
    const vahY = canvasHeight - ((this.vahPrice - priceMin) / priceRange) * canvasHeight;
    const valY = canvasHeight - ((this.valPrice - priceMin) / priceRange) * canvasHeight;
    
    ctx.fillStyle = 'rgba(255, 255, 0, 0.05)';
    ctx.fillRect(0, Math.min(vahY, valY), this.profileWidth, Math.abs(vahY - valY));
  }

  /**
   * Composite offscreen buffer to main canvas
   */
  composite(offsetX: number = 0): void {
    this.ctx.drawImage(this.offscreenCanvas, offsetX, 0);
  }

  /**
   * Get calculated values
   */
  getProfileData(): { poc: number; vah: number; val: number; totalVolume: number } {
    return {
      poc: this.pocPrice,
      vah: this.vahPrice,
      val: this.valPrice,
      totalVolume: this.totalVolume
    };
  }

  /**
   * Reset profile
   */
  reset(): void {
    for (const level of this.levels) {
      level.price = 0;
      level.volume = 0;
      level.isPOC = false;
      level.isVAH = false;
      level.isVAL = false;
    }
    this.totalVolume = 0;
    this.pocPrice = 0;
    this.vahPrice = 0;
    this.valPrice = 0;
  }

  /**
   * Resize renderer
   */
  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    this.offscreenCanvas.width = this.profileWidth;
    this.offscreenCanvas.height = height;
  }
}

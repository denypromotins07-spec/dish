/**
 * DrawingTools - Lightweight Vector-Based Drawing Tools
 * Trendlines, Fibonacci retracements, Order Blocks, FVGs
 * Stored in minimal Zustand slice, rendered via fast Canvas 2D paths
 * Zero impact on core WebGL rendering pipeline
 */

type ToolType = 'trendline' | 'fibonacci' | 'orderBlock' | 'fvg' | 'rectangle' | 'text';

interface Point {
  x: number;
  y: number;
  price?: number;
  timestamp?: number;
}

interface Drawing {
  id: string;
  type: ToolType;
  points: Point[];
  color: string;
  lineWidth: number;
  label?: string;
  visible: boolean;
  createdAt: number;
}

interface FibonacciLevels {
  level: number;
  y: number;
  label: string;
}

export class DrawingTools {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  
  // Drawings storage (pre-allocated)
  private drawings: Map<string, Drawing> = new Map();
  private maxDrawings: number = 100;
  
  // Current tool state
  private currentTool: ToolType | null = null;
  private isDrawing: boolean = false;
  private startPoint: Point | null = null;
  private currentPoint: Point | null = null;
  
  // Viewport bindings
  private minPrice: number = 0;
  private maxPrice: number = 0;
  private minTime: number = 0;
  private maxTime: number = 0;
  private width: number = 0;
  private height: number = 0;
  
  // Default styles
  private defaultColors: Record<ToolType, string> = {
    trendline: '#00ffff',
    fibonacci: '#ff00ff',
    orderBlock: 'rgba(255, 255, 0, 0.2)',
    fvg: 'rgba(0, 255, 0, 0.2)',
    rectangle: 'rgba(100, 100, 100, 0.3)',
    text: '#ffffff'
  };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) throw new Error('Failed to create 2D context');
    this.ctx = ctx;
  }

  /**
   * Set viewport bindings for coordinate conversion
   */
  setViewport(minPrice: number, maxPrice: number, minTime: number, maxTime: number): void {
    this.minPrice = minPrice;
    this.maxPrice = maxPrice;
    this.minTime = minTime;
    this.maxTime = maxTime;
  }

  /**
   * Set current drawing tool
   */
  setTool(tool: ToolType | null): void {
    this.currentTool = tool;
    this.isDrawing = false;
    this.startPoint = null;
    this.currentPoint = null;
  }

  /**
   * Start drawing
   */
  startDrawing(x: number, y: number): void {
    if (!this.currentTool) return;
    
    this.isDrawing = true;
    this.startPoint = { x, y };
    this.currentPoint = { x, y };
  }

  /**
   * Update drawing position
   */
  updateDrawing(x: number, y: number): void {
    if (!this.isDrawing || !this.startPoint) return;
    this.currentPoint = { x, y };
  }

  /**
   * Finish drawing
   */
  finishDrawing(): Drawing | null {
    if (!this.isDrawing || !this.startPoint || !this.currentPoint || !this.currentTool) return null;
    
    const drawing: Drawing = {
      id: `drawing_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: this.currentTool,
      points: [this.startPoint, this.currentPoint],
      color: this.defaultColors[this.currentTool],
      lineWidth: this.currentTool === 'trendline' || this.currentTool === 'fibonacci' ? 2 : 1,
      visible: true,
      createdAt: Date.now()
    };
    
    // Add label for Fibonacci
    if (this.currentTool === 'fibonacci') {
      drawing.label = 'Fib Retracement';
    }
    
    this.addDrawing(drawing);
    
    this.isDrawing = false;
    this.startPoint = null;
    this.currentPoint = null;
    
    return drawing;
  }

  /**
   * Add drawing to storage
   */
  addDrawing(drawing: Drawing): void {
    // Enforce max drawings limit
    if (this.drawings.size >= this.maxDrawings) {
      const oldestId = Array.from(this.drawings.keys())[0];
      this.drawings.delete(oldestId);
    }
    
    this.drawings.set(drawing.id, drawing);
  }

  /**
   * Remove drawing
   */
  removeDrawing(id: string): void {
    this.drawings.delete(id);
  }

  /**
   * Clear all drawings
   */
  clearAll(): void {
    this.drawings.clear();
  }

  /**
   * Render all drawings
   */
  render(): void {
    const ctx = this.ctx;
    
    // Clear canvas
    ctx.clearRect(0, 0, this.width, this.height);
    
    // Render saved drawings
    for (const drawing of this.drawings.values()) {
      if (!drawing.visible) continue;
      this.renderDrawing(drawing);
    }
    
    // Render current drawing in progress
    if (this.isDrawing && this.startPoint && this.currentPoint && this.currentTool) {
      const tempDrawing: Drawing = {
        id: 'temp',
        type: this.currentTool,
        points: [this.startPoint, this.currentPoint],
        color: this.defaultColors[this.currentTool],
        lineWidth: 2,
        visible: true,
        createdAt: Date.now(),
        label: this.currentTool === 'fibonacci' ? 'Fib...' : undefined
      };
      this.renderDrawing(tempDrawing, true);
    }
  }

  /**
   * Render single drawing
   */
  private renderDrawing(drawing: Drawing, isPreview: boolean = false): void {
    const ctx = this.ctx;
    const { type, points, color, lineWidth, label } = drawing;
    
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    if (points.length < 2) return;
    
    const [p1, p2] = points;
    
    switch (type) {
      case 'trendline':
        this.drawTrendline(p1, p2);
        break;
      
      case 'fibonacci':
        this.drawFibonacci(p1, p2);
        break;
      
      case 'orderBlock':
      case 'rectangle':
        this.drawRectangle(p1, p2, type === 'orderBlock');
        break;
      
      case 'fvg':
        this.drawFVG(p1, p2);
        break;
    }
    
    // Draw label
    if (label) {
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, p1.x + 10, p1.y - 10);
    }
  }

  /**
   * Draw trendline
   */
  private drawTrendline(p1: Point, p2: Point, extend: boolean = true): void {
    const ctx = this.ctx;
    
    ctx.beginPath();
    
    if (extend) {
      // Extend line beyond endpoints
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const length = Math.sqrt(dx * dx + dy * dy);
      const extendFactor = 10;
      
      const startX = p1.x - (dx / length) * extendFactor * 50;
      const startY = p1.y - (dy / length) * extendFactor * 50;
      const endX = p2.x + (dx / length) * extendFactor * 50;
      const endY = p2.y + (dy / length) * extendFactor * 50;
      
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
    } else {
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
    }
    
    ctx.stroke();
    
    // Draw endpoint circles
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.arc(p1.x, p1.y, 4, 0, Math.PI * 2);
    ctx.arc(p2.x, p2.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  /**
   * Draw Fibonacci retracement levels
   */
  private drawFibonacci(p1: Point, p2: Point): void {
    const ctx = this.ctx;
    const levels: FibonacciLevels[] = [
      { level: 0, y: p1.y, label: '0%' },
      { level: 0.236, y: p1.y + (p2.y - p1.y) * 0.236, label: '0.236' },
      { level: 0.382, y: p1.y + (p2.y - p1.y) * 0.382, label: '0.382' },
      { level: 0.5, y: p1.y + (p2.y - p1.y) * 0.5, label: '0.5' },
      { level: 0.618, y: p1.y + (p2.y - p1.y) * 0.618, label: '0.618' },
      { level: 0.786, y: p1.y + (p2.y - p1.y) * 0.786, label: '0.786' },
      { level: 1, y: p2.y, label: '100%' }
    ];
    
    // Draw main trendline
    this.drawTrendline(p1, p2, false);
    
    // Draw horizontal levels
    ctx.setLineDash([5, 5]);
    for (const fib of levels) {
      ctx.beginPath();
      ctx.moveTo(p1.x, fib.y);
      ctx.lineTo(p2.x + 100, fib.y);
      ctx.stroke();
      
      // Draw label
      ctx.fillStyle = '#ffffff';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText(fib.label, p2.x + 5, fib.y + 3);
    }
    ctx.setLineDash([]);
  }

  /**
   * Draw rectangle (order block)
   */
  private drawRectangle(p1: Point, p2: Point, filled: boolean = false): void {
    const ctx = this.ctx;
    const x = Math.min(p1.x, p2.x);
    const y = Math.min(p1.y, p2.y);
    const width = Math.abs(p2.x - p1.x);
    const height = Math.abs(p2.y - p1.y);
    
    if (filled) {
      ctx.fillRect(x, y, width, height);
    } else {
      ctx.strokeRect(x, y, width, height);
    }
  }

  /**
   * Draw Fair Value Gap (FVG)
   */
  private drawFVG(p1: Point, p2: Point): void {
    // FVG is typically a 3-candle pattern, simplified here as a rectangle
    this.drawRectangle(p1, p2, true);
    
    // Draw border
    const ctx = this.ctx;
    const x = Math.min(p1.x, p2.x);
    const y = Math.min(p1.y, p2.y);
    const width = Math.abs(p2.x - p1.x);
    const height = Math.abs(p2.y - p1.y);
    
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
  }

  /**
   * Convert screen coordinates to price/timestamp
   */
  screenToData(x: number, y: number): { price: number; timestamp: number } {
    const priceRange = this.maxPrice - this.minPrice;
    const timeRange = this.maxTime - this.minTime;
    
    const price = this.maxPrice - (y / this.height) * priceRange;
    const timestamp = this.minTime + (x / this.width) * timeRange;
    
    return { price, timestamp };
  }

  /**
   * Convert price/timestamp to screen coordinates
   */
  dataToScreen(price: number, timestamp: number): { x: number; y: number } {
    const priceRange = this.maxPrice - this.minPrice;
    const timeRange = this.maxTime - this.minTime;
    
    const x = ((timestamp - this.minTime) / timeRange) * this.width;
    const y = this.height - ((price - this.minPrice) / priceRange) * this.height;
    
    return { x, y };
  }

  /**
   * Resize canvas
   */
  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    this.width = width;
    this.height = height;
  }

  /**
   * Get all drawings
   */
  getAllDrawings(): Drawing[] {
    return Array.from(this.drawings.values());
  }

  /**
   * Get drawing by ID
   */
  getDrawing(id: string): Drawing | undefined {
    return this.drawings.get(id);
  }

  /**
   * Toggle drawing visibility
   */
  toggleVisibility(id: string): void {
    const drawing = this.drawings.get(id);
    if (drawing) {
      drawing.visible = !drawing.visible;
    }
  }
}

/**
 * ViewportManager - Pure-Math Viewport Manager for Chart Interactions
 * Handles zoom (mouse wheel), pan (click-drag), and auto-scroll logic
 * Calculates visible time/price ranges and updates WebGL uniforms directly
 * Zero DOM layout recalculations
 */

interface ViewportState {
  minTime: number;
  maxTime: number;
  minPrice: number;
  maxPrice: number;
  isPanning: boolean;
  isZooming: boolean;
  lastMouseX: number;
  lastMouseY: number;
  autoScroll: boolean;
}

interface ViewportCallbacks {
  onViewportChange?: (viewport: ViewportState) => void;
  onZoom?: (factor: number) => void;
  onPan?: (deltaTime: number, deltaPrice: number) => void;
}

export class ViewportManager {
  private state: ViewportState = {
    minTime: 0,
    maxTime: 0,
    minPrice: 0,
    maxPrice: 0,
    isPanning: false,
    isZooming: false,
    lastMouseX: 0,
    lastMouseY: 0,
    autoScroll: false
  };
  
  // Constraints
  private minTimeRange: number = 60000; // 1 minute minimum
  private maxTimeRange: number = 86400000; // 24 hours maximum
  private minPriceRange: number = 0.01;
  private zoomSensitivity: number = 0.001;
  private panSensitivity: number = 1.0;
  
  // Callbacks
  private callbacks: ViewportCallbacks = {};
  
  // Animation frame for auto-scroll
  private animationFrame: number = 0;
  private autoScrollSpeed: number = 100; // ms per frame

  constructor(initialState?: Partial<ViewportState>) {
    if (initialState) {
      this.state = { ...this.state, ...initialState };
    }
  }

  /**
   * Set initial viewport bounds
   */
  setBounds(minTime: number, maxTime: number, minPrice: number, maxPrice: number): void {
    this.state.minTime = minTime;
    this.state.maxTime = maxTime;
    this.state.minPrice = minPrice;
    this.state.maxPrice = maxPrice;
    this.notifyViewportChange();
  }

  /**
   * Handle mouse wheel for zoom/pan
   */
  handleWheel(deltaX: number, deltaY: number, ctrlKey: boolean, clientX: number, clientY: number): void {
    if (ctrlKey || deltaY !== 0) {
      // Zoom
      const zoomFactor = Math.exp(-deltaY * this.zoomSensitivity);
      this.zoom(zoomFactor, clientX, clientY, ctrlKey ? 'price' : 'time');
    } else {
      // Pan
      this.pan(-deltaX * this.panSensitivity, -deltaY * this.panSensitivity);
    }
  }

  /**
   * Zoom viewport around cursor position
   */
  zoom(factor: number, cursorX: number, cursorY: number, axis: 'time' | 'price' | 'both' = 'both'): void {
    const { minTime, maxTime, minPrice, maxPrice } = this.state;
    
    if (axis === 'time' || axis === 'both') {
      const timeRange = maxTime - minTime;
      const newTimeRange = Math.max(this.minTimeRange, Math.min(this.maxTimeRange, timeRange * factor));
      
      // Calculate cursor position as ratio
      const cursorRatio = cursorX / (typeof window !== 'undefined' ? window.innerWidth : 1);
      const cursorTime = minTime + cursorRatio * timeRange;
      
      // Zoom around cursor
      const leftExpand = (cursorTime - minTime) * (factor - 1);
      const rightExpand = (maxTime - cursorTime) * (factor - 1);
      
      this.state.minTime = Math.round(minTime - leftExpand);
      this.state.maxTime = Math.round(maxTime + rightExpand);
    }
    
    if (axis === 'price' || axis === 'both') {
      const priceRange = maxPrice - minPrice;
      const newPriceRange = Math.max(this.minPriceRange, priceRange * factor);
      
      // Calculate cursor position as ratio
      const cursorRatio = 1 - (cursorY / (typeof window !== 'undefined' ? window.innerHeight : 1));
      const cursorPrice = minPrice + cursorRatio * priceRange;
      
      // Zoom around cursor
      const bottomExpand = (cursorPrice - minPrice) * (factor - 1);
      const topExpand = (maxPrice - cursorPrice) * (factor - 1);
      
      this.state.minPrice = minPrice - bottomExpand;
      this.state.maxPrice = maxPrice + topExpand;
    }
    
    this.callbacks.onZoom?.(factor);
    this.notifyViewportChange();
  }

  /**
   * Pan viewport
   */
  pan(deltaTime: number, deltaPrice: number): void {
    this.state.minTime += deltaTime;
    this.state.maxTime += deltaTime;
    this.state.minPrice += deltaPrice;
    this.state.maxPrice += deltaPrice;
    
    this.callbacks.onPan?.(deltaTime, deltaPrice);
    this.notifyViewportChange();
  }

  /**
   * Start panning
   */
  startPan(clientX: number, clientY: number): void {
    this.state.isPanning = true;
    this.state.lastMouseX = clientX;
    this.state.lastMouseY = clientY;
  }

  /**
   * Continue panning
   */
  continuePan(clientX: number, clientY: number): void {
    if (!this.state.isPanning) return;
    
    const deltaX = clientX - this.state.lastMouseX;
    const deltaY = clientY - this.state.lastMouseY;
    
    // Convert pixel delta to time/price delta
    const timeRange = this.state.maxTime - this.state.minTime;
    const priceRange = this.state.maxPrice - this.state.minPrice;
    
    const canvasWidth = typeof window !== 'undefined' ? window.innerWidth : 1920;
    const canvasHeight = typeof window !== 'undefined' ? window.innerHeight : 1080;
    
    const deltaTime = -(deltaX / canvasWidth) * timeRange;
    const deltaPrice = (deltaY / canvasHeight) * priceRange;
    
    this.pan(deltaTime, deltaPrice);
    
    this.state.lastMouseX = clientX;
    this.state.lastMouseY = clientY;
  }

  /**
   * End panning
   */
  endPan(): void {
    this.state.isPanning = false;
  }

  /**
   * Toggle auto-scroll
   */
  toggleAutoScroll(): void {
    this.state.autoScroll = !this.state.autoScroll;
    
    if (this.state.autoScroll) {
      this.startAutoScroll();
    } else {
      this.stopAutoScroll();
    }
  }

  /**
   * Start auto-scroll animation
   */
  private startAutoScroll(): void {
    const scroll = () => {
      if (!this.state.autoScroll) return;
      
      const timeRange = this.state.maxTime - this.state.minTime;
      const scrollAmount = (this.autoScrollSpeed / timeRange) * timeRange * 0.01;
      
      this.pan(scrollAmount, 0);
      
      this.animationFrame = requestAnimationFrame(scroll);
    };
    
    this.animationFrame = requestAnimationFrame(scroll);
  }

  /**
   * Stop auto-scroll
   */
  stopAutoScroll(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = 0;
    }
  }

  /**
   * Reset to default view
   */
  reset(minTime?: number, maxTime?: number, minPrice?: number, maxPrice?: number): void {
    this.stopAutoScroll();
    
    if (minTime !== undefined) this.state.minTime = minTime;
    if (maxTime !== undefined) this.state.maxTime = maxTime;
    if (minPrice !== undefined) this.state.minPrice = minPrice;
    if (maxPrice !== undefined) this.state.maxPrice = maxPrice;
    
    this.state.autoScroll = false;
    this.notifyViewportChange();
  }

  /**
   * Register callbacks
   */
  setCallbacks(callbacks: ViewportCallbacks): void {
    this.callbacks = callbacks;
  }

  /**
   * Notify viewport change
   */
  private notifyViewportChange(): void {
    this.callbacks.onViewportChange?.({ ...this.state });
  }

  /**
   * Get current viewport state
   */
  getState(): ViewportState {
    return { ...this.state };
  }

  /**
   * Get visible time range
   */
  getTimeRange(): { min: number; max: number; range: number } {
    return {
      min: this.state.minTime,
      max: this.state.maxTime,
      range: this.state.maxTime - this.state.minTime
    };
  }

  /**
   * Get visible price range
   */
  getPriceRange(): { min: number; max: number; range: number } {
    return {
      min: this.state.minPrice,
      max: this.state.maxPrice,
      range: this.state.maxPrice - this.state.minPrice
    };
  }

  /**
   * Check if point is in viewport
   */
  isInViewport(timestamp: number, price: number): boolean {
    return (
      timestamp >= this.state.minTime &&
      timestamp <= this.state.maxTime &&
      price >= this.state.minPrice &&
      price <= this.state.maxPrice
    );
  }

  /**
   * Fit viewport to data bounds
   */
  fitToData(
    timestamps: number[],
    prices: number[],
    paddingRatio: number = 0.05
  ): void {
    if (timestamps.length === 0 || prices.length === 0) return;
    
    let minT = Infinity, maxT = -Infinity;
    let minP = Infinity, maxP = -Infinity;
    
    for (const t of timestamps) {
      if (t < minT) minT = t;
      if (t > maxT) maxT = t;
    }
    
    for (const p of prices) {
      if (p < minP) minP = p;
      if (p > maxP) maxP = p;
    }
    
    // Add padding
    const timePadding = (maxT - minT) * paddingRatio;
    const pricePadding = (maxP - minP) * paddingRatio;
    
    this.setBounds(
      minT - timePadding,
      maxT + timePadding,
      minP - pricePadding,
      maxP + pricePadding
    );
  }

  /**
   * Cleanup
   */
  destroy(): void {
    this.stopAutoScroll();
    this.callbacks = {};
  }
}

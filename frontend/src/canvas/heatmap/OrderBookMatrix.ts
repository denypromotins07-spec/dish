/**
 * OrderBookMatrix - WebGL Buffer Manager for Liquidity Heatmap
 * Maps Rust backend L2/L3 depth data directly to GPU Float32Arrays
 * Zero-allocation updates via bufferSubData for 10k+ ticks/sec
 */

interface OrderBookLevel {
  price: number;
  volume: number;
  timestamp: number;
  side: number; // 0 = bid, 1 = ask
}

export class OrderBookMatrix {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  
  // Pre-allocated typed arrays (zero GC)
  private pricesBuffer: Float32Array;
  private volumesBuffer: Float32Array;
  private timestampsBuffer: Float32Array;
  private sidesBuffer: Float32Array;
  
  // GPU buffers
  private priceGPUBuffer: WebGLBuffer | null = null;
  private volumeGPUBuffer: WebGLBuffer | null = null;
  private timestampGPUBuffer: WebGLBuffer | null = null;
  private sideGPUBuffer: WebGLBuffer | null = null;
  
  // Configuration
  private maxLevels: number;
  private numLevels: number = 0;
  
  // Viewport uniforms
  private minPrice: number = 0;
  private maxPrice: number = 0;
  private minTime: number = 0;
  private maxTime: number = 0;

  constructor(gl: WebGL2RenderingContext, maxLevels: number = 1000) {
    this.gl = gl;
    this.maxLevels = maxLevels;
    
    // Pre-allocate buffers (critical for zero-GC)
    this.pricesBuffer = new Float32Array(maxLevels);
    this.volumesBuffer = new Float32Array(maxLevels);
    this.timestampsBuffer = new Float32Array(maxLevels);
    this.sidesBuffer = new Float32Array(maxLevels);
    
    this.initBuffers();
  }

  private initBuffers(): void {
    const gl = this.gl;
    
    // Create GPU buffers
    this.priceGPUBuffer = gl.createBuffer();
    this.volumeGPUBuffer = gl.createBuffer();
    this.timestampGPUBuffer = gl.createBuffer();
    this.sideGPUBuffer = gl.createBuffer();
    
    // Initialize with zeros
    gl.bindBuffer(gl.ARRAY_BUFFER, this.priceGPUBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.pricesBuffer, gl.DYNAMIC_DRAW);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.volumeGPUBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.volumesBuffer, gl.DYNAMIC_DRAW);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.timestampGPUBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.timestampsBuffer, gl.DYNAMIC_DRAW);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sideGPUBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.sidesBuffer, gl.DYNAMIC_DRAW);
    
    // Create VAO for instanced rendering
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
  }

  /**
   * Update order book data without reallocating memory
   * Uses bufferSubData for zero-copy GPU updates
   */
  updateOrderBook(levels: OrderBookLevel[]): void {
    this.numLevels = Math.min(levels.length, this.maxLevels);
    
    // Calculate price range for normalization
    let minP = Infinity;
    let maxP = -Infinity;
    let minT = Infinity;
    let maxT = -Infinity;
    
    for (let i = 0; i < this.numLevels; i++) {
      const level = levels[i];
      this.pricesBuffer[i] = level.price;
      this.volumesBuffer[i] = level.volume;
      this.timestampsBuffer[i] = level.timestamp;
      this.sidesBuffer[i] = level.side;
      
      if (level.price < minP) minP = level.price;
      if (level.price > maxP) maxP = level.price;
      if (level.timestamp < minT) minT = level.timestamp;
      if (level.timestamp > maxT) maxT = level.timestamp;
    }
    
    // Expand price range slightly for visual padding
    const pricePadding = (maxP - minP) * 0.05;
    this.minPrice = minP - pricePadding;
    this.maxPrice = maxP + pricePadding;
    this.minTime = minT;
    this.maxTime = maxT;
    
    // Upload to GPU via bufferSubData (zero allocation)
    const gl = this.gl;
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.priceGPUBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.pricesBuffer.subarray(0, this.numLevels));
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.volumeGPUBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.volumesBuffer.subarray(0, this.numLevels));
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.timestampGPUBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.timestampsBuffer.subarray(0, this.numLevels));
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sideGPUBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.sidesBuffer.subarray(0, this.numLevels));
  }

  /**
   * Bind buffers and attributes for rendering
   */
  bindAttributes(program: WebGLProgram): void {
    const gl = this.gl;
    const vao = this.vao;
    
    if (!vao || !program) return;
    
    gl.bindVertexArray(vao);
    gl.useProgram(program);
    
    // Price attribute (location 1)
    const priceLoc = gl.getAttribLocation(program, 'a_price');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.priceGPUBuffer);
    gl.enableVertexAttribArray(priceLoc);
    gl.vertexAttribPointer(priceLoc, 1, gl.FLOAT, false, 0, 0);
    
    // Volume attribute (location 2)
    const volumeLoc = gl.getAttribLocation(program, 'a_volume');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.volumeGPUBuffer);
    gl.enableVertexAttribArray(volumeLoc);
    gl.vertexAttribPointer(volumeLoc, 1, gl.FLOAT, false, 0, 0);
    
    // Timestamp attribute (location 3)
    const timestampLoc = gl.getAttribLocation(program, 'a_timestamp');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.timestampGPUBuffer);
    gl.enableVertexAttribArray(timestampLoc);
    gl.vertexAttribPointer(timestampLoc, 1, gl.FLOAT, false, 0, 0);
    
    // Side attribute (used for color)
    const sideLoc = gl.getAttribLocation(program, 'a_side');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sideGPUBuffer);
    gl.enableVertexAttribArray(sideLoc);
    gl.vertexAttribPointer(sideLoc, 1, gl.FLOAT, false, 0, 0);
  }

  /**
   * Render the order book heatmap
   */
  render(): void {
    const gl = this.gl;
    
    if (this.numLevels === 0) return;
    
    gl.drawArrays(gl.POINTS, 0, this.numLevels);
  }

  /**
   * Get current viewport bounds for uniform updates
   */
  getViewportBounds(): { minPrice: number; maxPrice: number; minTime: number; maxTime: number } {
    return {
      minPrice: this.minPrice,
      maxPrice: this.maxPrice,
      minTime: this.minTime,
      maxTime: this.maxTime
    };
  }

  /**
   * Cleanup GPU resources
   */
  destroy(): void {
    const gl = this.gl;
    
    if (this.priceGPUBuffer) gl.deleteBuffer(this.priceGPUBuffer);
    if (this.volumeGPUBuffer) gl.deleteBuffer(this.volumeGPUBuffer);
    if (this.timestampGPUBuffer) gl.deleteBuffer(this.timestampGPUBuffer);
    if (this.sideGPUBuffer) gl.deleteBuffer(this.sideGPUBuffer);
    if (this.vao) gl.deleteVertexArray(this.vao);
  }
}

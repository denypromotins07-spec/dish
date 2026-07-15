// WebGL2 context initialization and shader compilation setup
// Prepares the GPU to render millions of order book liquidity nodes
// and historical footprint clusters as a high-performance, glowing heatmap

export interface WebGLConfig {
  antialias: boolean;
  alpha: boolean;
  depth: boolean;
  stencil: boolean;
  premultipliedAlpha: boolean;
  preserveDrawingBuffer: boolean;
  powerPreference: 'default' | 'high-performance' | 'low-power';
}

const defaultConfig: WebGLConfig = {
  antialias: false, // Performance optimization for HFT
  alpha: true,
  depth: false, // 2D rendering doesn't need depth buffer
  stencil: false,
  premultipliedAlpha: true,
  preserveDrawingBuffer: false,
  powerPreference: 'high-performance',
};

const vertexShaderSource = `#version 300 es
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;
layout(location = 2) in float a_size;

uniform vec2 u_resolution;
uniform float u_pixelRatio;

out vec4 v_color;

void main() {
  vec2 clipSpace = ((a_position / u_resolution) * 2.0 - 1.0) * vec2(1, -1);
  gl_Position = vec4(clipSpace, 0, 1);
  gl_PointSize = a_size * u_pixelRatio;
  v_color = a_color;
}
`;

const fragmentShaderSource = `#version 300 es
precision highp float;

in vec4 v_color;
out vec4 fragColor;

uniform float u_glowIntensity;

void main() {
  // Circular point with glow effect
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);
  
  if (dist > 0.5) {
    discard;
  }
  
  // Soft edge with glow
  float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
  alpha *= v_color.a;
  
  // Add glow based on intensity uniform
  float glow = exp(-dist * 4.0) * u_glowIntensity;
  fragColor = vec4(v_color.rgb, alpha + glow * 0.3);
}
`;

export class WebGLContext {
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private config: WebGLConfig;
  private uniforms: Map<string, WebGLUniformLocation> = new Map();
  private isInitialized: boolean = false;

  constructor(config: Partial<WebGLConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  initialize(canvas: HTMLCanvasElement): boolean {
    const gl = canvas.getContext('webgl2', this.config);
    
    if (!gl) {
      console.error('[WebGL] Failed to create WebGL2 context');
      return false;
    }

    this.gl = gl;

    // Enable extensions for better performance
    const extensions = [
      'EXT_color_buffer_float',
      'OES_texture_float_linear',
      'WEBGL_color_buffer_float',
    ];

    extensions.forEach((ext) => {
      gl.getExtension(ext);
    });

    // Compile shaders and link program
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource);

    if (!vertexShader || !fragmentShader) {
      console.error('[WebGL] Shader compilation failed');
      return false;
    }

    const program = gl.createProgram();
    if (!program) {
      console.error('[WebGL] Program creation failed');
      return false;
    }

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('[WebGL] Program link failed:', gl.getProgramInfoLog(program));
      return false;
    }

    this.program = program;
    gl.useProgram(program);

    // Cache uniform locations
    this.uniforms.set('u_resolution', gl.getUniformLocation(program, 'u_resolution')!);
    this.uniforms.set('u_pixelRatio', gl.getUniformLocation(program, 'u_pixelRatio')!);
    this.uniforms.set('u_glowIntensity', gl.getUniformLocation(program, 'u_glowIntensity')!);

    // Set up blending for transparent points
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    this.isInitialized = true;
    console.log('[WebGL] Context initialized successfully');
    return true;
  }

  private compileShader(type: number, source: string): WebGLShader | null {
    if (!this.gl) return null;

    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) return null;

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('[WebGL] Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  getGL(): WebGL2RenderingContext | null {
    return this.gl;
  }

  getProgram(): WebGLProgram | null {
    return this.program;
  }

  getUniform(name: string): WebGLUniformLocation | null {
    return this.uniforms.get(name) || null;
  }

  setUniform(name: string, ...values: number[]): void {
    if (!this.gl || !this.program) return;

    const location = this.uniforms.get(name);
    if (!location) return;

    switch (values.length) {
      case 1:
        this.gl.uniform1f(location, values[0]);
        break;
      case 2:
        this.gl.uniform2f(location, values[0], values[1]);
        break;
      case 3:
        this.gl.uniform3f(location, values[0], values[1], values[2]);
        break;
      case 4:
        this.gl.uniform4f(location, values[0], values[1], values[2], values[3]);
        break;
    }
  }

  resize(width: number, height: number, pixelRatio: number = 1): void {
    if (!this.gl) return;

    this.gl.viewport(0, 0, width * pixelRatio, height * pixelRatio);
    this.setUniform('u_resolution', width, height);
    this.setUniform('u_pixelRatio', pixelRatio);
  }

  clear(r: number = 0, g: number = 0, b: number = 0, a: number = 0): void {
    if (!this.gl) return;

    this.gl.clearColor(r, g, b, a);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
  }

  setGlowIntensity(intensity: number): void {
    this.setUniform('u_glowIntensity', intensity);
  }

  destroy(): void {
    if (this.gl && this.program) {
      this.gl.deleteProgram(this.program);
    }
    this.gl = null;
    this.program = null;
    this.uniforms.clear();
    this.isInitialized = false;
  }

  isReady(): boolean {
    return this.isInitialized && this.gl !== null && this.program !== null;
  }
}

export const webGLContext = new WebGLContext();

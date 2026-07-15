import { useEffect, useRef, useCallback } from 'react';
import { WebGLContext } from '../../canvas/WebGLContext';

interface WalkForwardData {
  window: number;
  inSampleSharpe: number;
  outSampleSharpe: number;
  inSampleReturn: number;
  outSampleReturn: number;
}

export const WalkForwardChart: React.FC<{ data: WalkForwardData[] }> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const bufferRef = useRef<WebGLBuffer | null>(null);
  const animationFrameRef = useRef<number>();

  // Pre-allocated vertex buffer
  const vertexBufferRef = useRef<Float32Array>(new Float32Array(10000));
  const vertexCountRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = WebGLContext.init(canvas);
    if (!gl) return;
    glRef.current = gl;

    // Vertex shader
    const vsSource = `#version 300 es
      in vec2 a_position;
      in vec4 a_color;
      
      uniform mat3 u_transform;
      
      out vec4 v_color;
      
      void main() {
        vec3 pos = u_transform * vec3(a_position, 1.0);
        gl_Position = vec4(pos, 1.0);
        v_color = a_color;
      }
    `;

    // Fragment shader with opacity for overfitting visualization
    const fsSource = `#version 300 es
      precision highp float;
      
      in vec4 v_color;
      
      out vec4 fragColor;
      
      void main() {
        fragColor = v_color;
      }
    `;

    const program = WebGLContext.createProgram(gl, vsSource, fsSource);
    if (!program) return;
    programRef.current = program;
    gl.useProgram(program);

    const buffer = gl.createBuffer();
    bufferRef.current = buffer;

    const handleResize = () => {
      canvas.width = canvas.clientWidth * window.devicePixelRatio;
      canvas.height = canvas.clientHeight * window.devicePixelRatio;
      gl.viewport(0, 0, canvas.width, canvas.height);
    };
    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const canvas = canvasRef.current;

    if (!gl || !program || !canvas) return;

    gl.clearColor(0.02, 0.02, 0.05, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Transform matrix
    const transform = new Float32Array([
      2 / canvas.width, 0, -1,
      0, -2 / canvas.height, 1,
      0, 0, 1
    ]);

    const transformLoc = gl.getUniformLocation(program, 'u_transform');
    gl.uniformMatrix3fv(transformLoc, false, transform);

    // Bind data
    const vertexData = vertexBufferRef.current.subarray(0, vertexCountRef.current * 6);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, bufferRef.current);
    gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.DYNAMIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    const colorLoc = gl.getAttribLocation(program, 'a_color');

    const stride = 6 * Float32Array.BYTES_PER_ELEMENT;

    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, stride, 0);

    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * Float32Array.BYTES_PER_ELEMENT);

    // Draw lines
    gl.drawArrays(gl.LINE_STRIP, 0, vertexCountRef.current);
  }, []);

  // Update data
  useEffect(() => {
    let idx = 0;
    const width = canvasRef.current?.clientWidth || 800;
    const height = canvasRef.current?.clientHeight || 400;
    
    // Find min/max for normalization
    let minSharpe = Infinity, maxSharpe = -Infinity;
    data.forEach(d => {
      minSharpe = Math.min(minSharpe, d.inSampleSharpe, d.outSampleSharpe);
      maxSharpe = Math.max(maxSharpe, d.inSampleSharpe, d.outSampleSharpe);
    });
    const range = maxSharpe - minSharpe || 1;

    // In-sample line (cyan)
    data.forEach((d, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = ((d.inSampleSharpe - minSharpe) / range) * height;
      
      vertexBufferRef.current[idx++] = x;
      vertexBufferRef.current[idx++] = y;
      vertexBufferRef.current[idx++] = 0;   // R
      vertexBufferRef.current[idx++] = 1;   // G
      vertexBufferRef.current[idx++] = 1;   // B
      vertexBufferRef.current[idx++] = 0.7; // A
    });

    // Out-of-sample line (magenta, more transparent to show degradation)
    data.forEach((d, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = ((d.outSampleSharpe - minSharpe) / range) * height;
      
      // Reduce opacity based on degradation
      const degradation = Math.max(0, d.inSampleSharpe - d.outSampleSharpe);
      const alpha = Math.max(0.3, 0.8 - degradation * 0.2);
      
      vertexBufferRef.current[idx++] = x;
      vertexBufferRef.current[idx++] = y;
      vertexBufferRef.current[idx++] = 1;   // R
      vertexBufferRef.current[idx++] = 0;   // G
      vertexBufferRef.current[idx++] = 1;   // B
      vertexBufferRef.current[idx++] = alpha;
    });

    vertexCountRef.current = Math.floor(idx / 6);

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    animationFrameRef.current = requestAnimationFrame(render);
  }, [data, render]);

  return (
    <div className="relative w-full h-full bg-gray-950 rounded-lg overflow-hidden border border-cyan-500/20">
      <canvas ref={canvasRef} className="w-full h-full" />
      <div className="absolute top-2 right-2 flex gap-4 text-xs font-mono">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-cyan-500/70 rounded" />
          <span className="text-cyan-400">In-Sample</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-fuchsia-500/70 rounded" />
          <span className="text-fuchsia-400">Out-of-Sample</span>
        </div>
      </div>
    </div>
  );
};

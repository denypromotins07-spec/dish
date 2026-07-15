import { useEffect, useRef, useCallback } from 'react';
import { useReplayStore } from './ReplayStore';
import { WebGLContext } from '../../canvas/WebGLContext';

interface ReplayVertex {
  x: number; // time normalized
  y: number; // price normalized
  type: number; // 0=candle, 1=order, 2=fill
  color: [number, number, number, number];
}

export const TimeMachineCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const bufferRef = useRef<WebGLBuffer | null>(null);
  const animationFrameRef = useRef<number>();
  
  const { currentTime, isPlaying, replayData } = useReplayStore();
  const vertexDataRef = useRef<Float32Array>(new Float32Array(100000)); // Pre-allocated buffer
  const vertexCountRef = useRef<number>(0);

  // Initialize WebGL context and shaders
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = WebGLContext.init(canvas);
    if (!gl) return;
    glRef.current = gl;

    // Vertex shader for time-machine visualization
    const vsSource = `#version 300 es
      in vec4 a_position;
      in vec4 a_color;
      in float a_type;
      
      uniform mat3 u_transform;
      
      out vec4 v_color;
      out float v_type;
      
      void main() {
        vec3 pos = u_transform * vec3(a_position.xy, 1.0);
        gl_Position = vec4(pos, 1.0);
        v_color = a_color;
        v_type = a_type;
      }
    `;

    // Fragment shader with glow effects for fills
    const fsSource = `#version 300 es
      precision highp float;
      
      in vec4 v_color;
      in float v_type;
      
      out vec4 fragColor;
      
      void main() {
        vec4 color = v_color;
        
        // Add glow effect for fill markers
        if (v_type == 2.0) {
          color.rgb *= 1.5;
          color.a *= 0.8;
        }
        
        fragColor = color;
      }
    `;

    const program = WebGLContext.createProgram(gl, vsSource, fsSource);
    if (!program) return;
    programRef.current = program;
    gl.useProgram(program);

    // Create buffer
    const buffer = gl.createBuffer();
    bufferRef.current = buffer;

    // Handle resize
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

  // Render loop
  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const canvas = canvasRef.current;
    
    if (!gl || !program || !canvas) return;

    // Clear canvas
    gl.clearColor(0.02, 0.02, 0.05, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Update transform matrix based on current time viewport
    const transform = new Float32Array([
      2 / canvas.width, 0, -1,
      0, -2 / canvas.height, 1,
      0, 0, 1
    ]);
    
    const transformLoc = gl.getUniformLocation(program, 'u_transform');
    gl.uniformMatrix3fv(transformLoc, false, transform);

    // Bind vertex data
    const vertexData = vertexDataRef.current.subarray(0, vertexCountRef.current * 7); // 7 components per vertex
    
    gl.bindBuffer(gl.ARRAY_BUFFER, bufferRef.current);
    gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.DYNAMIC_DRAW);

    // Setup attributes
    const positionLoc = gl.getAttribLocation(program, 'a_position');
    const colorLoc = gl.getAttribLocation(program, 'a_color');
    const typeLoc = gl.getAttribLocation(program, 'a_type');

    const stride = 7 * Float32Array.BYTES_PER_ELEMENT;
    
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, stride, 0);

    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * Float32Array.BYTES_PER_ELEMENT);

    gl.enableVertexAttribArray(typeLoc);
    gl.vertexAttribPointer(typeLoc, 1, gl.FLOAT, false, stride, 6 * Float32Array.BYTES_PER_ELEMENT);

    // Draw
    gl.drawArrays(gl.POINTS, 0, vertexCountRef.current);

    if (isPlaying) {
      animationFrameRef.current = requestAnimationFrame(render);
    }
  }, [isPlaying]);

  // Update vertex data from replay store
  useEffect(() => {
    const data = replayData;
    let idx = 0;
    
    // Convert replay data to vertex format
    for (let i = 0; i < data.length && idx < 100000; i++) {
      const item = data[i];
      
      // Candle vertex
      vertexDataRef.current[idx++] = item.time;
      vertexDataRef.current[idx++] = item.price;
      vertexDataRef.current[idx++] = item.type;
      vertexDataRef.current[idx++] = item.color[0];
      vertexDataRef.current[idx++] = item.color[1];
      vertexDataRef.current[idx++] = item.color[2];
      vertexDataRef.current[idx++] = item.color[3];
    }
    
    vertexCountRef.current = Math.floor(idx / 7);

    if (isPlaying) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      animationFrameRef.current = requestAnimationFrame(render);
    }
  }, [replayData, isPlaying, render]);

  return (
    <div className="relative w-full h-full bg-gray-950 rounded-lg overflow-hidden border border-cyan-500/20">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ imageRendering: 'pixelated' }}
      />
      <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur-sm rounded text-xs text-cyan-400 font-mono">
        Time: {new Date(currentTime).toISOString()}
      </div>
    </div>
  );
};

import { useEffect, useRef, useCallback } from 'react';
import { WebGLContext } from '../../canvas/WebGLContext';

interface ParameterTrial {
  param1: number;
  param2: number;
  param3: number;
  sharpe: number;
  sortino: number;
}

export const ParameterHeatmap3D: React.FC<{ data: ParameterTrial[] }> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const bufferRef = useRef<WebGLBuffer | null>(null);
  const animationFrameRef = useRef<number>();

  // Pre-allocated buffers
  const vertexBufferRef = useRef<Float32Array>(new Float32Array(50000));
  const instanceBufferRef = useRef<Float32Array>(new Float32Array(50000));
  const pointCountRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = WebGLContext.init(canvas);
    if (!gl) return;
    glRef.current = gl;

    // Enable extensions for instanced rendering
    const ext = gl.getExtension('ANGLE_instanced_arrays');
    
    // Vertex shader for 3D scatter plot
    const vsSource = `#version 300 es
      in vec3 a_position;
      in vec4 a_color;
      in float a_size;
      
      uniform mat4 u_modelViewProjection;
      uniform vec3 u_rotation;
      
      out vec4 v_color;
      
      mat4 rotateX(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat4(
          1.0, 0.0, 0.0, 0.0,
          0.0, c, s, 0.0,
          0.0, -s, c, 0.0,
          0.0, 0.0, 0.0, 1.0
        );
      }
      
      mat4 rotateY(float angle) {
        float c = cos(angle);
        float s = sin(angle);
        return mat4(
          c, 0.0, -s, 0.0,
          0.0, 1.0, 0.0, 0.0,
          s, 0.0, c, 0.0,
          0.0, 0.0, 0.0, 1.0
        );
      }
      
      void main() {
        vec3 pos = a_position;
        
        // Apply rotation
        pos = (rotateY(u_rotation.y) * rotateX(u_rotation.x) * vec4(pos, 1.0)).xyz;
        
        vec4 clipPos = u_modelViewProjection * vec4(pos, 1.0);
        gl_Position = clipPos;
        gl_PointSize = a_size * (300.0 / clipPos.w);
        v_color = a_color;
      }
    `;

    // Fragment shader with glow effect
    const fsSource = `#version 300 es
      precision highp float;
      
      in vec4 v_color;
      
      out vec4 fragColor;
      
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);
        
        if (dist > 0.5) discard;
        
        float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
        fragColor = vec4(v_color.rgb, v_color.a * alpha);
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

  const render = useCallback((rotationX: number, rotationY: number) => {
    const gl = glRef.current;
    const program = programRef.current;
    const canvas = canvasRef.current;

    if (!gl || !program || !canvas) return;

    gl.clearColor(0.02, 0.02, 0.05, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Create MVP matrix
    const aspect = canvas.width / canvas.height;
    const fov = Math.PI / 4;
    const near = 0.1;
    const far = 100.0;
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);

    const projection = new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0
    ]);

    const modelView = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, -5, 1
    ]);

    const mvp = projection.map((v, i) => v * modelView[i]);

    const mvpLoc = gl.getUniformLocation(program, 'u_modelViewProjection');
    const rotationLoc = gl.getUniformLocation(program, 'u_rotation');
    
    gl.uniformMatrix4fv(mvpLoc, false, mvp);
    gl.uniform3f(rotationLoc, rotationX, rotationY, 0);

    // Bind vertex data
    const vertexData = vertexBufferRef.current.subarray(0, pointCountRef.current * 8);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, bufferRef.current);
    gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.DYNAMIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    const colorLoc = gl.getAttribLocation(program, 'a_color');
    const sizeLoc = gl.getAttribLocation(program, 'a_size');

    const stride = 8 * Float32Array.BYTES_PER_ELEMENT;

    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, stride, 0);

    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 3 * Float32Array.BYTES_PER_ELEMENT);

    gl.enableVertexAttribArray(sizeLoc);
    gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, stride, 7 * Float32Array.BYTES_PER_ELEMENT);

    // Draw points
    gl.drawArrays(gl.POINTS, 0, pointCountRef.current);
  }, []);

  // Update data and animate
  useEffect(() => {
    let idx = 0;
    
    // Normalize data
    let minP1 = Infinity, maxP1 = -Infinity;
    let minP2 = Infinity, maxP2 = -Infinity;
    let minP3 = Infinity, maxP3 = -Infinity;
    let maxSharpe = -Infinity;

    data.forEach(d => {
      minP1 = Math.min(minP1, d.param1); maxP1 = Math.max(maxP1, d.param1);
      minP2 = Math.min(minP2, d.param2); maxP2 = Math.max(maxP2, d.param2);
      minP3 = Math.min(minP3, d.param3); maxP3 = Math.max(maxP3, d.param3);
      maxSharpe = Math.max(maxSharpe, d.sharpe);
    });

    const range1 = maxP1 - minP1 || 1;
    const range2 = maxP2 - minP2 || 1;
    const range3 = maxP3 - minP3 || 1;

    data.forEach(d => {
      const x = ((d.param1 - minP1) / range1) * 2 - 1;
      const y = ((d.param2 - minP2) / range2) * 2 - 1;
      const z = ((d.param3 - minP3) / range3) * 2 - 1;
      
      // Color based on Sharpe ratio (blue -> cyan -> green -> yellow -> red)
      const sharpeNorm = d.sharpe / maxSharpe;
      let r, g, b;
      
      if (sharpeNorm < 0.25) {
        r = 0; g = 0; b = 1;
      } else if (sharpeNorm < 0.5) {
        r = 0; g = (sharpeNorm - 0.25) * 4; b = 1 - (sharpeNorm - 0.25) * 4;
      } else if (sharpeNorm < 0.75) {
        r = (sharpeNorm - 0.5) * 4; g = 1; b = 0;
      } else {
        r = 1; g = 1 - (sharpeNorm - 0.75) * 4; b = 0;
      }

      const size = 5 + sharpeNorm * 15;

      vertexBufferRef.current[idx++] = x;
      vertexBufferRef.current[idx++] = y;
      vertexBufferRef.current[idx++] = z;
      vertexBufferRef.current[idx++] = r;
      vertexBufferRef.current[idx++] = g;
      vertexBufferRef.current[idx++] = b;
      vertexBufferRef.current[idx++] = 0.6;
      vertexBufferRef.current[idx++] = size;
    });

    pointCountRef.current = Math.floor(idx / 8);

    let rotationX = 0;
    let rotationY = 0;
    let animId: number;

    const animate = () => {
      rotationY += 0.005;
      render(rotationX, rotationY);
      animId = requestAnimationFrame(animate);
    };
    animate();

    return () => cancelAnimationFrame(animId);
  }, [data, render]);

  return (
    <div className="relative w-full h-full bg-gray-950 rounded-lg overflow-hidden border border-fuchsia-500/20">
      <canvas ref={canvasRef} className="w-full h-full" />
      <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur-sm rounded text-xs text-fuchsia-400 font-mono">
        3D Parameter Space (Sharpe)
      </div>
    </div>
  );
};

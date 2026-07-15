// GLSL Vertex Shader for Liquidity Heatmap
// Optimized for AMD Radeon GCN/RDNA architectures
export const heatmapVertexShader = `#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_price;
layout(location = 2) in float a_volume;
layout(location = 3) in float a_timestamp;

uniform float u_time;
uniform float u_minPrice;
uniform float u_maxPrice;
uniform float u_minTime;
uniform float u_maxTime;
uniform vec2 u_resolution;

out float v_volume;
out float v_age;
out vec2 v_uv;

void main() {
    // Normalize coordinates
    float x = (a_timestamp - u_minTime) / (u_maxTime - u_minTime);
    float y = (a_price - u_minPrice) / (u_maxPrice - u_minPrice);
    
    // Convert to clip space
    vec4 clipPosition = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    
    // Pass data to fragment shader
    v_volume = a_volume;
    v_age = u_time - a_timestamp;
    v_uv = vec2(x, y);
    
    gl_Position = clipPosition;
    gl_PointSize = max(2.0, min(20.0, log2(a_volume + 1.0) * 3.0));
}
`;

// GLSL Fragment Shader with glowing effects and spoofing detection
export const heatmapFragmentShader = `#version 300 es
precision highp float;

in float v_volume;
in float v_age;
in vec2 v_uv;

uniform float u_spoofThreshold;
uniform vec3 u_bidColor;
uniform vec3 u_askColor;
uniform float u_alphaBase;

out vec4 fragColor;

// Gaussian function for glow effect
float gaussian(float x, float sigma) {
    return exp(-x * x / (2.0 * sigma * sigma));
}

// Color mapping based on volume intensity
vec3 heatColor(float intensity) {
    vec3 cold = vec3(0.0, 0.1, 0.2);
    vec3 warm = vec3(1.0, 0.3, 0.0);
    vec3 hot = vec3(1.0, 1.0, 0.8);
    
    float t = clamp(intensity, 0.0, 1.0);
    if (t < 0.5) {
        return mix(cold, warm, t * 2.0);
    } else {
        return mix(warm, hot, (t - 0.5) * 2.0);
    }
}

void main() {
    // Circular point shape
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(gl_PointCoord, center);
    if (dist > 0.5) discard;
    
    // Volume-based intensity with logarithmic scaling
    float intensity = log(v_volume + 1.0) / 15.0;
    intensity = clamp(intensity, 0.0, 1.0);
    
    // Age-based fading for spoofing detection
    // Older orders fade out, pulled orders disappear quickly
    float ageFactor = exp(-v_age * 0.5);
    float spoofFactor = step(v_age, u_spoofThreshold) ? 1.0 : 0.3;
    
    // Combine factors
    float alpha = u_alphaBase * intensity * ageFactor * spoofFactor;
    
    // Apply gaussian glow at edges
    float glow = gaussian(dist * 2.0, 0.6);
    alpha *= glow;
    
    // Determine color based on price position (simplified bid/ask split)
    vec3 baseColor = v_uv.y > 0.5 ? u_askColor : u_bidColor;
    vec3 finalColor = baseColor * heatColor(intensity);
    
    fragColor = vec4(finalColor, alpha);
}
`;

// Compute shader for real-time order book aggregation (WebGL 2.0 extension)
export const heatmapComputeShader = `#version 300 es
precision highp float;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer OrderBookInput {
    float prices[];
};

layout(std430, binding = 1) buffer OrderBookVolumes {
    float volumes[];
};

layout(std430, binding = 2) buffer HeatmapOutput {
    vec4 heatmapData[];
};

uniform int u_numLevels;
uniform float u_priceStep;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_numLevels)) return;
    
    float price = prices[idx];
    float volume = volumes[idx];
    
    // Aggregate nearby levels for smoother visualization
    float aggregatedVolume = 0.0;
    for (int i = -2; i <= 2; i++) {
        int neighborIdx = int(idx) + i;
        if (neighborIdx >= 0 && neighborIdx < u_numLevels) {
            aggregatedVolume += volumes[neighborIdx] * (1.0 - float(abs(i)) * 0.2);
        }
    }
    
    // Encode as RGBA: price normalized, volume, timestamp, flags
    float priceNorm = fract(price / 10000.0);
    heatmapData[idx] = vec4(priceNorm, aggregatedVolume, float(idx) / float(u_numLevels), 0.0);
}
`;

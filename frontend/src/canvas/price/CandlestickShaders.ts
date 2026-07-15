// GLSL Vertex Shader for Instanced Candlestick Rendering
// Renders 50,000+ candles in single draw call via instance attributes
export const candlestickVertexShader = `#version 300 es
precision highp float;

// Instance attributes (per candle)
layout(location = 0) in float a_open;
layout(location = 1) in float a_high;
layout(location = 2) in float a_low;
layout(location = 3) in float a_close;
layout(location = 4) in float a_timestamp;
layout(location = 5) in float a_volume;

uniform float u_minPrice;
uniform float u_maxPrice;
uniform float u_minTime;
uniform float u_maxTime;
uniform vec2 u_resolution;
uniform float u_candleWidth;

out float v_isBullish;
out float v_volume;
out vec2 v_uv;

void main() {
    // Determine if bullish or bearish
    v_isBullish = float(a_close >= a_open);
    v_volume = a_volume;
    
    // Calculate candle position
    float timeNorm = (a_timestamp - u_minTime) / (u_maxTime - u_minTime);
    float priceNormLow = (a_low - u_minPrice) / (u_maxPrice - u_minPrice);
    float priceNormHigh = (a_high - u_minPrice) / (u_maxPrice - u_minPrice);
    float priceNormOpen = (a_open - u_minPrice) / (u_maxPrice - u_minPrice);
    float priceNormClose = (a_close - u_minPrice) / (u_maxPrice - u_minPrice);
    
    // Vertex ID determines which part of candle we're rendering
    // 0-3: Wick, 4-7: Body
    int vertexId = gl_VertexID % 8;
    int instanceId = gl_VertexID / 8;
    
    float xBase = timeNorm * 2.0 - 1.0;
    float halfWidth = u_candleWidth / u_resolution.x;
    
    vec2 position;
    
    if (vertexId < 4) {
        // Wick vertices
        float wickX = (vertexId < 2) ? xBase - halfWidth * 0.3 : xBase + halfWidth * 0.3;
        float wickY = (vertexId % 2 == 0) ? priceNormLow * 2.0 - 1.0 : priceNormHigh * 2.0 - 1.0;
        position = vec2(wickX, wickY);
    } else {
        // Body vertices
        int bodyVertex = vertexId - 4;
        float bodyLeft = xBase - halfWidth;
        float bodyRight = xBase + halfWidth;
        float bodyBottom = min(priceNormOpen, priceNormClose) * 2.0 - 1.0;
        float bodyTop = max(priceNormOpen, priceNormClose) * 2.0 - 1.0;
        
        if (bodyVertex == 0) position = vec2(bodyLeft, bodyBottom);
        else if (bodyVertex == 1) position = vec2(bodyRight, bodyBottom);
        else if (bodyVertex == 2) position = vec2(bodyLeft, bodyTop);
        else position = vec2(bodyRight, bodyTop);
    }
    
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
`;

// GLSL Fragment Shader for Candlesticks with volume-based glow
export const candlestickFragmentShader = `#version 300 es
precision highp float;

in float v_isBullish;
in float v_volume;
in vec2 v_uv;

uniform vec3 u_bullishColor;
uniform vec3 u_bearishColor;
uniform float u_volumeGlow;

out vec4 fragColor;

void main() {
    // Base color based on bullish/bearish
    vec3 baseColor = v_isBullish > 0.5 ? u_bullishColor : u_bearishColor;
    
    // Volume-based intensity (logarithmic scaling)
    float volumeIntensity = log(v_volume + 1.0) / 20.0;
    volumeIntensity = clamp(volumeIntensity, 0.0, 1.0);
    
    // Apply volume glow
    vec3 finalColor = baseColor * (1.0 + volumeIntensity * u_volumeGlow);
    
    // Simple alpha for transparency
    float alpha = 0.9;
    
    fragColor = vec4(finalColor, alpha);
}
`;

// Compute shader for OHLC aggregation (WebGL 2.0)
export const ohlcComputeShader = `#version 300 es
precision highp float;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer TickData {
    vec4 ticks[]; // timestamp, price, volume, aggressorSide
};

layout(std430, binding = 1) buffer CandleData {
    vec4 candles[]; // open, high, low, close
};

uniform int u_numTicks;
uniform int u_ticksPerCandle;
uniform float u_firstTimestamp;

shared float sharedOpen[64];
shared float sharedHigh[64];
shared float sharedLow[64];
shared float sharedClose[64];

void main() {
    uint candleIdx = gl_GlobalInvocationID.x;
    uint tickStart = candleIdx * u_ticksPerCandle;
    
    if (tickStart >= uint(u_numTicks)) return;
    
    // Initialize OHLC from first tick in bucket
    vec4 firstTick = ticks[tickStart];
    float open = firstTick.y;
    float high = firstTick.y;
    float low = firstTick.y;
    float close = firstTick.y;
    
    // Aggregate ticks in this candle bucket
    for (uint i = tickStart; i < min(tickStart + uint(u_ticksPerCandle), uint(u_numTicks)); i++) {
        float price = ticks[i].y;
        high = max(high, price);
        low = min(low, price);
        close = price;
    }
    
    candles[candleIdx] = vec4(open, high, low, close);
}
`;

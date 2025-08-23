#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 texCoord [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float2 texCoord;
};

struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
};

vertex VertexOut vertexShader(const VertexIn vertex_in [[stage_in]],
                            constant Uniforms &uniforms [[buffer(1)]]) {
    VertexOut vertex_out;
    
    // Transform position
    float4 position = float4(vertex_in.position, 1.0);
    position = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix * position;
    vertex_out.position = position;
    
    // Transform normal
    float3x3 normalMatrix = float3x3(uniforms.modelMatrix.columns[0].xyz,
                                   uniforms.modelMatrix.columns[1].xyz,
                                   uniforms.modelMatrix.columns[2].xyz);
    vertex_out.normal = normalize(normalMatrix * vertex_in.normal);
    
    // Pass through texture coordinates
    vertex_out.texCoord = vertex_in.texCoord;
    
    return vertex_out;
}

fragment float4 fragmentShader(VertexOut fragment_in [[stage_in]],
                             texture2d<float> colorTexture [[texture(0)]],
                             sampler textureSampler [[sampler(0)]]) {
    // Basic lighting calculation
    float3 lightDirection = normalize(float3(1.0, 1.0, 1.0));
    float3 normal = normalize(fragment_in.normal);
    float diffuseIntensity = max(0.0, dot(normal, lightDirection));
    
    // Sample texture
    float4 color = colorTexture.sample(textureSampler, fragment_in.texCoord);
    
    // Combine lighting with texture
    float3 finalColor = color.rgb * (diffuseIntensity * 0.8 + 0.2); // Add ambient light
    
    return float4(finalColor, color.a);
}

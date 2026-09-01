#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

#if !defined(__cplusplus) || defined(__INTELLISENSE__)

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

#ifdef __spirv__

struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;

// Descriptor index tables are indexed by the Xenos fetch constant slot (0-31),
// not by the per-stage D3D9 sampler register, so each table is 32 uints wide and
// pixel/vertex samplers can never collide. Layout:
//   [  0.. 127] Texture2D descriptor indices   (32 slots)
//   [128.. 255] Texture3D descriptor indices   (32 slots)
//   [256.. 383] TextureCube descriptor indices (32 slots)
//   [384.. 511] Sampler descriptor indices     (32 slots)
//   [512..    ] scalars below
#define g_Booleans                 vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 512)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 516)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 520)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 528)
#define g_BlendPremultModeRgb     vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 532)
#define g_BlendPremultModeA       vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 536)
#define g_BlendPremultConstant    vk::RawBufferLoad<float4>(g_PushConstants.SharedConstants + 544, 0x10)

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;

#define g_SpecConstants() g_SpecConstants

#else

// Mirrors the __spirv__ byte layout above: 4 tables of 32 uints (c0-c31), then
// the scalars starting at c32.
#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c32.x); \
    uint g_SwappedTexcoords : packoffset(c32.y); \
    float2 g_HalfPixelOffset : packoffset(c32.z); \
    float g_AlphaThreshold : packoffset(c33.x); \
    uint g_BlendPremultModeRgb : packoffset(c33.y); \
    uint g_BlendPremultModeA : packoffset(c33.z); \
    float4 g_BlendPremultConstant : packoffset(c34);

uint g_SpecConstants();

#endif

Texture2D<float4> g_Texture2DDescriptorHeap[] : register(t0, space0);
Texture3D<float4> g_Texture3DDescriptorHeap[] : register(t0, space1);
TextureCube<float4> g_TextureCubeDescriptorHeap[] : register(t0, space2);
SamplerState g_SamplerDescriptorHeap[] : register(s0, space3);

uint2 getTexture2DDimensions(Texture2D<float4> texture)
{
    uint2 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y);
    return dimensions;
}

float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture));
}

// Explicit LOD / LOD bias / explicit gradient variants.
//
// A Xenos tfetch only takes the gradient path when the "use computed LOD" bit is
// set AND the stage can actually produce derivatives. Vertex shaders have no
// quads, so a vertex tfetch is always an explicit LOD fetch on hardware; emitting
// Sample() for one is rejected outright by both DXIL validation ("Opcode Sample
// not valid in shader model vs") and SPIR-V ("sampling with implicit lod is only
// allowed in fragment and compute shaders"). The register LOD written by
// setTexLOD and the instruction LOD bias fold into a plain LOD bias when
// gradients are in play, which is what SampleBias does.
float4 tfetch2DLod(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float lod)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture), lod);
}

float4 tfetch2DBias(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float bias)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.SampleBias(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture), bias);
}

float4 tfetch2DGrad(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float2 gradientH, float2 gradientV)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.SampleGrad(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture), gradientH, gradientV);
}

// Xenos getCompTexLOD returns the LOD the hardware would pick for these
// coordinates, broadcast across the destination components.
float4 getCompTexLod2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.CalculateLevelOfDetail(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord).xxxx;
}

// Xenos getGradients packs the coarse screen-space derivatives of the first two
// source components as (ddx.x, ddy.x, ddx.y, ddy.y).
float4 getGradients(float2 texCoord)
{
    return float4(ddx_coarse(texCoord.x), ddy_coarse(texCoord.x), ddx_coarse(texCoord.y), ddy_coarse(texCoord.y));
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DDimensions(texture) + offset - 0.5));
}

float w0(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a)
{
    return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a)
{
    return (1.0f / 6.0f) * (a * a * a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

float4 tfetch2DBicubic(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    uint2 dimensions = getTexture2DDimensions(texture);
    
    float x = texCoord.x * dimensions.x + offset.x;
    float y = texCoord.y * dimensions.y + offset.y;

    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float4 r =
        g0(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h0y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h0y) / float2(dimensions))) +
        g1(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h1y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h1y) / float2(dimensions)));

    return r;
}

float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord);
}

float4 tfetch3DLod(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float lod)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord, lod);
}

float4 tfetch3DBias(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float bias)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].SampleBias(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord, bias);
}

float4 tfetch3DGrad(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float3 gradientH, float3 gradientV)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].SampleGrad(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord, gradientH, gradientV);
}

struct CubeMapData
{
    float3 cubeMapDirections[2];
    uint cubeMapIndex;
};

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]);
}

float4 tfetchCubeLod(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData, float lod)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z], lod);
}

float4 tfetchCubeBias(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData, float bias)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].SampleBias(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z], bias);
}

float4 tfetchCubeGrad(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData, float3 gradientH, float3 gradientV)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].SampleGrad(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z], gradientH, gradientV);
}

// Packed normal/tangent/binormal unpack.
//
// The spec-constant branch takes the RAW 32-bit vertex dword in value.x, so the
// host must bind the attribute as R32_UINT for it. The other branch is the
// pass-through for a normal that is already float data arriving through this
// uint4-typed input, which asfloat() simply recovers.
//
// The packing is k_2_10_10_10 signed: three 10-bit two's-complement components
// (the 2-bit W is unused by every shader that calls this). Measured on Midnight
// Club: Los Angeles -- 2148 of 2158 normal/tangent attributes in one frame are
// this format, and the raw dword 0x00000201 unpacks to exactly (-1, 0, 0),
// length 1.000, where the older 11/11/10 reading gave (0.501, 0, 0), length
// 0.501. Nothing in that frame used 11/11/10 at all.
//
// Getting this wrong was not a subtle shading error: with neither branch able
// to decode 2_10_10_10, the pass-through ran asfloat() over a small integer --
// a denormal indistinguishable from zero -- so every lit surface in the game
// was shaded with a zero normal.
float4 tfetchR11G11B10(uint4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        int3 packed = int3(value.x, value.x >> 10, value.x >> 20) & 0x3FF;
        // Two's complement over 10 bits, normalised by the positive maximum.
        int3 signed10 = packed - ((packed & 0x200) << 1);
        // The 2-bit W is NOT unused, whatever an earlier comment here claimed:
        // for a TANGENT it carries the binormal handedness, and a shader that
        // needs a binormal builds it as cross(normal, tangent) * tangent.w.
        // Returning 0.0 made that product zero for every vertex, and the pixel
        // shader then normalised a zero-length vector -- rsqrt(0) clamped to
        // FLT_MAX (note FLT_MIN here is -FLT_MAX, so the clamp does not stop
        // it), which lands as +-65504 in an fp16 target and reads as black.
        //
        // Measured on the driver's head (xCharacter_skin_blendshape_normalmap,
        // which was solid black): every one of its 685 pixels came out exactly
        // +-65504, and the two high bits of the tangent take exactly two values
        // over 400 vertices -- 1 on 263 and 3 on 137, i.e. +1 and -1 in 2-bit
        // two's complement. Normals in the same buffer carry 0 there, which is
        // why nothing noticed until a shader read the tangent's.
        int packedW = int(value.x >> 30) & 0x3;
        int signedW = packedW - ((packedW & 0x2) << 1);
        // Normalised by the positive maximum, which for two signed bits is 1.
        return float4(signed10 / 511.0, float(signedW));
    }
    else
    {
        return asfloat(value);
    }
}

float4 tfetchTexcoord(uint swappedTexcoords, float4 value, uint semanticIndex)
{
    return (swappedTexcoords & (1ull << semanticIndex)) != 0 ? value.yxwz : value;
}

float4 cube(float4 value, inout CubeMapData cubeMapData)
{
    uint index = cubeMapData.cubeMapIndex;
    cubeMapData.cubeMapDirections[index] = value.xyz;
    ++cubeMapData.cubeMapIndex;
    
    return float4(0.0, 0.0, 0.0, index);
}

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

float4 max4(float4 src0)
{
    return max(max(src0.x, src0.y), max(src0.z, src0.w));
}

float2 getPixelCoord(uint resourceDescriptorIndex, float2 texCoord)
{
    return getTexture2DDimensions(g_Texture2DDescriptorHeap[resourceDescriptorIndex]) * texCoord;
}

float computeMipLevel(float2 pixelCoord)
{
    float2 dx = ddx(pixelCoord);
    float2 dy = ddy(pixelCoord);
    float deltaMaxSqr = max(dot(dx, dx), dot(dy, dy));
    return max(0.0, 0.5 * log2(deltaMaxSqr));
}

#endif

#endif

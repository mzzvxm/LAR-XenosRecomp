#pragma once

// The Vulkan locations of a vertex shader's inputs.
//
// A guest vertex shader reads up to eight vertex elements, each a D3DDECLUSAGE
// and an index (POSITION0, TEXCOORD3, ...). The HLSL declares each as a stage
// input, and for SPIR-V the input needs a location. They used to be fixed per
// usage (POSITION0 = 0 ... TEXCOORD7 = 15, POSITION1 = 16), so a shader reading
// POSITION1 had an input at location 16 and a pipeline needed 17 vertex input
// attributes, one more than the 16 the Vulkan specification guarantees
// (Android's Mali and PowerVR and Mesa's Dozen report 16).
//
// Now a shader's inputs are numbered 0 to n-1, in the order of the old table
// (the rank below), for the usages the shader uses. Two shaders may give one
// usage different locations: whoever builds a pipeline numbers the inputs of
// the pipeline's vertex shader by the same rule, from the shader's list of
// vertex elements (the renderer does: src/gpu/vertex_locations.h of LARecomp,
// which has the same table and a test that checks the same numbers as
// vertex_locations_test.cpp here).
//
// Only the SPIR-V uses the numbers. DXIL matches inputs by semantic name and
// index, and the compiler ignores [[vk::location]] there, so the DXIL of a
// shader does not change with them.
//
// Standard library only, so that the test needs nothing else.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace xenos_recomp
{

// A vertex input as the container declares it: a D3DDECLUSAGE (DeclUsage in
// shader.h) and its index.
struct VertexInput
{
    uint32_t usage;
    uint32_t usageIndex;
};

// The order of the old fixed table; a usage's old location is its rank.
// DeclUsage values: Position 0, BlendWeight 1, BlendIndices 2, Normal 3,
// TexCoord 5, Tangent 6, Binormal 7, Color 10.
inline constexpr VertexInput kVertexInputOrder[] =
{
    { 0, 0 },  // Position 0
    { 3, 0 },  // Normal 0
    { 6, 0 },  // Tangent 0
    { 7, 0 },  // Binormal 0
    { 5, 0 },  // TexCoord 0
    { 5, 1 },  // TexCoord 1
    { 5, 2 },  // TexCoord 2
    { 5, 3 },  // TexCoord 3
    { 10, 0 }, // Color 0
    { 2, 0 },  // BlendIndices 0
    { 1, 0 },  // BlendWeight 0
    { 10, 1 }, // Color 1
    { 5, 4 },  // TexCoord 4
    { 5, 5 },  // TexCoord 5
    { 5, 6 },  // TexCoord 6
    { 5, 7 },  // TexCoord 7
    { 0, 1 },  // Position 1: MCLA's own declarations use it in 8 vertex shaders
};

inline constexpr size_t kVertexInputOrderCount = sizeof(kVertexInputOrder) / sizeof(kVertexInputOrder[0]);

// The vertex inputs every Vulkan device guarantees (maxVertexInputAttributes,
// at least 16). A shader with more is still translated, with locations 0 to n-1;
// whoever builds the pack warns, and the renderer compares the pack's largest
// count with the device's limit.
inline constexpr uint32_t kVertexInputsGuaranteed = 16;

// Where a usage stands in the order. A usage the table does not list (the game
// declares none) comes after all that it does, by usage and then index (both
// are four bits in a vertex element), so every input has a place.
constexpr uint32_t vertexInputRank(uint32_t usage, uint32_t usageIndex)
{
    for (size_t i = 0; i < kVertexInputOrderCount; i++)
    {
        if (kVertexInputOrder[i].usage == usage && kVertexInputOrder[i].usageIndex == usageIndex)
            return uint32_t(i);
    }

    return uint32_t(kVertexInputOrderCount) + (usage & 0xF) * 16 + (usageIndex & 0xF);
}

// The location of each input: its place in the inputs sorted by rank, 0 to n-1.
// locations[i] belongs to inputs[i], whatever order the container lists them in.
// Equal ranks (a usage declared twice, which no valid shader does) keep the
// order of the list.
inline std::vector<uint32_t> assignVertexLocations(const VertexInput* inputs, size_t count)
{
    std::vector<uint32_t> locations(count);

    for (size_t i = 0; i < count; i++)
    {
        const uint32_t rank = vertexInputRank(inputs[i].usage, inputs[i].usageIndex);
        uint32_t before = 0;

        for (size_t j = 0; j < count; j++)
        {
            const uint32_t other = vertexInputRank(inputs[j].usage, inputs[j].usageIndex);

            if (other < rank || (other == rank && j < i))
                before++;
        }

        locations[i] = before;
    }

    return locations;
}

}

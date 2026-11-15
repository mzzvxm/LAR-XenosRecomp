// A check of how a vertex shader's inputs are numbered (vertex_locations.h), and of
// the HLSL the translator writes from it. No game data: the container is made here.
//
//   cmake --build <dir> --target XenosRecompVertexLocationsTest
//   run XenosRecompVertexLocationsTest (exit 0 = passed)
//
// LARecomp's src/gpu/vertex_locations_test.cpp checks the same numbers (kGolden below
// is its table): the renderer numbers a pipeline's inputs by its own copy of the rule.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "vertex_locations.h"
#include "xenos_recomp.h"

namespace
{

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const char* what, int line)
{
    ++g_checks;
    if (!ok)
    {
        ++g_failed;
        std::printf("FAIL line %d: %s\n", line, what);
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

using xenos_recomp::VertexInput;

// DeclUsage values.
constexpr uint32_t Position = 0, BlendWeight = 1, BlendIndices = 2, Normal = 3, PointSize = 4, TexCoord = 5,
                   Tangent = 6, Binormal = 7, Color = 10;

std::vector<uint32_t> locationsOf(const std::vector<VertexInput>& inputs)
{
    return xenos_recomp::assignVertexLocations(inputs.data(), inputs.size());
}

// ---------------------------------------------------------------------------
// The rule
// ---------------------------------------------------------------------------

void testRule()
{
    // Nothing in, nothing out.
    CHECK(locationsOf({}).empty());

    // The case the old table could not do in 16: POSITION1 and TEXCOORD0. The old order is kept
    // (TEXCOORD0 came before POSITION1), so TEXCOORD0 is 0 and POSITION1 is 1, however the
    // container lists them.
    {
        const auto a = locationsOf({ { Position, 1 }, { TexCoord, 0 } });
        CHECK(a.size() == 2 && a[0] == 1 && a[1] == 0);
        const auto b = locationsOf({ { TexCoord, 0 }, { Position, 1 } });
        CHECK(b.size() == 2 && b[0] == 0 && b[1] == 1);
    }

    // A shader that uses POSITION0 and POSITION1 only: 0 and 1.
    {
        const auto a = locationsOf({ { Position, 1 }, { Position, 0 } });
        CHECK(a[0] == 1 && a[1] == 0);
    }

    // The old numbers: all seventeen usages of the old table, in the old order, are numbered 0 to
    // 16 as they were (the table's row is its location).
    {
        std::vector<VertexInput> all(std::begin(xenos_recomp::kVertexInputOrder), std::end(xenos_recomp::kVertexInputOrder));
        const auto locations = locationsOf(all);
        CHECK(all.size() == 17 && locations.size() == 17);
        for (uint32_t i = 0; i < locations.size(); i++)
            CHECK(locations[i] == i);
    }

    // The same, handed over in reverse: every usage keeps its number.
    {
        std::vector<VertexInput> all(std::begin(xenos_recomp::kVertexInputOrder), std::end(xenos_recomp::kVertexInputOrder));
        std::reverse(all.begin(), all.end());
        const auto locations = locationsOf(all);
        for (uint32_t i = 0; i < locations.size(); i++)
            CHECK(locations[i] == uint32_t(all.size()) - 1 - i);
    }

    // The old table's values, spelled out (this is the table of LARecomp's test too).
    struct Golden
    {
        uint32_t usage, index, oldLocation;
    };
    static constexpr Golden kGolden[] =
    {
        { Position, 0, 0 },   { Normal, 0, 1 },       { Tangent, 0, 2 },     { Binormal, 0, 3 },
        { TexCoord, 0, 4 },   { TexCoord, 1, 5 },     { TexCoord, 2, 6 },    { TexCoord, 3, 7 },
        { Color, 0, 8 },      { BlendIndices, 0, 9 }, { BlendWeight, 0, 10 }, { Color, 1, 11 },
        { TexCoord, 4, 12 },  { TexCoord, 5, 13 },    { TexCoord, 6, 14 },   { TexCoord, 7, 15 },
        { Position, 1, 16 },
    };
    CHECK(std::size(kGolden) == xenos_recomp::kVertexInputOrderCount);
    for (const Golden& g : kGolden)
        CHECK(xenos_recomp::vertexInputRank(g.usage, g.index) == g.oldLocation);

    // A typical skinned mesh shader: POSITION0, NORMAL0, TEXCOORD0, TEXCOORD1, COLOR0, BLENDINDICES0,
    // BLENDWEIGHT0 (listed in container order, which is not the table's).
    {
        const auto a = locationsOf({ { BlendWeight, 0 }, { Color, 0 }, { TexCoord, 1 }, { Position, 0 }, { Normal, 0 },
                                     { BlendIndices, 0 }, { TexCoord, 0 } });
        // by rank: P0 N0 TC0 TC1 C0 BI0 BW0 -> 0 1 2 3 4 5 6
        CHECK(a[3] == 0 && a[4] == 1 && a[6] == 2 && a[2] == 3 && a[1] == 4 && a[5] == 5 && a[0] == 6);
    }

    // A usage the old table lists nothing for (the game has none) comes after the listed ones, by
    // usage then index, and every input still has its own number.
    {
        const auto a = locationsOf({ { PointSize, 0 }, { Position, 1 }, { 13, 3 }, { PointSize, 1 } });
        CHECK(a[1] == 0);  // POSITION1, the last of the table, still before the rest
        CHECK(a[0] == 1 && a[3] == 2 && a[2] == 3);
    }

    // Every result is a permutation of 0..n-1, for each size up to the 17 the table has.
    for (size_t n = 1; n <= xenos_recomp::kVertexInputOrderCount; n++)
    {
        std::vector<VertexInput> some(xenos_recomp::kVertexInputOrder, xenos_recomp::kVertexInputOrder + n);
        std::reverse(some.begin(), some.end());
        auto locations = locationsOf(some);
        std::sort(locations.begin(), locations.end());
        bool permutation = locations.size() == n;
        for (uint32_t i = 0; permutation && i < n; i++)
            permutation = locations[i] == i;
        CHECK(permutation);
    }

    // The guaranteed count is the Vulkan minimum.
    CHECK(xenos_recomp::kVertexInputsGuaranteed == 16);
}

// ---------------------------------------------------------------------------
// A vertex shader container made in memory (big endian, no code), through the translator
// ---------------------------------------------------------------------------

void put32(std::vector<uint8_t>& bytes, size_t at, uint32_t value)
{
    bytes[at + 0] = uint8_t(value >> 24);
    bytes[at + 1] = uint8_t(value >> 16);
    bytes[at + 2] = uint8_t(value >> 8);
    bytes[at + 3] = uint8_t(value);
}

// A vertex shader with these vertex elements, no interpolators and no instructions.
std::vector<uint8_t> makeVertexShaderContainer(const std::vector<VertexInput>& elements)
{
    constexpr uint32_t virtualSize = 512;
    constexpr uint32_t physicalSize = 16;
    constexpr uint32_t constantTableOffset = 36;
    constexpr uint32_t shaderOffset = 72;

    std::vector<uint8_t> bytes(virtualSize + physicalSize, 0);

    // ShaderContainer: flags (bit 0: vertex), virtualSize, physicalSize, fieldC, constantTableOffset,
    // definitionTableOffset (0: none), shaderOffset, field1C, field20.
    put32(bytes, 0, 0x102A1101);
    put32(bytes, 4, virtualSize);
    put32(bytes, 8, physicalSize);
    put32(bytes, 16, constantTableOffset);
    put32(bytes, 24, shaderOffset);

    // ConstantTableContainer: size, then ConstantTable: size, creator, version, constants (0), ...
    put32(bytes, constantTableOffset, 32);
    put32(bytes, constantTableOffset + 4, 28);

    // VertexShader: physicalOffset (0), size (0 instructions), field8, fieldC, field10, interpolatorInfo
    // (0 interpolators), field18 (0), vertexElementCount, field20, then the elements.
    put32(bytes, shaderOffset + 28, uint32_t(elements.size()));
    for (size_t i = 0; i < elements.size(); i++)
    {
        // address:12 | usage:4 | usageIndex:4, the address being the vfetch's own (unused here).
        const uint32_t word = uint32_t(i) | (elements[i].usage << 12) | (elements[i].usageIndex << 16);
        put32(bytes, shaderOffset + 36 + i * 4, word);
    }

    return bytes;
}

// The line of the HLSL that declares the input `variable`.
std::string declarationOf(const std::string& hlsl, const std::string& variable)
{
    const size_t at = hlsl.find(" " + variable + " : ");
    if (at == std::string::npos)
        return {};
    const size_t begin = hlsl.rfind('\n', at) + 1;
    const size_t end = hlsl.find('\n', at);
    return hlsl.substr(begin, end - begin);
}

xenos_recomp::RecompiledShader translate(const std::vector<VertexInput>& elements, bool* ok)
{
    const std::vector<uint8_t> container = makeVertexShaderContainer(elements);
    xenos_recomp::RecompiledShader shader;
    *ok = xenos_recomp::RecompileContainer(container.data(), container.size(), "// test include\n", shader);
    return shader;
}

void testTranslation()
{
    // POSITION1 + TEXCOORD0: TEXCOORD0 at 0, POSITION1 at 1, in the order of the old table.
    {
        bool ok = false;
        const auto shader = translate({ { Position, 1 }, { TexCoord, 0 } }, &ok);
        CHECK(ok);
        CHECK(!shader.isPixelShader);
        CHECK(shader.vertexInputCount == 2);
        CHECK(declarationOf(shader.hlsl, "iPosition1") == "\t[[vk::location(1)]] in float4 iPosition1 : POSITION1,");
        CHECK(declarationOf(shader.hlsl, "iTexCoord0") == "\t[[vk::location(0)]] in float4 iTexCoord0 : TEXCOORD0,");
        // The semantic of each input (what D3D12 matches by) is what it was.
        CHECK(shader.hlsl.find("iPosition1 : POSITION1,") != std::string::npos);
        CHECK(shader.hlsl.find("iTexCoord0 : TEXCOORD0,") != std::string::npos);
    }

    // The container's own order does not matter.
    {
        bool ok = false;
        const auto shader = translate({ { TexCoord, 0 }, { Position, 1 } }, &ok);
        CHECK(ok);
        CHECK(declarationOf(shader.hlsl, "iPosition1") == "\t[[vk::location(1)]] in float4 iPosition1 : POSITION1,");
        CHECK(declarationOf(shader.hlsl, "iTexCoord0") == "\t[[vk::location(0)]] in float4 iTexCoord0 : TEXCOORD0,");
    }

    // Packed normals stay uint4 and take their place too: NORMAL0 after POSITION0.
    {
        bool ok = false;
        const auto shader = translate({ { Normal, 0 }, { Position, 0 } }, &ok);
        CHECK(ok);
        CHECK(declarationOf(shader.hlsl, "iPosition0") == "\t[[vk::location(0)]] in float4 iPosition0 : POSITION0,");
        CHECK(declarationOf(shader.hlsl, "iNormal0") == "\t[[vk::location(1)]] in uint4 iNormal0 : NORMAL0,");
    }

    // A usage the old table never listed gets a location too (a shader that numbers some inputs
    // has to number them all).
    {
        bool ok = false;
        const auto shader = translate({ { PointSize, 0 }, { Position, 0 } }, &ok);
        CHECK(ok);
        CHECK(declarationOf(shader.hlsl, "iPointSize0") == "\t[[vk::location(1)]] in float4 iPointSize0 : PSIZE0,");
    }

    // More than the 16 inputs Vulkan guarantees is translated all the same (the caller warns):
    // all 17 usages of the old table, locations 0 to 16, and the count is reported.
    {
        const std::vector<VertexInput> all(std::begin(xenos_recomp::kVertexInputOrder), std::end(xenos_recomp::kVertexInputOrder));
        bool ok = false;
        const auto shader = translate(all, &ok);
        CHECK(ok);
        CHECK(shader.vertexInputCount == 17);
        CHECK(shader.vertexInputCount > xenos_recomp::kVertexInputsGuaranteed);
        CHECK(!shader.hlsl.empty());
        for (uint32_t i = 0; i < 17; i++)
        {
            const std::string at = "[[vk::location(" + std::to_string(i) + ")]] in ";
            CHECK(shader.hlsl.find(at) != std::string::npos);
        }
        CHECK(shader.hlsl.find("[[vk::location(17)]]") == std::string::npos);
    }

    // A shader with no vertex elements has no locations, and says so.
    {
        bool ok = false;
        const auto shader = translate({}, &ok);
        CHECK(ok);
        CHECK(shader.vertexInputCount == 0);
        CHECK(shader.hlsl.find("[[vk::location(") == std::string::npos);
    }
}

}  // namespace

int main()
{
    testRule();
    testTranslation();
    std::printf("vertex_locations_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}

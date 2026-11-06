#pragma once

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>

#include <fmt/format.h>

#include "shader.h"
#include "shader_code.h"
#include "xenos_recomp.h" // ControlFlowForm

struct StringBuffer
{
    std::string out;

    template<class... Args>
    void print(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
    }

    template<class... Args>
    void println(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
        out += '\n';
    }
};

struct ShaderRecompiler : StringBuffer
{
    uint32_t indentation = 0;
    bool isPixelShader = false;
    const uint8_t* constantTableData = nullptr;
    std::unordered_map<uint32_t, VertexElement> vertexElements;
    std::unordered_map<uint32_t, std::string> interpolators;
    std::unordered_map<uint32_t, const ConstantInfo*> float4Constants;
    std::unordered_map<uint32_t, const char*> boolConstants;
    std::unordered_map<uint32_t, const char*> samplers;
    uint32_t specConstantsMask = 0;
    ControlFlowForm controlFlowForm = ControlFlowForm::Flat;

    // Set when the shader refers to something its container never declares (a
    // vertex element or an interpolator). recompile() then returns false.
    bool malformed = false;

    // Set by a pre-pass over the control flow when the shader actually issues
    // setTexLOD / setGradientsHorz / setGradientsVert. Shaders that never touch
    // the implicit fetch state keep emitting plain Sample() calls.
    bool usesTextureLodRegister = false;
    bool usesTextureGradientRegisters = false;

#ifdef UNLEASHED_RECOMP
    bool hasMtxProjection = false;
    bool hasMtxPrevInvViewProjection = false;
#endif

    void indent()
    {
        for (uint32_t i = 0; i < indentation; i++)
            out += '\t';
    }

    void printDstSwizzle(uint32_t dstSwizzle, bool operand);
    void printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle);

    void recompile(const VertexFetchInstruction& instr, uint32_t address);
    void recompile(const TextureFetchInstruction& instr, bool bicubic);
    void recompile(const AluInstruction& instr);

    // Returns false, instead of asserting, when the data is not a usable shader
    // container. The output is only meaningful when it returns true.
    bool recompile(const uint8_t* shaderData, const std::string_view& include);
};

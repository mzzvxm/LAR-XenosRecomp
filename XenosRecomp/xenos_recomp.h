#pragma once

// The interface for programs that embed the translator: link XenosRecompLib and
// include this header as "XenosRecomp/xenos_recomp.h". It needs nothing but the
// standard library. Finding the containers in a file or a memory image is
// container_scan.h.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// How the control flow program of a shader was translated; see planControlFlow
// in shader_recompiler.cpp.
enum class ControlFlowForm
{
    Flat,       // no jumps
    Structured, // jumps expressed as if / if-else statements
    PcMachine   // a switch over the control flow index inside a loop
};

namespace xenos_recomp
{

struct RecompiledShader
{
    std::string hlsl; // starts with the include text it was given
    uint32_t specConstantsMask = 0;
    bool isPixelShader = false;
    ControlFlowForm controlFlowForm = ControlFlowForm::Flat;
};

// Translates one Xbox 360 shader container to HLSL. container points at the
// start of the container and size is how many bytes are readable from there (the
// size container_scan.h reports, or more). include is the text pasted at the top
// of the HLSL, normally ShaderCommonText().
//
// Returns false, and leaves result empty, when the data is not a usable shader
// container. Only the header and the lookups the translation itself makes are
// checked: containers are expected to come from the game's own files, found with
// forEachShaderContainer. Keeps no state between calls, so any number of threads
// can call it at once.
bool RecompileContainer(const uint8_t* container, size_t size, std::string_view include, RecompiledShader& result);

// The commit this translator was built from (a git hash, or "unknown"). The HLSL
// it produces can change with every commit, so anything built from it should be
// rebuilt when this changes.
const char* Revision();

// shader_common.h, the text the generated HLSL starts with, embedded when the
// library is built. kShaderCommonTextSize does not count a terminator; there is
// none.
extern const uint8_t kShaderCommonText[];
extern const size_t kShaderCommonTextSize;

inline std::string_view ShaderCommonText()
{
    return std::string_view(reinterpret_cast<const char*>(kShaderCommonText), kShaderCommonTextSize);
}

}

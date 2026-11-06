#include "xenos_recomp.h"

#include <utility>

#include "container_scan.h"
#include "shader_recompiler.h"

#ifndef XENOS_RECOMP_REVISION
#define XENOS_RECOMP_REVISION "unknown"
#endif

namespace xenos_recomp
{

bool RecompileContainer(const uint8_t* container, size_t size, std::string_view include, RecompiledShader& result)
{
    result = {};

    size_t containerSize = 0;
    if (container == nullptr || size < sizeof(ShaderContainer) || !matchShaderContainer(container, size, containerSize))
        return false;

    ShaderRecompiler recompiler;
    if (!recompiler.recompile(container, include))
        return false;

    result.hlsl = std::move(recompiler.out);
    result.specConstantsMask = recompiler.specConstantsMask;
    result.isPixelShader = recompiler.isPixelShader;
    result.controlFlowForm = recompiler.controlFlowForm;
    return true;
}

const char* Revision()
{
    return XENOS_RECOMP_REVISION;
}

}

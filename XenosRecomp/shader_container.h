#pragma once

#include <cstdint>

#include "be.h"

struct ShaderContainer
{
    be<uint32_t> flags;
    be<uint32_t> virtualSize;
    be<uint32_t> physicalSize;
    be<uint32_t> fieldC;
    be<uint32_t> constantTableOffset;
    be<uint32_t> definitionTableOffset;
    be<uint32_t> shaderOffset;
    be<uint32_t> field1C;
    be<uint32_t> field20;
};

#pragma once

// Finding and naming Xbox 360 shader containers inside arbitrary data. Header
// only, so the command line tool and other programs share one implementation.

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "shader.h"

// True when the available bytes at data start with a shader container. dataSize
// receives the size the container declares for itself. This is the signature
// test forEachShaderContainer walks with, see there.
inline bool matchShaderContainer(const uint8_t* data, size_t available, size_t& dataSize)
{
    auto shaderContainer = reinterpret_cast<const ShaderContainer*>(data);
    dataSize = shaderContainer->virtualSize + shaderContainer->physicalSize;

    return (shaderContainer->flags & 0xFFFFFF00) == 0x102A1100 &&
        dataSize != 0 &&
        dataSize <= available &&
        shaderContainer->constantTableOffset != 0 &&
        shaderContainer->constantTableOffset < dataSize &&
        shaderContainer->shaderOffset < dataSize &&
        shaderContainer->field1C == 0 &&
        shaderContainer->field20 == 0;
}

// Walks a blob looking for embedded Xbox 360 shader containers.
//
// Container boundaries are NOT guaranteed to be 4 byte aligned. RAGE's "rgxa"
// effect bundles (Midnight Club: Los Angeles and friends) prefix every container
// with an unpadded name/parameter string table followed by two little endian
// uint16 sizes, so containers routinely start at offsets that are 1, 2 or 3 mod
// 4. Stepping by sizeof(uint32_t) silently skipped ~75% of the shaders in those
// bundles, so the walk advances one byte at a time. The signature test is strong
// enough to make that safe: the top 24 bits of the flags must match, field1C and
// field20 must both be zero, and the declared size has to fit in the remaining
// bytes. On a hit the walk skips the whole container, so the extra granularity
// only costs work in the gaps between shaders.
template<typename F>
inline void forEachShaderContainer(const uint8_t* data, size_t size, F&& callback)
{
    for (size_t i = 0; size > sizeof(ShaderContainer) && i < size - sizeof(ShaderContainer) - 1;)
    {
        size_t dataSize = 0;

        if (matchShaderContainer(data + i, size - i, dataSize))
        {
            callback(i, dataSize);
            i += dataSize;
        }
        else
        {
            i += 1;
        }
    }
}

// Xbox 360 shader containers built with debug info carry the absolute path of
// the .updb file they were compiled from, e.g.
//   C:\soft\mc4\game\src\mcCarShaders\xblackmatte\VS_ZPre.updb
// That is the only human readable identity a container has once it has been
// packed into a bundle, so it is used to name the emitted HLSL.
struct ShaderName
{
    std::string group;      // "xblackmatte"
    std::string entryPoint; // "VS_ZPre"
    bool valid = false;
};

inline ShaderName extractShaderName(const uint8_t* data, size_t size)
{
    ShaderName result;

    static const char suffix[] = ".updb";
    constexpr size_t suffixLength = sizeof(suffix) - 1;

    for (size_t i = 0; i + suffixLength <= size; i++)
    {
        if (std::memcmp(data + i, suffix, suffixLength) != 0)
            continue;

        // Walk backwards over the printable path characters.
        size_t begin = i;
        while (begin > 0)
        {
            char c = char(data[begin - 1]);
            if (c < 0x20 || c > 0x7E)
                break;
            --begin;
        }

        std::string path(reinterpret_cast<const char*>(data + begin), i - begin);
        if (path.empty())
            continue;

        size_t lastSep = path.find_last_of("\\/");
        if (lastSep == std::string::npos)
            continue;

        result.entryPoint = path.substr(lastSep + 1);

        std::string parent = path.substr(0, lastSep);
        size_t prevSep = parent.find_last_of("\\/");
        result.group = (prevSep == std::string::npos) ? parent : parent.substr(prevSep + 1);
        result.valid = !result.entryPoint.empty();
        return result;
    }

    return result;
}

inline std::string sanitizeFileName(const std::string& value)
{
    std::string result;
    result.reserve(value.size());
    for (char c : value)
        result += (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') ? c : '_';
    return result;
}

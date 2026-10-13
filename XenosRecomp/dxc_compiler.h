#pragma once

struct DxcCompiler
{
    IDxcCompiler3* dxcCompiler = nullptr;

    DxcCompiler();
    ~DxcCompiler();

    // errorsOut, when non-null, receives the DXC diagnostic text instead of it
    // being written straight to stderr. Batch modes need the message attached to
    // the shader that produced it rather than interleaved across worker threads.
    //
    // staticSpecConstants, when not NO_STATIC_SPEC_CONSTANTS, compiles the shader
    // with its spec constants fixed (-D XENOS_SPEC_CONSTANTS=<value>, see
    // shader_common.h): one variant, as the LAR pack builder compiles it.
    static constexpr uint32_t NO_STATIC_SPEC_CONSTANTS = ~0u;
    IDxcBlob* compile(const std::string& shaderSource, bool compilePixelShader, bool compileLibrary, bool compileSpirv, std::string* errorsOut = nullptr,
        uint32_t staticSpecConstants = NO_STATIC_SPEC_CONSTANTS);
};

#pragma once

struct DxcCompiler
{
    IDxcCompiler3* dxcCompiler = nullptr;

    DxcCompiler();
    ~DxcCompiler();

    // errorsOut, when non-null, receives the DXC diagnostic text instead of it
    // being written straight to stderr. Batch modes need the message attached to
    // the shader that produced it rather than interleaved across worker threads.
    IDxcBlob* compile(const std::string& shaderSource, bool compilePixelShader, bool compileLibrary, bool compileSpirv, std::string* errorsOut = nullptr);
};

#include "shader.h"
#include "shader_recompiler.h"
#include "dxc_compiler.h"

static std::unique_ptr<uint8_t[]> readAllBytes(const char* filePath, size_t& fileSize)
{
    FILE* file = fopen(filePath, "rb");
    fseek(file, 0, SEEK_END);
    fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    auto data = std::make_unique<uint8_t[]>(fileSize);
    fread(data.get(), 1, fileSize, file);
    fclose(file);
    return data;
}

static void writeAllBytes(const char* filePath, const void* data, size_t dataSize)
{
    FILE* file = fopen(filePath, "wb");
    fwrite(data, 1, dataSize, file);
    fclose(file);
}

struct RecompiledShader
{
    uint8_t* data = nullptr;
    IDxcBlob* dxil = nullptr;
    std::vector<uint8_t> spirv;
    uint32_t specConstantsMask = 0;
};

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
static void forEachShaderContainer(const uint8_t* data, size_t size, F&& callback)
{
    for (size_t i = 0; size > sizeof(ShaderContainer) && i < size - sizeof(ShaderContainer) - 1;)
    {
        auto shaderContainer = reinterpret_cast<const ShaderContainer*>(data + i);
        size_t dataSize = shaderContainer->virtualSize + shaderContainer->physicalSize;

        if ((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100 &&
            dataSize != 0 &&
            dataSize <= (size - i) &&
            shaderContainer->constantTableOffset != 0 &&
            shaderContainer->constantTableOffset < dataSize &&
            shaderContainer->shaderOffset < dataSize &&
            shaderContainer->field1C == 0 &&
            shaderContainer->field20 == 0)
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

static ShaderName extractShaderName(const uint8_t* data, size_t size)
{
    ShaderName result;

    static const char suffix[] = ".updb";
    constexpr size_t suffixLength = sizeof(suffix) - 1;

    for (size_t i = 0; i + suffixLength <= size; i++)
    {
        if (memcmp(data + i, suffix, suffixLength) != 0)
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

static std::string sanitizeFileName(const std::string& value)
{
    std::string result;
    result.reserve(value.size());
    for (char c : value)
        result += (isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') ? c : '_';
    return result;
}

struct HlslShader
{
    const uint8_t* data = nullptr;
    size_t size = 0;
    std::string sourceFile;   // path relative to the input directory
    std::string category;     // first path component of sourceFile
    ShaderName name;
    XXH64_hash_t hash = 0;

    std::string hlsl;
    uint32_t specConstantsMask = 0;
    bool isPixelShader = false;
    bool recompiled = false;
    bool dxilOk = false;
    bool spirvOk = false;
    std::string dxilError;
    std::string spirvError;
    std::string outputPath;
};

// Emits one .hlsl per unique shader container found under the input directory and
// validates each one through DXC for both DXIL and SPIR-V. Failures are recorded
// per shader instead of aborting, so a single bad shader cannot hide the state of
// the other few thousand.
static int recompileDirectoryToHlsl(const char* input, const char* output, const std::string_view& include)
{
    std::filesystem::path inputRoot(input);
    std::filesystem::path outputRoot(output);
    std::filesystem::create_directories(outputRoot);

    std::vector<std::unique_ptr<uint8_t[]>> files;
    std::vector<HlslShader> shaders;
    std::unordered_map<XXH64_hash_t, size_t> uniqueShaders;
    size_t duplicateCount = 0;
    size_t scannedFileCount = 0;

    for (auto& file : std::filesystem::recursive_directory_iterator(inputRoot))
    {
        if (file.is_directory())
            continue;

        ++scannedFileCount;

        size_t fileSize = 0;
        auto fileData = readAllBytes(file.path().string().c_str(), fileSize);
        bool foundAny = false;

        std::string relative = std::filesystem::relative(file.path(), inputRoot).generic_string();
        std::string category = relative.substr(0, relative.find('/'));
        if (category == relative)
            category = "root";

        forEachShaderContainer(fileData.get(), fileSize, [&](size_t offset, size_t dataSize)
            {
                const uint8_t* blob = fileData.get() + offset;
                XXH64_hash_t hash = XXH3_64bits(blob, dataSize);

                auto inserted = uniqueShaders.try_emplace(hash, shaders.size());
                if (!inserted.second)
                {
                    ++duplicateCount;
                    return;
                }

                HlslShader shader;
                shader.data = blob;
                shader.size = dataSize;
                shader.sourceFile = relative;
                shader.category = category;
                shader.name = extractShaderName(blob, dataSize);
                shader.hash = hash;
                shaders.emplace_back(std::move(shader));
                foundAny = true;
            });

        if (foundAny)
            files.emplace_back(std::move(fileData));
    }

    fmt::println("Scanned {} files, found {} unique shader containers ({} duplicates).",
        scannedFileCount, shaders.size(), duplicateCount);

    if (shaders.empty())
        return 1;

    // Reserve output paths serially so the naming is deterministic and collision
    // free before anything runs in parallel.
    std::unordered_map<std::string, uint32_t> nameUses;
    for (auto& shader : shaders)
    {
        std::string base;
        if (shader.name.valid)
        {
            base = sanitizeFileName(shader.name.group);
            if (!base.empty())
                base += "__";
            base += sanitizeFileName(shader.name.entryPoint);
        }
        else
        {
            base = fmt::format("unnamed_{:016X}", shader.hash);
        }

        uint32_t use = nameUses[base]++;
        if (use != 0)
            base += fmt::format("__{}", use);

        auto directory = outputRoot / shader.category;
        std::filesystem::create_directories(directory);
        shader.outputPath = (directory / (base + ".hlsl")).string();
    }

    std::atomic<uint32_t> progress = 0;

    std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](HlslShader& shader)
        {
            thread_local ShaderRecompiler recompiler;
            recompiler = {};
            recompiler.recompile(shader.data, include);

            shader.hlsl = recompiler.out;
            shader.specConstantsMask = recompiler.specConstantsMask;
            shader.isPixelShader = recompiler.isPixelShader;
            shader.recompiled = !shader.hlsl.empty();

            writeAllBytes(shader.outputPath.c_str(), shader.hlsl.data(), shader.hlsl.size());

            thread_local DxcCompiler dxcCompiler;

            IDxcBlob* dxil = dxcCompiler.compile(shader.hlsl, shader.isPixelShader,
                shader.specConstantsMask != 0, false, &shader.dxilError);
            shader.dxilOk = dxil != nullptr;
            if (dxil != nullptr)
                dxil->Release();

            IDxcBlob* spirv = dxcCompiler.compile(shader.hlsl, shader.isPixelShader,
                false, true, &shader.spirvError);
            shader.spirvOk = spirv != nullptr;
            if (spirv != nullptr)
                spirv->Release();

            size_t currentProgress = ++progress;
            if ((currentProgress % 100) == 0)
                fmt::println("Recompiling shaders... {}/{}", currentProgress, shaders.size());
        });

    size_t dxilFailures = 0;
    size_t spirvFailures = 0;
    size_t vertexShaders = 0;

    StringBuffer report;
    report.println("source_file\tcategory\tgroup\tentry_point\thash\tstage\tspec_constants\thlsl\tdxil\tspirv");

    StringBuffer errors;

    for (auto& shader : shaders)
    {
        if (!shader.isPixelShader)
            ++vertexShaders;
        if (!shader.dxilOk)
            ++dxilFailures;
        if (!shader.spirvOk)
            ++spirvFailures;

        report.println("{}\t{}\t{}\t{}\t{:016X}\t{}\t0x{:X}\t{}\t{}\t{}",
            shader.sourceFile,
            shader.category,
            shader.name.valid ? shader.name.group : std::string("?"),
            shader.name.valid ? shader.name.entryPoint : std::string("?"),
            shader.hash,
            shader.isPixelShader ? "ps" : "vs",
            shader.specConstantsMask,
            std::filesystem::relative(shader.outputPath, outputRoot).generic_string(),
            shader.dxilOk ? "ok" : "FAIL",
            shader.spirvOk ? "ok" : "FAIL");

        if (!shader.dxilOk || !shader.spirvOk)
        {
            errors.println("=== {} ({}) from {} ===",
                std::filesystem::relative(shader.outputPath, outputRoot).generic_string(),
                shader.isPixelShader ? "ps" : "vs", shader.sourceFile);
            if (!shader.dxilOk)
                errors.println("[DXIL]\n{}", shader.dxilError);
            if (!shader.spirvOk)
                errors.println("[SPIRV]\n{}", shader.spirvError);
            errors.println("");
        }
    }

    auto reportPath = (outputRoot / "shader_report.tsv").string();
    writeAllBytes(reportPath.c_str(), report.out.data(), report.out.size());

    auto errorsPath = (outputRoot / "shader_errors.txt").string();
    writeAllBytes(errorsPath.c_str(), errors.out.data(), errors.out.size());

    fmt::println("");
    fmt::println("Shaders             : {} ({} vertex, {} pixel)", shaders.size(), vertexShaders, shaders.size() - vertexShaders);
    fmt::println("HLSL emitted        : {}", shaders.size());
    fmt::println("DXIL compiled       : {} ({} failed)", shaders.size() - dxilFailures, dxilFailures);
    fmt::println("SPIR-V compiled     : {} ({} failed)", shaders.size() - spirvFailures, spirvFailures);
    fmt::println("Report              : {}", reportPath);
    fmt::println("Errors              : {}", errorsPath);

    return (dxilFailures == 0 && spirvFailures == 0) ? 0 : 2;
}

int main(int argc, char** argv)
{
    std::vector<const char*> positional;
    bool hlslMode = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--hlsl") == 0)
            hlslMode = true;
        else
            positional.push_back(argv[i]);
    }

#ifndef XENOS_RECOMP_INPUT
    if (positional.size() < 3)
    {
        printf("Usage: XenosRecomp [input path] [output path] [shader common header file path] [--hlsl]\n");
        printf("  input file  + output file : recompile a single shader container to HLSL\n");
        printf("  input dir   + output file : build a DXIL/SPIR-V shader cache .cpp\n");
        printf("  input dir   + output dir  + --hlsl : emit one .hlsl per shader and validate through DXC\n");
        return 0;
    }
#endif

    const char* input =
#ifdef XENOS_RECOMP_INPUT
        XENOS_RECOMP_INPUT
#else
        positional[0]
#endif
    ;

    const char* output =
#ifdef XENOS_RECOMP_OUTPUT
        XENOS_RECOMP_OUTPUT
#else
        positional[1]
#endif
        ;

    const char* includeInput =
#ifdef XENOS_RECOMP_INCLUDE_INPUT
        XENOS_RECOMP_INCLUDE_INPUT
#else
        positional[2]
#endif
        ;

    size_t includeSize = 0;
    auto includeData = readAllBytes(includeInput, includeSize);
    std::string_view include(reinterpret_cast<const char*>(includeData.get()), includeSize);

    if (hlslMode)
    {
        if (!std::filesystem::is_directory(input))
        {
            fmt::println("--hlsl requires the input path to be a directory.");
            return 1;
        }

        return recompileDirectoryToHlsl(input, output, include);
    }

    if (std::filesystem::is_directory(input))
    {
        std::vector<std::unique_ptr<uint8_t[]>> files;
        std::map<XXH64_hash_t, RecompiledShader> shaders;

        for (auto& file : std::filesystem::recursive_directory_iterator(input))
        {
            if (std::filesystem::is_directory(file))
            {
                continue;
            }

            size_t fileSize = 0;
            auto fileData = readAllBytes(file.path().string().c_str(), fileSize);
            bool foundAny = false;

            forEachShaderContainer(fileData.get(), fileSize, [&](size_t offset, size_t dataSize)
                {
                    XXH64_hash_t hash = XXH3_64bits(fileData.get() + offset, dataSize);
                    auto shader = shaders.try_emplace(hash);
                    if (shader.second)
                    {
                        shader.first->second.data = fileData.get() + offset;
                        foundAny = true;
                    }
                });

            if (foundAny)
                files.emplace_back(std::move(fileData));
        }

        std::atomic<uint32_t> progress = 0;

        std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](auto& hashShaderPair)
            {
                auto& shader = hashShaderPair.second;

                thread_local ShaderRecompiler recompiler;
                recompiler = {};
                recompiler.recompile(shader.data, include);

                shader.specConstantsMask = recompiler.specConstantsMask;

                thread_local DxcCompiler dxcCompiler;

#ifdef XENOS_RECOMP_DXIL
                shader.dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, recompiler.specConstantsMask != 0, false);
                assert(shader.dxil != nullptr);
                assert(*(reinterpret_cast<uint32_t *>(shader.dxil->GetBufferPointer()) + 1) != 0 && "DXIL was not signed properly!");
#endif

                IDxcBlob* spirv = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true);
                assert(spirv != nullptr);

                bool result = smolv::Encode(spirv->GetBufferPointer(), spirv->GetBufferSize(), shader.spirv, smolv::kEncodeFlagStripDebugInfo);
                assert(result);

                spirv->Release();

                size_t currentProgress = ++progress;
                if ((currentProgress % 10) == 0 || (currentProgress == shaders.size() - 1))
                    fmt::println("Recompiling shaders... {}%", currentProgress / float(shaders.size()) * 100.0f);
            });

        fmt::println("Creating shader cache...");

        StringBuffer f;
        f.println("#include \"shader_cache.h\"");
        f.println("ShaderCacheEntry g_shaderCacheEntries[] = {{");

        std::vector<uint8_t> dxil;
        std::vector<uint8_t> spirv;

        for (auto& [hash, shader] : shaders)
        {
            f.println("\t{{ 0x{:X}, {}, {}, {}, {}, {} }},",
                hash, dxil.size(), (shader.dxil != nullptr) ? shader.dxil->GetBufferSize() : 0, spirv.size(), shader.spirv.size(), shader.specConstantsMask);

            if (shader.dxil != nullptr)
            {
                dxil.insert(dxil.end(), reinterpret_cast<uint8_t *>(shader.dxil->GetBufferPointer()),
                    reinterpret_cast<uint8_t *>(shader.dxil->GetBufferPointer()) + shader.dxil->GetBufferSize());
            }

            spirv.insert(spirv.end(), shader.spirv.begin(), shader.spirv.end());
        }

        f.println("}};");

        fmt::println("Compressing DXIL cache...");

        int level = ZSTD_maxCLevel();

#ifdef XENOS_RECOMP_DXIL
        std::vector<uint8_t> dxilCompressed(ZSTD_compressBound(dxil.size()));
        dxilCompressed.resize(ZSTD_compress(dxilCompressed.data(), dxilCompressed.size(), dxil.data(), dxil.size(), level));

        f.print("const uint8_t g_compressedDxilCache[] = {{");

        for (auto data : dxilCompressed)
            f.print("{},", data);

        f.println("}};");
        f.println("const size_t g_dxilCacheCompressedSize = {};", dxilCompressed.size());
        f.println("const size_t g_dxilCacheDecompressedSize = {};", dxil.size());
#endif

        fmt::println("Compressing SPIRV cache...");

        std::vector<uint8_t> spirvCompressed(ZSTD_compressBound(spirv.size()));
        spirvCompressed.resize(ZSTD_compress(spirvCompressed.data(), spirvCompressed.size(), spirv.data(), spirv.size(), level));

        f.print("const uint8_t g_compressedSpirvCache[] = {{");

        for (auto data : spirvCompressed)
            f.print("{},", data);

        f.println("}};");

        f.println("const size_t g_spirvCacheCompressedSize = {};", spirvCompressed.size());
        f.println("const size_t g_spirvCacheDecompressedSize = {};", spirv.size());
        f.println("const size_t g_shaderCacheEntryCount = {};", shaders.size());

        writeAllBytes(output, f.out.data(), f.out.size());
    }
    else
    {
        ShaderRecompiler recompiler;
        size_t fileSize;
        recompiler.recompile(readAllBytes(input, fileSize).get(), include);
        writeAllBytes(output, recompiler.out.data(), recompiler.out.size());
    }

    return 0;
}

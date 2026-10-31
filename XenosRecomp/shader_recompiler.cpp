#include "shader_recompiler.h"
#include "shader_common.h"

static constexpr char SWIZZLES[] = 
{ 
    'x',
    'y', 
    'z', 
    'w', 
    '0', 
    '1',
    '_',
    '_'
};

// How each vertex input is declared in the generated HLSL. This is a contract
// with the host: the type here has to match the family of the vertex format the
// host binds, because the recompiled code assigns the input straight into an
// ALU register and the microcode's own constants do the rest.
//
// NORMAL/TANGENT/BINORMAL are uint4 on purpose - they carry a packed
// R11G11B10 that tfetchR11G11B10() unpacks in the shader.
//
// BLENDINDICES is float4, NOT uint4. The Xenos fetches blend indices through
// the D3DCOLOR declaration type, which the hardware NORMALISES: the shader sees
// index/255 and the microcode scales it back with its own constant (measured in
// Midnight Club: Los Angeles, c254.z = 765.006 = 255 * 3, three constant
// registers per bone). Declaring the input uint4 hands the ALU the raw byte
// instead, so that constant overshoots by 255x: bone 1 lands at register 765,
// the shader's own min(index, 191) clamp catches it, and every vertex not bound
// to bone 0 collapses onto the same matrix - the whole skinned mesh folds into a
// sliver. Keeping it float4 lets the host bind the normalised 8-bit format the
// declaration already asks for.
static constexpr const char* USAGE_TYPES[] =
{
    "float4", // POSITION
    "float4", // BLENDWEIGHT
    "float4", // BLENDINDICES
    "uint4", // NORMAL
    "float4", // PSIZE
    "float4", // TEXCOORD
    "uint4", // TANGENT
    "uint4", // BINORMAL
    "float4", // TESSFACTOR
    "float4", // POSITIONT
    "float4", // COLOR
    "float4", // FOG
    "float4", // DEPTH
    "float4", // SAMPLE
};

static constexpr const char* USAGE_VARIABLES[] =
{
    "Position",
    "BlendWeight",
    "BlendIndices",
    "Normal",
    "PointSize",
    "TexCoord",
    "Tangent",
    "Binormal",
    "TessFactor",
    "PositionT",
    "Color",
    "Fog",
    "Depth",
    "Sample"
};

static constexpr const char* USAGE_SEMANTICS[] =
{
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE"
};

struct DeclUsageLocation
{
    DeclUsage usage;
    uint32_t usageIndex;
    uint32_t location;
};

// NOTE: These are specialized Vulkan locations for Unleashed Recompiled. Change as necessary. Likely not going to work with other games.
static constexpr DeclUsageLocation USAGE_LOCATIONS[] =
{
    { DeclUsage::Position, 0, 0 },
    { DeclUsage::Normal, 0, 1 },
    { DeclUsage::Tangent, 0, 2 },
    { DeclUsage::Binormal, 0, 3 },
    { DeclUsage::TexCoord, 0, 4 },
    { DeclUsage::TexCoord, 1, 5 },
    { DeclUsage::TexCoord, 2, 6 },
    { DeclUsage::TexCoord, 3, 7 },
    { DeclUsage::Color, 0, 8 },
    { DeclUsage::BlendIndices, 0, 9 },
    { DeclUsage::BlendWeight, 0, 10 },
    { DeclUsage::Color, 1, 11 },
    { DeclUsage::TexCoord, 4, 12 },
    { DeclUsage::TexCoord, 5, 13 },
    { DeclUsage::TexCoord, 6, 14 },
    { DeclUsage::TexCoord, 7, 15 },
    // POSITION1 previously shared location 15 with TEXCOORD7, which is a SPIR-V
    // validation error for any vertex shader declaring both. Midnight Club uses
    // POSITION1 as a real vertex element in 8 shaders, so it gets its own slot.
    { DeclUsage::Position, 1, 16 },
};

static constexpr std::pair<DeclUsage, size_t> INTERPOLATORS[] =
{
    { DeclUsage::TexCoord, 0 },
    { DeclUsage::TexCoord, 1 },
    { DeclUsage::TexCoord, 2 },
    { DeclUsage::TexCoord, 3 },
    { DeclUsage::TexCoord, 4 },
    { DeclUsage::TexCoord, 5 },
    { DeclUsage::TexCoord, 6 },
    { DeclUsage::TexCoord, 7 },
    { DeclUsage::TexCoord, 8 },
    { DeclUsage::TexCoord, 9 },
    { DeclUsage::TexCoord, 10 },
    { DeclUsage::TexCoord, 11 },
    { DeclUsage::TexCoord, 12 },
    { DeclUsage::TexCoord, 13 },
    { DeclUsage::TexCoord, 14 },
    { DeclUsage::TexCoord, 15 },
    { DeclUsage::Color, 0 },
    { DeclUsage::Color, 1 },
    // Declared for every stage so the vertex output and pixel input signatures
    // stay identical across all shaders. Midnight Club exports POSITION1 as an
    // interpolator (one vertex/pixel pair); without it the generated code
    // referenced an oPosition1/iPosition1 that was never in the entry signature.
    { DeclUsage::Position, 1 }
};

static constexpr std::string_view TEXTURE_DIMENSIONS[] =
{
    "2D",
    "3D",
    "Cube"
};

// The Xenos texture fetch constant file has 32 slots shared by both stages.
// Xbox 360 D3D9 hands slots 0-15 to the pixel shader and 16-31 to the vertex
// shader, so a `sampler` declared at register s0 inside a vs_3_0 shader is
// addressed as fetch constant 16 by the tfetch instructions in its ISA. The CTAB
// only ever records the per-stage D3D register index, so the hardware slot has to
// be reconstructed here. Getting this wrong meant every vertex texture fetch in
// Midnight Club: Los Angeles resolved against a sampler that was never declared.
static constexpr uint32_t FETCH_CONSTANT_SLOT_COUNT = 32;
static constexpr uint32_t VERTEX_SHADER_FETCH_CONSTANT_BASE = 16;

static uint32_t getFetchConstantSlot(uint32_t registerIndex, bool isPixelShader)
{
    return isPixelShader ? registerIndex : (registerIndex + VERTEX_SHADER_FETCH_CONSTANT_BASE);
}

static FetchDestinationSwizzle getDestSwizzle(uint32_t dstSwizzle, uint32_t index)
{
    return FetchDestinationSwizzle((dstSwizzle >> (index * 3)) & 0x7);
}

void ShaderRecompiler::printDstSwizzle(uint32_t dstSwizzle, bool operand)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            out += SWIZZLES[operand ? uint32_t(swizzle) : i];
    }
}

void ShaderRecompiler::printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle == FetchDestinationSwizzle::Zero)
        {
            indent();
            println("r{}.{} = 0.0;", dstRegister, SWIZZLES[i]);
        }
        else if (swizzle == FetchDestinationSwizzle::One)
        {
            indent();
            println("r{}.{} = 1.0;", dstRegister, SWIZZLES[i]);
        }
    }
}

void ShaderRecompiler::recompile(const VertexFetchInstruction& instr, uint32_t address)
{
    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";

    auto findResult = vertexElements.find(address);
    assert(findResult != vertexElements.end());

    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(");
        break;

    case DeclUsage::TexCoord:
        print("tfetchTexcoord(g_SwappedTexcoords, ");
        break;
    }

    print("i{}{}", USAGE_VARIABLES[uint32_t(findResult->second.usage)], uint32_t(findResult->second.usageIndex));

    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        out += ')';
        break;

    case DeclUsage::TexCoord:
        print(", {})", uint32_t(findResult->second.usageIndex));
        break;
    }

    out += '.';
    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const TextureFetchInstruction& instr, bool bicubic)
{
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    case FetchOpcode::GetTextureWeights:
    case FetchOpcode::GetTextureComputedLod:
    case FetchOpcode::GetTextureGradients:
    case FetchOpcode::SetTextureLod:
    case FetchOpcode::SetTextureGradientsHorz:
    case FetchOpcode::SetTextureGradientsVert:
        break;
    default:
        return;
    }

    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predCondition ? "" : "!");

        indent();
        out += "{\n";
        ++indentation;
    }

    auto printSrcRegister = [&](size_t componentCount)
        {
            print("r{}.", instr.srcRegister);

            for (size_t i = 0; i < componentCount; i++)
                out += SWIZZLES[((instr.srcSwizzle >> (i * 2))) & 0x3];
        };

    auto closePredication = [&]()
        {
            if (instr.isPredicated)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        };

    // setTexLOD / setGradients only update the implicit fetch state registers;
    // they never write a destination register.
    switch (instr.opcode)
    {
    case FetchOpcode::SetTextureLod:
    {
        indent();
        out += "tfetchLod = ";
        printSrcRegister(1);
        out += ";\n";
        closePredication();
        return;
    }

    case FetchOpcode::SetTextureGradientsHorz:
    case FetchOpcode::SetTextureGradientsVert:
    {
        indent();
        print("tfetchGradient{} = ", instr.opcode == FetchOpcode::SetTextureGradientsHorz ? "H" : "V");
        printSrcRegister(3);
        out += ";\n";
        closePredication();
        return;
    }

    default:
        break;
    }

    // getGradients does not touch a texture at all.
    if (instr.opcode == FetchOpcode::GetTextureGradients)
    {
        indent();
        print("r{}.", instr.dstRegister);
        printDstSwizzle(instr.dstSwizzle, false);
        out += " = getGradients(";
        printSrcRegister(2);
        out += ").";
        printDstSwizzle(instr.dstSwizzle, true);
        out += ";\n";
        printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
        closePredication();
        return;
    }

    std::string constName;
    const char* constNamePtr = nullptr;
#ifdef UNLEASHED_RECOMP
    bool subtractFromOne = false;
#endif

    auto findResult = samplers.find(instr.constIndex);
    if (findResult != samplers.end())
    {
        constNamePtr = findResult->second;

    #ifdef UNLEASHED_RECOMP
        subtractFromOne = hasMtxPrevInvViewProjection && strcmp(constNamePtr, "sampZBuffer") == 0;
    #endif
    }
    else
    {
        constName = fmt::format("s{}", instr.constIndex);
        constNamePtr = constName.c_str();
    }

#ifdef UNLEASHED_RECOMP
    if (instr.constIndex == 0 && instr.dimension == TextureDimension::Texture2D)
    {
        indent();
        print("pixelCoord = getPixelCoord({}_Texture2DDescriptorIndex, ", constNamePtr);
        printSrcRegister(2);
        out += ");\n";
    }
#endif

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    // A Xenos tfetch only derives its LOD from gradients when the "use computed
    // LOD" bit is set and the stage can produce them: vertex shaders have no
    // derivatives, so unless explicit register gradients were supplied the fetch
    // is an explicit LOD fetch. This mirrors Xenia's translator.
    const bool useComputedLod = instr.useCompLod && (isPixelShader || instr.useRegGradients);
    const bool useRegisterGradients = useComputedLod && instr.useRegGradients && usesTextureGradientRegisters;
    const bool useRegisterLod = instr.useRegLod && usesTextureLodRegister;
    const float instructionLodBias = instr.lodBias * (1.0f / 16.0f);

    // Register LOD and instruction LOD bias both add to the sampled LOD.
    std::string lodExpression;
    if (useRegisterLod)
        lodExpression = "tfetchLod";
    if (instructionLodBias != 0.0f)
        lodExpression += (lodExpression.empty() ? fmt::format("{}", instructionLodBias) : fmt::format(" + {}", instructionLodBias));

    const bool isTextureFetch = instr.opcode == FetchOpcode::TextureFetch;

    // Suffix selects the sampling form: gradients, explicit LOD, LOD bias, or the
    // plain implicit-derivative Sample that the vast majority of fetches use.
    const char* fetchSuffix = "";
    if (isTextureFetch)
    {
        if (useRegisterGradients)
            fetchSuffix = "Grad";
        else if (!useComputedLod)
            fetchSuffix = "Lod";
        else if (!lodExpression.empty())
            fetchSuffix = "Bias";
    }

    out += " = ";
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    {
    #ifdef UNLEASHED_RECOMP
        if (subtractFromOne)
            out += "1.0 - ";
    #endif

        out += "tfetch";
        break;
    }
    case FetchOpcode::GetTextureWeights:
    {
        out += "getWeights";
        break;
    }
    case FetchOpcode::GetTextureComputedLod:
    {
        out += "getCompTexLod";
        break;
    }
    }

    std::string_view dimension;
    uint32_t componentCount = 0;

    switch (instr.dimension)
    {
    case TextureDimension::Texture1D:
        dimension = "1D";
        componentCount = 1;
        break;
    case TextureDimension::Texture2D:
        dimension = "2D";
        componentCount = 2;
        break;
    case TextureDimension::Texture3D:
        dimension = "3D";
        componentCount = 3;
        break;
    case TextureDimension::TextureCube:
        dimension = "Cube";
        componentCount = 3;
        break;
    }

    out += dimension;

#ifdef UNLEASHED_RECOMP
    if (bicubic)
        out += "Bicubic";
#endif

    out += fetchSuffix;

    print("({0}_Texture{1}DescriptorIndex, {0}_SamplerDescriptorIndex, ", constNamePtr, dimension);
    printSrcRegister(componentCount);

    // getCompTexLod takes no pixel offset; every other form keeps the existing
    // argument list.
    if (instr.opcode != FetchOpcode::GetTextureComputedLod)
    {
        switch (instr.dimension)
        {
        case TextureDimension::Texture2D:
            print(", float2({}, {})", instr.offsetX * 0.5f, instr.offsetY * 0.5f);
            break;
        case TextureDimension::TextureCube:
            out += ", cubeMapData";
            break;
        }
    }

    if (isTextureFetch)
    {
        if (useRegisterGradients)
        {
            // Xenia folds the register LOD and instruction bias into a gradient
            // scale of exp2(lod), which is exactly what biasing the LOD does.
            const char* swizzle = (componentCount == 3) ? ".xyz" : ".xy";
            if (lodExpression.empty())
                print(", tfetchGradientH{0}, tfetchGradientV{0}", swizzle);
            else
                print(", tfetchGradientH{0} * exp2({1}), tfetchGradientV{0} * exp2({1})", swizzle, lodExpression);
        }
        else if (!useComputedLod)
        {
            print(", {}", lodExpression.empty() ? "0.0" : lodExpression);
        }
        else if (!lodExpression.empty())
        {
            print(", {}", lodExpression);
        }
    }

    out += ").";

    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

void ShaderRecompiler::recompile(const AluInstruction& instr)
{
    if (instr.isPredicated)
    {
        indent();
        println("if ({}p0)", instr.predicateCondition ? "" : "!");

        indent(); 
        out += "{\n";
        ++indentation;
    }

    enum
    {
        VECTOR_0,
        VECTOR_1,
        VECTOR_2,
        SCALAR_0,
        SCALAR_1,
        SCALAR_CONSTANT_0,
        SCALAR_CONSTANT_1
    };

    // Set below when the scalar half of this instruction reads a GPR component
    // that the vector half overwrites; the scalar operand then comes out of the
    // psSrc snapshot instead of the clobbered register.
    int scalarSnapshotRegister = -1;

    auto op = [&](size_t operand)
        {
            size_t reg = 0;
            size_t swizzle = 0;
            bool select = true;
            bool negate = false;
            bool abs = false;

            switch (operand)
            {
            case SCALAR_CONSTANT_0:
                reg = instr.src3Register;
                swizzle = instr.src3Swizzle;
                select = false;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            case SCALAR_CONSTANT_1:
                reg = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
                swizzle = instr.src3Swizzle;
                select = true;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            default:
                switch (operand)
                {
                case VECTOR_0:
                    reg = instr.src1Register;
                    swizzle = instr.src1Swizzle;
                    select = instr.src1Select;
                    negate = instr.src1Negate;
                    break;
                case VECTOR_1:
                    reg = instr.src2Register;
                    swizzle = instr.src2Swizzle;
                    select = instr.src2Select;
                    negate = instr.src2Negate;
                    break;
                case VECTOR_2:
                case SCALAR_0:
                case SCALAR_1:
                    reg = instr.src3Register;
                    swizzle = instr.src3Swizzle;
                    select = instr.src3Select;
                    negate = instr.src3Negate;
                    break;
                }

                if (select)
                {
                    abs = (reg & 0x80) != 0;
                    reg &= 0x3F;
                }
                else
                {
                    abs = instr.absConstants;
                }

                break;
            }

            std::string regFormatted;

            if (select)
            {
                const bool scalarOperand = operand == SCALAR_0 || operand == SCALAR_1 || operand == SCALAR_CONSTANT_1;
                if (scalarOperand && int(reg) == scalarSnapshotRegister)
                    regFormatted = "psSrc";
                else
                    regFormatted = fmt::format("r{}", reg);
            }
            else
            {
                auto findResult = float4Constants.find(reg);
                if (findResult != float4Constants.end())
                {
                    const char* constantName = reinterpret_cast<const char*>(constantTableData + findResult->second->name);
                    if (findResult->second->registerCount > 1)
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (hasMtxProjection && strcmp(constantName, "g_MtxProjection") == 0)
                        {
                            regFormatted = fmt::format("(iterationIndex == 0 ? mtxProjectionReverseZ[{0}] : mtxProjection[{0}])",
                                reg - findResult->second->registerIndex);
                        }
                        else
                    #endif
                        {
                            // WHICH of the two relative bits belongs to THIS
                            // operand. The encoding has const0Relative and
                            // const1Relative, and the slot is decided by how many
                            // CONSTANT sources the instruction has -- slot 0 is the
                            // first constant operand, slot 1 the second -- NOT by the
                            // source index. `select` is false for a constant source
                            // and true for a temp register, so src1 being a constant
                            // is what pushes a src2 constant into slot 1.
                            //
                            // Reading const0Relative for everything is what made
                            // xRimMain__VS_Common emit
                            //   oTexCoord4.xy = tintColors(0 + a0).xy;                 // max: src1 == src2, one constant  -> slot 0
                            //   oTexCoord4.zw = select(..., tintColors(0).zz, ...);    // cnd: src1 is c255, ALSO a constant -> tintColors is slot 1
                            // and the second line lost its a0. The wheel picks its
                            // paint slot with a0 = trunc(TEXCOORD0.z), so losing it
                            // takes the BLUE channel from tint slot 0 while red and
                            // green come from the part's own slot: a black rim
                            // rendered (0.0066, 0.0066, 0.9574), pure blue.
                            //
                            // Do NOT simplify this to "src2 always uses
                            // const1Relative". Tried 2026-09-07: 54 reads gained a
                            // relative index and 2234 LOST one -- every
                            // `gCarMtxBuffer(N + a0)` and `gLightMatrixArr(N + a0)`
                            // collapsed to index 0, because there src1 is a TEMP and
                            // the constant at src2 is the FIRST constant, slot 0.
                            //
                            // src3 (the scalar constant sources) is left on slot 0,
                            // which is what this always did: no case has been measured
                            // for it and guessing would change shaders nothing has
                            // shown to be wrong.
                            const bool secondConstant = (operand == VECTOR_1) && !instr.src1Select;
                            const bool relative = secondConstant ? instr.const1Relative
                                                                 : instr.const0Relative;
                            regFormatted = fmt::format("{}({}{})", constantName,
                                reg - findResult->second->registerIndex, relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "");
                        }
                    }
                    else
                    {
                        assert(!instr.const0Relative && !instr.const1Relative);
                        regFormatted = constantName;
                    }
                }
                else
                {
                    assert(!instr.const0Relative && !instr.const1Relative);
                    regFormatted = fmt::format("c{}", reg);
                }
            }

            std::string result;

            if (negate)
                result += '-';

            if (abs)
                result += "abs(";

            result += regFormatted;
            result += '.';

            switch (operand)
            {
            case VECTOR_0:
            case VECTOR_1:
            case VECTOR_2:
            {
                uint32_t mask;

                switch (instr.vectorOpcode)
                {
                case AluVectorOpcode::Dp2Add:
                    mask = (operand == VECTOR_2) ? 0b1 : 0b11;
                    break;

                case AluVectorOpcode::Dp3:
                    mask = 0b111;
                    break;

                case AluVectorOpcode::Dp4:
                case AluVectorOpcode::Max4:
                    mask = 0b1111;
                    break;

                default:
                    mask = instr.vectorWriteMask != 0 ? instr.vectorWriteMask : 0b1;
                    break;
                }

                for (size_t i = 0; i < 4; i++)
                {
                    if ((mask >> i) & 0x1)
                        result += SWIZZLES[((swizzle >> (i * 2)) + i) & 0x3];
                }

                break;
            }

            case SCALAR_0:
            case SCALAR_CONSTANT_0:
                result += SWIZZLES[((swizzle >> 6) + 3) & 0x3];
                break;

            case SCALAR_1:
            case SCALAR_CONSTANT_1:
                result += SWIZZLES[swizzle & 0x3];
                break;
            }

            if (abs)
                result += ")";

            return result;
        };

    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::KillEq:
        indent();
        println("clip(any({} == {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGt:
        indent();
        println("clip(any({} > {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGe:
        indent();
        println("clip(any({} >= {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillNe:
        indent();
        println("clip(any({} != {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    }

    bool closeIfBracket = false;

    std::string_view exportRegister;
    if (instr.exportData)
    {
        if (isPixelShader)
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::PSColor0:
                exportRegister = "oC0";
                break;        
            case ExportRegister::PSColor1:
                exportRegister = "oC1";
                break;        
            case ExportRegister::PSColor2:
                exportRegister = "oC2";
                break;            
            case ExportRegister::PSColor3:
                exportRegister = "oC3";
                break;           
            case ExportRegister::PSDepth:
                exportRegister = "oDepth";
                break;
            }
        }
        else
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::VSPosition:
                exportRegister = "oPos";

            #ifdef UNLEASHED_RECOMP
                if (hasMtxProjection)
                {
                    indent();
                    out += "if ((g_SpecConstants() & SPEC_CONSTANT_REVERSE_Z) == 0 || iterationIndex == 0)\n";
                    indent();
                    out += "{\n";
                    ++indentation;

                    closeIfBracket = true;
                }
            #endif

                break;

            default:
            {
                auto findResult = interpolators.find(instr.vectorDest);
                assert(findResult != interpolators.end());
                exportRegister = findResult->second;
                break;
            }
            }
        }
    }

    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        indent();
        print("p0 = {} == 0.0 && {} ", op(VECTOR_0), op(VECTOR_1));

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::SetpEqPush:
            out += "==";
            break;
        case AluVectorOpcode::SetpNePush:
            out += "!=";
            break;
        case AluVectorOpcode::SetpGtPush:
            out += ">";
            break;
        case AluVectorOpcode::SetpGePush:
            out += ">=";
            break;
        }

        out += " 0.0;\n";
    }
    else if (instr.vectorOpcode >= AluVectorOpcode::MaxA)
    {
        indent();
        println("a0 = (int)clamp(floor(({}).w + 0.5), -256.0, 255.0);", op(VECTOR_0));
    }

    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;

    // A Xenos ALU instruction issues its vector and scalar halves in the same
    // cycle: both read the register file as it stood BEFORE the instruction, and
    // only then are the results written back. Emitting the two halves as
    // sequential statements breaks that whenever the scalar half reads a GPR
    // component the vector half overwrites -- the compiler co-issues exactly
    // those pairs, because the hardware makes them free. Snapshot the register
    // before the vector write and read the scalar operand out of the snapshot.
    if (vectorWriteMask != 0 && exportRegister.empty() && instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        const bool constantForm = instr.scalarOpcode >= AluScalarOpcode::Mulsc0 &&
                                  instr.scalarOpcode <= AluScalarOpcode::Subsc1;

        bool readsSecondOperand;
        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds:
        case AluScalarOpcode::Muls:
        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::Mins:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
        case AluScalarOpcode::Subs:
            readsSecondOperand = true;
            break;
        default:
            readsSecondOperand = false;
            break;
        }

        uint32_t scalarRegister = 0;
        uint32_t componentMask = 0;

        if (constantForm)
        {
            // SCALAR_CONSTANT_0 comes from the constant file; only the second
            // operand of the *sc forms is a GPR, and its index is encoded here.
            scalarRegister = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
            componentMask = 1u << (instr.src3Swizzle & 0x3);
        }
        else if (instr.src3Select && instr.scalarOpcode != AluScalarOpcode::SetpClr)
        {
            scalarRegister = instr.src3Register;
            componentMask = 1u << (((instr.src3Swizzle >> 6) + 3) & 0x3);
            if (readsSecondOperand)
                componentMask |= 1u << (instr.src3Swizzle & 0x3);
        }

        if (componentMask != 0 && (scalarRegister & 0x3F) == instr.vectorDest && (componentMask & vectorWriteMask) != 0)
        {
            scalarSnapshotRegister = int(instr.vectorDest);
            indent();
            println("psSrc = r{};", instr.vectorDest);
        }
    }

    if (vectorWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.vectorDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ";

        if (instr.vectorSaturate)
            out += "saturate(";

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add:
            print("{} + {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Mul:
            print("{} * {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA:
            print("max({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Min:
            print("min({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Seq:
            print("{} == {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sgt:
            print("{} > {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sge:
            print("{} >= {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sne:
            print("{} != {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Frc:
            print("frac({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Trunc:
            print("trunc({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Floor:
            print("floor({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Mad:
            print("{} * {} + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndEq:
            print("select({} == 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGe:
            print("select({} >= 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGt:
            print("select({} > 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Dp4:
        case AluVectorOpcode::Dp3:
            print("dot({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dp2Add:
            print("dot({}, {}) + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Cube:
            print("cube(r{}, cubeMapData)", instr.src1Register);
            break;

        case AluVectorOpcode::Max4:
            print("max4({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
            print("p0 ? 0.0 : {} + 1.0", op(VECTOR_0));
            break;

        case AluVectorOpcode::KillEq:
            print("any({} == {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGt:
            print("any({} > {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGe:
            print("any({} >= {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillNe:
            print("any({} != {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dst:
            print("dst({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;
        }

        if (instr.vectorSaturate)
            out += ')';

        out += ";\n";
    }

    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        if (instr.scalarOpcode >= AluScalarOpcode::SetpEq && instr.scalarOpcode <= AluScalarOpcode::SetpRstr)
        {
            indent();
            out += "p0 = ";

            switch (instr.scalarOpcode)
            {
            case AluScalarOpcode::SetpEq:
                print("{} == 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpNe:
                print("{} != 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGt:
                print("{} > 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGe:
                print("{} >= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpInv:
                print("{} == 1.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpPop:
                print("{} - 1.0 <= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpClr:
                out += "false";
                break;

            case AluScalarOpcode::SetpRstr:
                print("{} == 0.0", op(SCALAR_0));
                break;
            }

            out += ";\n";
        }

        indent();
        out += "ps = ";
        if (instr.scalarSaturate)
            out += "saturate(";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds:
            print("{} + {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::AddsPrev:
            print("{} + ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Muls:
            print("{} * {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::MulsPrev:
        case AluScalarOpcode::MulsPrev2:
            print("{} * ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
            print("max({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Mins:
            print("min({}, {})", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::Seqs:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sgts:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sges:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Snes:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Frcs:
            print("frac({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Truncs:
            print("trunc({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Floors:
            print("floor({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Exp:
            print("exp2({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Logc:
        case AluScalarOpcode::Log:
            print("clamp(log2({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rcpc:
        case AluScalarOpcode::Rcpf:
        case AluScalarOpcode::Rcp:
            print("clamp(rcp({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rsqc:
        case AluScalarOpcode::Rsqf:
        case AluScalarOpcode::Rsq:
            print("clamp(rsqrt({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Subs:
            print("{} - {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::SubsPrev:
            print("{} - ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpEq:
        case AluScalarOpcode::SetpNe:
        case AluScalarOpcode::SetpGt:
        case AluScalarOpcode::SetpGe:
            out += "p0 ? 0.0 : 1.0";
            break;

        case AluScalarOpcode::SetpInv:
            print("{0} == 0.0 ? 1.0 : {0}", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpPop:
            print("p0 ? 0.0 : ({} - 1.0)", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpClr:
            out += "FLT_MAX";
            break;

        case AluScalarOpcode::SetpRstr:
            print("p0 ? 0.0 : {}", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsEq:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGt:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGe:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsNe:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsOne:
            print("{} == 1.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sqrt:
            print("sqrt({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1:
            print("{} * {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1:
            print("{} + {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1:
            print("{} - {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Sin:
            print("sin({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Cos:
            print("cos({})", op(SCALAR_0));
            break;
        }

        if (instr.scalarSaturate)
            out += ')';

        out += ";\n";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::MaxAs:
            indent();
            println("a0 = (int)clamp(floor({} + 0.5), -256.0, 255.0);", op(SCALAR_0));
            break;     
        case AluScalarOpcode::MaxAsf:
            indent();
            println("a0 = (int)clamp(floor({}), -256.0, 255.0);", op(SCALAR_0));
            break;
        }
    }

    uint32_t scalarWriteMask = instr.scalarWriteMask;
    if (instr.exportData)
        scalarWriteMask &= ~instr.vectorWriteMask;

    if (scalarWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.scalarDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ps;\n";
    }

    if (instr.exportData)
    {
        uint32_t zeroMask = instr.scalarDestRelative ? (0b1111 & ~(instr.vectorWriteMask | instr.scalarWriteMask)) : 0;
        uint32_t oneMask = instr.vectorWriteMask & instr.scalarWriteMask;

        for (size_t i = 0; i < 4; i++)
        {
            uint32_t mask = 1 << i;
            if (zeroMask & mask)
            {
                indent();
                println("{}.{} = 0.0;", exportRegister, SWIZZLES[i]);
            }
            else if (oneMask & mask)
            {
                indent();
                println("{}.{} = 1.0;", exportRegister, SWIZZLES[i]);
            }
        }
    }

    if (instr.scalarOpcode >= AluScalarOpcode::KillsEq && instr.scalarOpcode <= AluScalarOpcode::KillsOne)
    {
        indent();
        out += "clip(ps != 0.0 ? -1 : 1);\n";
    }

    if (closeIfBracket)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    if (instr.isPredicated)
    {
        --indentation;
        indent();
        out += "}\n";
    }
}

// The Xenos control flow program is a list of instructions addressed by index,
// with jumps between them. HLSL has no goto, so the jumps become statements:
//
//   - a conditional jump forward over a block of instructions is
//         if (the jump is not taken) { block }
//   - when that block ends in an unconditional jump forward over a second
//     block, the pair is
//         if (the jump is not taken) { block } else { second block }
//
// as long as every block nests inside the blocks and loops around it. That is
// the structured form. A program with any other jump -- backward, into another
// block, an unconditional one that does not end a then-block -- becomes a
// switch over the control flow index inside an endless loop instead (the pc
// machine). Both are correct for DXC, but vkd3d-proton's dxil-spirv rejects the
// pc machine's DXIL as an infinite loop and the driver crashes on the result,
// so the structured form is the one to produce wherever it exists. Measured
// across Midnight Club's 2333 shaders: 77 conditional jumps, all predicated
// and forward, and 50 unconditional ones, all forward, each ending the
// then-block of an if-else (41 *MultiLightInterior pixel shaders and 9 particle
// vertex shaders); no backward jump, no bool-constant jump, no call. The
// structured form of the 41 pixel shaders compiles to the same DXIL, byte for
// byte, as the if/else rewrite they were given by hand for dxil-spirv.
struct ControlFlowPlan
{
    ControlFlowForm form = ControlFlowForm::Flat;

    enum class BlockAction : uint8_t
    {
        Close,             // }
        CloseThenOpenElse  // } else {
    };

    // Per control flow index, plus one entry for the end of the program: what
    // to emit before that instruction, innermost block first.
    std::vector<std::vector<BlockAction>> blockActionsBefore;

    // Per control flow index: the unconditional jump that ends a then-block. It
    // is the "else" of the statement, so nothing else is emitted for it.
    std::vector<bool> isElseJump;

    // Set when the program uses control flow this translator does not
    // implement in either form.
    std::string error;
};

static ControlFlowPlan planControlFlow(const std::vector<ControlFlowInstruction>& controlFlow)
{
    const uint32_t size = uint32_t(controlFlow.size());

    ControlFlowPlan plan;
    plan.blockActionsBefore.resize(size + 1);
    plan.isElseJump.resize(size);

    enum class BlockKind
    {
        If,
        Else,
        Loop
    };

    struct Block
    {
        BlockKind kind;
        uint32_t end;     // index the block closes at; for a loop, its LoopEnd
        uint32_t elseEnd; // an if that has an else-block: where that one closes
    };

    std::vector<Block> open;
    bool hasJumps = false;
    bool structured = true;

    // A block opened inside the first `depth` open blocks must close no later
    // than the innermost of them.
    auto enclosingEnd = [&](size_t depth)
    {
        return depth == 0 ? size : open[depth - 1].end;
    };

    for (uint32_t index = 0; index <= size && structured; index++)
    {
        while (!open.empty() && open.back().kind != BlockKind::Loop && open.back().end == index)
        {
            Block& block = open.back();
            if (block.kind == BlockKind::If && block.elseEnd != 0)
            {
                plan.blockActionsBefore[index].push_back(ControlFlowPlan::BlockAction::CloseThenOpenElse);
                block.kind = BlockKind::Else;
                block.end = block.elseEnd;
                block.elseEnd = 0;
            }
            else
            {
                plan.blockActionsBefore[index].push_back(ControlFlowPlan::BlockAction::Close);
                open.pop_back();
            }
        }

        // A block that should close here but is not the innermost one overlaps
        // another block instead of nesting in it.
        for (const Block& block : open)
        {
            if (block.kind != BlockKind::Loop && block.end <= index)
                structured = false;
        }

        if (index == size || !structured)
            break;

        const ControlFlowInstruction& cfInstr = controlFlow[index];
        switch (cfInstr.opcode)
        {
        case ControlFlowOpcode::CondJmp:
        {
            hasJumps = true;
            const uint32_t target = cfInstr.condJmp.address;

            if (cfInstr.condJmp.direction || target <= index || target > size)
            {
                structured = false;
            }
            else if (cfInstr.condJmp.isUnconditional)
            {
                // Only as the last instruction of an if's then-block, jumping over
                // an else-block that still nests in whatever encloses the if.
                if (!open.empty() && open.back().kind == BlockKind::If && open.back().elseEnd == 0 &&
                    open.back().end == index + 1 && target <= enclosingEnd(open.size() - 1))
                {
                    open.back().elseEnd = target;
                    plan.isElseJump[index] = true;
                }
                else
                {
                    structured = false;
                }
            }
            else if (target <= enclosingEnd(open.size()))
            {
                open.push_back({ BlockKind::If, target, 0 });
            }
            else
            {
                structured = false;
            }
            break;
        }

        case ControlFlowOpcode::LoopStart:
        {
            // The body runs from the next instruction to the LoopEnd, which sits
            // right before the LoopStart's exit address and jumps back to the
            // instruction after the LoopStart.
            const uint32_t exit = cfInstr.loopStart.address;
            if (exit < index + 2 || exit > enclosingEnd(open.size()) ||
                controlFlow[exit - 1].opcode != ControlFlowOpcode::LoopEnd ||
                controlFlow[exit - 1].loopEnd.address != index + 1 ||
                controlFlow[exit - 1].loopEnd.loopId != cfInstr.loopStart.loopId)
            {
                structured = false;
            }
            else
            {
                open.push_back({ BlockKind::Loop, exit - 1, 0 });
            }
            break;
        }

        case ControlFlowOpcode::LoopEnd:
            if (open.empty() || open.back().kind != BlockKind::Loop || open.back().end != index)
                structured = false;
            else
                open.pop_back();
            break;

        case ControlFlowOpcode::CondCall:
        case ControlFlowOpcode::Return:
            plan.error = fmt::format("control flow call/return at {}", index);
            break;

        default:
            break;
        }
    }

    if (!open.empty())
        structured = false;

    if (!structured)
    {
        plan.form = ControlFlowForm::PcMachine;
        plan.blockActionsBefore.assign(size + 1, {});
        plan.isElseJump.assign(size, false);
    }
    else
    {
        plan.form = hasJumps ? ControlFlowForm::Structured : ControlFlowForm::Flat;
    }

    return plan;
}

void ShaderRecompiler::recompile(const uint8_t* shaderData, const std::string_view& include)
{
    const auto shaderContainer = reinterpret_cast<const ShaderContainer*>(shaderData);

    assert((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100);
    assert(shaderContainer->constantTableOffset != NULL);

    out += include;
    out += '\n';

    isPixelShader = (shaderContainer->flags & 0x1) == 0;

    const auto constantTableContainer = reinterpret_cast<const ConstantTableContainer*>(shaderData + shaderContainer->constantTableOffset);
    constantTableData = reinterpret_cast<const uint8_t*>(&constantTableContainer->constantTable);

    out += "#ifdef __spirv__\n\n";

#ifdef UNLEASHED_RECOMP
    bool isMetaInstancer = false;
    bool hasIndexCount = false;
#endif

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

    #ifdef UNLEASHED_RECOMP
        if (!isPixelShader)
        {
            if (strcmp(constantName, "g_MtxProjection") == 0)
                hasMtxProjection = true;
            else if (strcmp(constantName, "g_InstanceTypes") == 0)
                isMetaInstancer = true;
            else if (strcmp(constantName, "g_IndexCount") == 0)
                hasIndexCount = true;
        }
        else
        {
            if (strcmp(constantName, "g_MtxPrevInvViewProjection") == 0)
                hasMtxPrevInvViewProjection = true;
        }
    #endif

        switch (constantInfo->registerSet)
        {
        case RegisterSet::Float4:
        {
            const char* shaderName = isPixelShader ? "Pixel" : "Vertex";

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;

                println("#define {}(INDEX) select((INDEX) < {}, vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + ({} + min(INDEX, {})) * 16, 0x10), 0.0)",
                    constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1);
            }
            else
            {
                println("#define {} vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + {}, 0x10)",
                    constantName, shaderName, constantInfo->registerIndex * 16);
            }
            
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
                float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);

            break;
        }

        case RegisterSet::Sampler:
        {
            uint32_t fetchSlot = getFetchConstantSlot(constantInfo->registerIndex, isPixelShader);

            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("#define {}_Texture{}DescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                    constantName, TEXTURE_DIMENSIONS[j], j * FETCH_CONSTANT_SLOT_COUNT * 4 + fetchSlot * 4);
            }

            println("#define {}_SamplerDescriptorIndex vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {})",
                constantName, std::size(TEXTURE_DIMENSIONS) * FETCH_CONSTANT_SLOT_COUNT * 4 + fetchSlot * 4);

            samplers.emplace(fetchSlot, constantName);
            break;
        }

        }
    }

    out += "\n#else\n\n";

    const uint32_t registerFileSize = isPixelShader ? 224 : 256;

    // The D3D9 effect framework happily lets two parameters share float4
    // registers when they can never be live at the same time, e.g. Midnight
    // Club's gBoneMtx[] sitting on top of gCarConstantBuffer[]. HLSL rejects
    // overlapping packoffset declarations outright, so those shaders declare the
    // whole constant register file as one array and alias each parameter onto it
    // with a macro instead. The generated code is identical either way; only the
    // declaration form changes. Non-overlapping shaders keep the packoffset form
    // so the common case stays untouched.
    bool hasOverlappingConstants = false;
    {
        std::vector<bool> occupied(registerFileSize, false);
        for (uint32_t i = 0; i < constantTableContainer->constantTable.constants && !hasOverlappingConstants; i++)
        {
            const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
                constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

            if (constantInfo->registerSet != RegisterSet::Float4)
                continue;

            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
            {
                uint32_t reg = constantInfo->registerIndex + j;
                if (reg >= registerFileSize)
                    break;

                if (occupied[reg])
                {
                    hasOverlappingConstants = true;
                    break;
                }
                occupied[reg] = true;
            }
        }
    }

    // A constant reached through a0/aL indexes the flat Xenos register file, so the
    // read can land outside the array the constant table declares for that one
    // parameter. Midnight Club's gCarMtxBuffer[60] sits at c64 and is read as
    // gCarMtxBuffer(3 * boneIndex); the accessor allows up to 191, which is the
    // distance to the end of the file, not the 60 the array has.
    //
    // In the packoffset form that read leaves the array. HLSL calls it undefined.
    // DXC lowers it to a raw cbufferLoadLegacy at the computed register rather
    // than clamping, so the right value only comes back while the bound view is
    // larger than the cbuffer the shader declares -- the paint shader declares
    // 3328 bytes and can read to byte 4080. A driver that bounds-checks against
    // the declared size returns zero instead, and a zeroed transform row is a
    // mesh collapsed to a point. The register-file form has no such dependency,
    // and it is what the SPIR-V path above already does.
    //
    // Measured across Midnight Club's 2333 shaders: 491 index a constant past its
    // declared size (gBoneMtx[144] clamped to 191 in 169 of them, gCarMtxBuffer[60]
    // to 191 in 107), and only 7 were taking the register-file path.
    bool hasRelativeConstantAddressing = false;
    {
        const auto shaderHeader = reinterpret_cast<const Shader*>(shaderData + shaderContainer->shaderOffset);
        const be<uint32_t>* code = reinterpret_cast<const be<uint32_t>*>(
            shaderData + shaderContainer->virtualSize + shaderHeader->physicalOffset);

        union
        {
            ControlFlowInstruction controlFlow[2];
            struct
            {
                uint32_t code0;
                uint32_t code1;
                uint32_t code2;
                uint32_t code3;
            };
        };

        auto controlFlowCode = code;
        uint32_t instrAddress = 0;
        uint32_t instrSize = shaderHeader->size;

        while (instrAddress < instrSize && !hasRelativeConstantAddressing)
        {
            code0 = controlFlowCode[0];
            code1 = controlFlowCode[1] & 0xFFFF;
            code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
            code3 = controlFlowCode[2] >> 16;

            for (auto& cfInstr : controlFlow)
            {
                uint32_t address = 0;
                uint32_t count = 0;

                switch (cfInstr.opcode)
                {
                case ControlFlowOpcode::Exec:
                case ControlFlowOpcode::ExecEnd:
                    address = cfInstr.exec.address;
                    count = cfInstr.exec.count;
                    break;

                case ControlFlowOpcode::CondExec:
                case ControlFlowOpcode::CondExecEnd:
                case ControlFlowOpcode::CondExecPredClean:
                case ControlFlowOpcode::CondExecPredCleanEnd:
                    address = cfInstr.condExec.address;
                    count = cfInstr.condExec.count;
                    break;

                case ControlFlowOpcode::CondExecPred:
                case ControlFlowOpcode::CondExecPredEnd:
                    address = cfInstr.condExecPred.address;
                    count = cfInstr.condExecPred.count;
                    break;

                default:
                    continue;
                }

                instrSize = std::min<uint32_t>(instrSize, address * 12);

                uint32_t sequence = cfInstr.exec.sequence;
                for (uint32_t i = 0; i < count && !hasRelativeConstantAddressing; i++, sequence >>= 2)
                {
                    // Bit 0 of each pair marks a fetch instruction; ALU otherwise.
                    if ((sequence & 0x1) != 0)
                        continue;

                    union
                    {
                        AluInstruction alu;
                        struct
                        {
                            uint32_t alu0;
                            uint32_t alu1;
                            uint32_t alu2;
                        };
                    };

                    const be<uint32_t>* aluCode = code + (address + i) * 3;
                    alu0 = aluCode[0];
                    alu1 = aluCode[1];
                    alu2 = aluCode[2];

                    if (alu.const0Relative || alu.const1Relative)
                        hasRelativeConstantAddressing = true;
                }

                if (hasRelativeConstantAddressing)
                    break;
            }

            controlFlowCode += 3;
            instrAddress += 12;
        }
    }

    // Either reason puts every parameter on the shared register file instead of
    // its own packoffset array. The generated body is identical; only the
    // declaration form changes.
    const bool useConstantRegisterFile = hasOverlappingConstants || hasRelativeConstantAddressing;

    const char* shaderStageName = isPixelShader ? "Pixel" : "Vertex";

    println("cbuffer {}ShaderConstants : register(b{}, space4)", shaderStageName, isPixelShader ? 1 : 0);
    out += "{\n";

    if (useConstantRegisterFile)
        println("\tfloat4 g_{}ShaderConstantFile[{}] : packoffset(c0);", shaderStageName, registerFileSize);

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
            uint32_t tailCount = registerFileSize - constantInfo->registerIndex;

            if (useConstantRegisterFile)
            {
                if (constantInfo->registerCount > 1)
                {
                    println("#define {0}(INDEX) select((INDEX) < {1}, g_{2}ShaderConstantFile[{3} + min(INDEX, {4})], 0.0)",
                        constantName, tailCount, shaderStageName, constantInfo->registerIndex.get(), tailCount - 1);
                }
                else
                {
                    println("#define {} g_{}ShaderConstantFile[{}]",
                        constantName, shaderStageName, constantInfo->registerIndex.get());
                }

                continue;
            }

            print("\tfloat4 {}", constantName);

            if (constantInfo->registerCount > 1)
                print("[{}]", constantInfo->registerCount.get());

            println(" : packoffset(c{});", constantInfo->registerIndex.get());

            if (constantInfo->registerCount > 1)
                println("#define {0}(INDEX) select((INDEX) < {1}, {0}[min(INDEX, {2})], 0.0)", constantName, tailCount, tailCount - 1);
        }
    }

    out += "};\n\n";

    out += "cbuffer SharedConstants : register(b2, space4)\n";
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Sampler)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            uint32_t fetchSlot = getFetchConstantSlot(constantInfo->registerIndex, isPixelShader);
            constexpr uint32_t slotRegisters = FETCH_CONSTANT_SLOT_COUNT / 4;

            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("\tuint {}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                    constantName, TEXTURE_DIMENSIONS[j], j * slotRegisters + fetchSlot / 4, SWIZZLES[fetchSlot % 4]);
            }

            println("\tuint {}_SamplerDescriptorIndex : packoffset(c{}.{});",
                constantName, slotRegisters * std::size(TEXTURE_DIMENSIONS) + fetchSlot / 4, SWIZZLES[fetchSlot % 4]);
        }
    }

    out += "\tDEFINE_SHARED_CONSTANTS();\n";
    out += "};\n\n";

    out += "#endif\n";
    out += "\n";
    out += "// Xenos applies the source blend factor BEFORE a MIN/MAX blend op. D3D12 and\n";
    out += "// Vulkan ignore the factors for those ops entirely, so the factor has to be\n";
    out += "// folded into the shader's own output instead.\n";
    out += "//\n";
    out += "// The runtime sets a non-zero mode only when the op is MIN or MAX, the source\n";
    out += "// factor is not ONE, and the destination factor IS ONE -- the destination term\n";
    out += "// is not ours to scale, so any other destination factor is left unemulated\n";
    out += "// rather than emulated wrong. Colour and alpha are independent equations and\n";
    out += "// carry independent modes. Mode 0 means leave it alone, which is what every\n";
    out += "// draw that does not hit the case uses, so the cost elsewhere is one uniform\n";
    out += "// compare.\n";
    out += "float3 blendPremultRgb(float3 rgb, float alpha)\n";
    out += "{\n";
    out += "    switch (g_BlendPremultModeRgb)\n";
    out += "    {\n";
    out += "        case 1:  return 0.0;\n";
    out += "        case 2:  return rgb * rgb;\n";
    out += "        case 3:  return rgb * (1.0 - rgb);\n";
    out += "        case 4:  return rgb * alpha;\n";
    out += "        case 5:  return rgb * (1.0 - alpha);\n";
    out += "        case 6:  return rgb * g_BlendPremultConstant.rgb;\n";
    out += "        case 7:  return rgb * (1.0 - g_BlendPremultConstant.rgb);\n";
    out += "        case 8:  return rgb * g_BlendPremultConstant.a;\n";
    out += "        case 9:  return rgb * (1.0 - g_BlendPremultConstant.a);\n";
    out += "        default: return rgb;\n";
    out += "    }\n";
    out += "}\n";
    out += "\n";
    out += "float blendPremultAlpha(float alpha)\n";
    out += "{\n";
    out += "    switch (g_BlendPremultModeA)\n";
    out += "    {\n";
    out += "        case 1:  return 0.0;\n";
    out += "        case 2:  return alpha * alpha;\n";
    out += "        case 3:  return alpha * (1.0 - alpha);\n";
    out += "        case 4:  return alpha * alpha;\n";
    out += "        case 5:  return alpha * (1.0 - alpha);\n";
    out += "        case 6:  return alpha * g_BlendPremultConstant.a;\n";
    out += "        case 7:  return alpha * (1.0 - g_BlendPremultConstant.a);\n";
    out += "        case 8:  return alpha * g_BlendPremultConstant.a;\n";
    out += "        case 9:  return alpha * (1.0 - g_BlendPremultConstant.a);\n";
    out += "        default: return alpha;\n";
    out += "    }\n";
    out += "}\n";
    out += "\n";
    out += "// Applied AFTER the alpha test: the test compares the shader's own alpha, and\n";
    out += "// pre-multiplying before it would change which pixels are discarded.\n";
    out += "float4 applyBlendPremult(float4 color)\n";
    out += "{\n";
    out += "    return float4(blendPremultRgb(color.rgb, color.a), blendPremultAlpha(color.a));\n";
    out += "}\n";
    out += "\n";

#ifndef UNLEASHED_RECOMP
    // Xenia's alpha to mask (DxbcShaderTranslator::CompletePixelShader_AlphaToMask,
    // BSD-3-Clause): each sample is covered when the shader's alpha -- before the
    // exponent bias, which the hardware applies after this test -- reaches its
    // threshold, lowered by a 2-bit dither offset chosen by the pixel's position
    // parity from RB_COLORCONTROL's alpha_to_mask_offset0..3. The host's own alpha
    // to coverage would test the biased alpha, so it is never enabled. Called by
    // the XENOS_ALPHA_COVERAGE variant only (shader_common.h).
    // g_AlphaToMask: bits 0-7 the four offsets, bit 8 enabled, bits 9-10 the
    // sample count log2. 2x: the guest's sample 0 (top) is the host's sample 1.
    out += "uint computeAlphaToMask(float alpha, float2 pixelPosition)\n";
    out += "{\n";
    out += "    uint params = g_AlphaToMask;\n";
    out += "    if ((params & 0x100u) == 0u)\n";
    out += "        return 0xFFFFFFFFu;\n";
    out += "    uint2 p = uint2(pixelPosition);\n";
    out += "    uint index = ((p.x & 1u) << 1) | (p.y & 1u);\n";
    out += "    float offset = float((params >> (index * 2u)) & 3u);\n";
    out += "    uint samplesLog2 = (params >> 9) & 3u;\n";
    out += "    uint coverage = 0u;\n";
    // Contract v8: 8 samples, which the host gives an image of the screen
    // the game multisamples at a higher MSAA level (the Xenos has no 8x).
    // Thresholds in eighths, the dither offset in 1/32 (four pixels make 32
    // levels, as the 4x branch's 16); the samples covered first are spread
    // over the pixel (the host's standard 8x positions).
    out += "    if (samplesLog2 == 3u)\n";
    out += "    {\n";
    out += "        float step = offset * (1.0 / 32.0);\n";
    out += "        coverage |= (alpha >= 0.875 - step) ? 1u : 0u;\n";
    out += "        coverage |= (alpha >= 1.0 - step) ? 2u : 0u;\n";
    out += "        coverage |= (alpha >= 0.375 - step) ? 4u : 0u;\n";
    out += "        coverage |= (alpha >= 0.125 - step) ? 8u : 0u;\n";
    out += "        coverage |= (alpha >= 0.75 - step) ? 16u : 0u;\n";
    out += "        coverage |= (alpha >= 0.5 - step) ? 32u : 0u;\n";
    out += "        coverage |= (alpha >= 0.25 - step) ? 64u : 0u;\n";
    out += "        coverage |= (alpha >= 0.625 - step) ? 128u : 0u;\n";
    out += "    }\n";
    out += "    else if (samplesLog2 == 2u)\n";
    out += "    {\n";
    out += "        float step = offset * (1.0 / 16.0);\n";
    out += "        coverage |= (alpha >= 0.75 - step) ? 1u : 0u;\n";
    out += "        coverage |= (alpha >= 0.25 - step) ? 2u : 0u;\n";
    out += "        coverage |= (alpha >= 0.5 - step) ? 4u : 0u;\n";
    out += "        coverage |= (alpha >= 1.0 - step) ? 8u : 0u;\n";
    out += "    }\n";
    out += "    else if (samplesLog2 == 1u)\n";
    out += "    {\n";
    out += "        float step = offset * (1.0 / 8.0);\n";
    out += "        coverage |= (alpha >= 0.5 - step) ? 2u : 0u;\n";
    out += "        coverage |= (alpha >= 1.0 - step) ? 1u : 0u;\n";
    out += "    }\n";
    out += "    else\n";
    out += "    {\n";
    out += "        coverage = (alpha >= 1.0 - offset * (1.0 / 4.0)) ? 1u : 0u;\n";
    out += "    }\n";
    out += "    return coverage;\n";
    out += "}\n";
    out += "\n";
#endif

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Bool)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
            println("\t#define {} (1 << {})", constantName, constantInfo->registerIndex + (isPixelShader ? 16 : 0));
            boolConstants.emplace(constantInfo->registerIndex, constantName);
        }
    }

    out += '\n';

    const auto shader = reinterpret_cast<const Shader*>(shaderData + shaderContainer->shaderOffset);

    out += "#ifndef __spirv__\n";

    if (isPixelShader)
        out += "[shader(\"pixel\")]\n";
    else
        out += "[shader(\"vertex\")]\n";

    out += "#endif\n";

    out += "void main(\n";

    if (isPixelShader)
    {
        out += "\tin float4 iPos : SV_Position,\n";

        // CENTROID. Without it D3D12 interpolates at the pixel CENTRE, which in a
        // partially covered pixel can lie OUTSIDE the triangle -- the attributes
        // are then extrapolated. 402 of MCLA's 1006 translated pixel shaders
        // divide by an interpolant (`clamp(rcp(...), FLT_MIN, FLT_MAX)`, this
        // translator's own rendering of the Xenos RECP), so an extrapolated value
        // near zero becomes millions.
        //
        // Measured in three RenderDoc captures of the game with 4x MSAA: ONE
        // sample of ONE texel gets a colossal value (-2007160, +47461 and -2040
        // seen) while its neighbours are fine. The MSAA resolve averages it in,
        // the post-process downsample spreads it over a 2x2 texel block at
        // 640x360, and the tonemap turns that into a 4x4 black (clamped) or white
        // (saturated) block on screen -- transient, anywhere, unreproducible.
        //
        // Centroid samples inside the covered area, which is what the modifier is
        // for. It changes nothing for a fully covered pixel, and with a single
        // sample centroid and centre ARE the same point, so it is a no-op when
        // MSAA is off.
        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\tin centroid float4 i{0}{1} : {2}{1},", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);

        out += "#ifdef __spirv__\n";
        out += "\tin bool iFace : SV_IsFrontFace\n";
        out += "#else\n";
        out += "\tin uint iFace : SV_IsFrontFace\n";
        out += "#endif\n";

        auto pixelShader = reinterpret_cast<const PixelShader*>(shader);
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR0)
        {
            out += ",\n\tout float4 oC0 : SV_Target0";
        #ifndef UNLEASHED_RECOMP
            // Alpha to coverage is the shader's (computeAlphaToMask), in the
            // variant whose coverage depends on alpha only: an SV_Coverage output
            // (or a discard) in every variant would cost every draw the host's
            // early depth/stencil test.
            out += "\n#if XENOS_ALPHA_COVERAGE\n\t, out uint oCoverage : SV_Coverage\n#endif\n";
        #endif
        }
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR1)
            out += ",\n\tout float4 oC1 : SV_Target1";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR2)
            out += ",\n\tout float4 oC2 : SV_Target2";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR3)
            out += ",\n\tout float4 oC3 : SV_Target3";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_DEPTH)
        {
        #ifdef UNLEASHED_RECOMP
            out += ",\n\tout float oDepth : SV_Depth";
        #else
            // precise for the reason oPos is: a depth that the depth test
            // compares against other shaders' must not depend on how the
            // compiler fused or reordered the arithmetic that produced it.
            out += ",\n\tout precise float oDepth : SV_Depth";
        #endif
        }
    }
    else
    {
        auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
        for (uint32_t i = 0; i < vertexShader->vertexElementCount; i++)
        {
            union
            {
                VertexElement vertexElement;
                uint32_t value;
            };

            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + i];

            const char* usageType = USAGE_TYPES[uint32_t(vertexElement.usage)];

        #ifdef UNLEASHED_RECOMP
            if ((vertexElement.usage == DeclUsage::TexCoord && vertexElement.usageIndex == 2 && isMetaInstancer) ||
                (vertexElement.usage == DeclUsage::Position && vertexElement.usageIndex == 1))
            {
                usageType = "uint4";
            }
        #endif

            out += '\t';

            for (auto& usageLocation : USAGE_LOCATIONS)
            {
                if (usageLocation.usage == vertexElement.usage && usageLocation.usageIndex == vertexElement.usageIndex)
                {
                    print("[[vk::location({})]] ", usageLocation.location);
                    break;
                }
            }

            println("in {0} i{1}{2} : {3}{2},", usageType, USAGE_VARIABLES[uint32_t(vertexElement.usage)],
                uint32_t(vertexElement.usageIndex), USAGE_SEMANTICS[uint32_t(vertexElement.usage)]);

            vertexElements.emplace(uint32_t(vertexElement.address), vertexElement);
        }

        // The Xenos preloads the vertex index into r0.x at vertex shader entry.
        // MCLA's billboard shaders depend on it: xPropFoliageImpostor computes
        // `r0.x % 4` to select a corner out of a four-entry table (uvs[4], the
        // UV/sign pairs of a quad), and the vfetch that reads POSITION takes its
        // index from `trunc(r0.x / 4)`. Without the input r0.x is a constant 0,
        // every corner selects entry 0, and each quad is assembled from four
        // DIFFERENT props -- one enormous black wedge per draw.
        //
        // Not gated on a game-specific constant: r0.x is the vertex index on the
        // hardware for every vertex shader, and 278 of MCLA's 1279 read it before
        // writing it. Shaders that write r0 first keep their own value, because
        // the initialiser below only runs for registers nothing else assigns.
        out += "\tin uint iVertexId : SV_VertexID,\n";
    #ifdef UNLEASHED_RECOMP
        if (hasIndexCount)
        {
            out += "\tin uint iInstanceId : SV_InstanceID,\n";
        }
    #endif

    #ifdef UNLEASHED_RECOMP
        out += "\tout float4 oPos : SV_Position";
    #else
        // precise: nothing that reaches SV_Position is contracted into a fused
        // multiply-add or reassociated, so two vertex shaders that run the same
        // guest arithmetic give the host the same bits. MCLA writes its depth
        // with a Z prepass and draws the colour pass with other vertex shaders
        // under GREATER_EQUAL; with fast math the driver fused a different
        // product of the position epilogue in each (the order DXC gave the
        // fadd's operands decided which), a vertex moved by one ulp, and runs
        // of the colour pass failed the depth test by one ulp for a frame,
        // leaving the clear colour on screen. Xenia keeps the same invariance
        // by not allowing refactoring (DXBC) and with NoContraction plus the
        // Invariant output decoration (SPIR-V). 18 is SPIR-V's Invariant; the
        // DXIL compile ignores the attribute like the vk::location ones.
        out += "\t[[vk::ext_decorate(18)]] out precise float4 oPos : SV_Position";
    #endif

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            print(",\n\tout float4 o{0}{1} : {2}{1}", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);
    }

    out += ")\n";
    out += "{\n";

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
    {
        specConstantsMask |= SPEC_CONSTANT_REVERSE_Z;

        out += "\toPos = 0.0;\n";

        out += "\tfloat4x4 mtxProjection = float4x4(g_MtxProjection(0), g_MtxProjection(1), g_MtxProjection(2), g_MtxProjection(3));\n";
        out += "\tfloat4x4 mtxProjectionReverseZ = mul(mtxProjection, float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1));\n";

        out += "\t[unroll] for (int iterationIndex = 0; iterationIndex < 2; iterationIndex++)\n";
        out += "\t{\n";
    }
#endif

    // Loop constants the shader defines itself (the i{N} literals below), by
    // index: count, start and step of the loop. Only these are known when the
    // shader is translated; see the LoopStart emission.
    struct LoopConstant
    {
        int count;
        int start;
        int step;
    };
    std::unordered_map<uint32_t, LoopConstant> definedLoopConstants;

    if (shaderContainer->definitionTableOffset != NULL)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable*>(shaderData + shaderContainer->definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            auto value = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + definition->physicalOffset);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
                println("\tfloat4 c{} = asfloat(uint4(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}));",
                    definition->registerIndex + i - (isPixelShader ? 256 : 0), value[0].get(), value[1].get(), value[2].get(), value[3].get());

                value += 4;
            }
            definitions += 2;
        }
        ++definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Int4Definition*>(definitions);
            for (uint16_t i = 0; i < definition->count; i++)
            {
                union
                {
                    uint32_t value;
                    struct
                    {
                        int8_t x;
                        int8_t y;
                        int8_t z;
                        int8_t w;
                    };
                };

                value = definition->values[i].get();

                println("\tint4 i{} = int4({}, {}, {}, {});",
                    (definition->registerIndex - 8992) / 4 + i, x, y, z, w);

                definedLoopConstants.emplace((definition->registerIndex - 8992) / 4 + i, LoopConstant{ x, y, z });
            }
            definitions += 2;
            definitions += definition->count;
        }

        out += "\n";
    }

    bool printedRegisters[32]{};

    uint32_t interpolatorCount = (shader->interpolatorInfo >> 5) & 0x1F;

    for (uint32_t i = 0; i < interpolatorCount; i++)
    {
        union
        {
            Interpolator interpolator;
            uint32_t value;
        };
    
        if (isPixelShader)
        {
            value = reinterpret_cast<const PixelShader*>(shader)->interpolators[i];
            println("\tfloat4 r{} = i{}{};", uint32_t(interpolator.reg), USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex));
            printedRegisters[interpolator.reg] = true;
        }
        else
        {
            auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + vertexShader->vertexElementCount + i];
            interpolators.emplace(i, fmt::format("o{}{}", USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex)));
        }
    }

    if (!isPixelShader)
    {
    #ifdef UNLEASHED_RECOMP
        if (!hasMtxProjection)
            out += "\toPos = 0.0;\n";
    #endif

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\to{}{} = 0.0;", USAGE_VARIABLES[uint32_t(usage)], usageIndex);

        out += "\n";
    }

    for (size_t i = 0; i < 32; i++)
    {
        if (!printedRegisters[i])
        {
            print("\tfloat4 r{} = ", i);
            if (isPixelShader && i == ((shader->fieldC >> 8) & 0xFF))
            {
                // In guest pixels (contract v7, g_InvResolutionScale in
                // shader_common.h); the sign of x is the face.
                out += "float4((iPos.xy * g_InvResolutionScale - 0.5) * float2(iFace ? 1.0 : -1.0, 1.0), 0.0, 0.0);\n";
            }
        #ifdef UNLEASHED_RECOMP
            else if (!isPixelShader && hasIndexCount && i == 0)
            {
                out += "float4(iVertexId + g_IndexCount.x * iInstanceId, 0.0, 0.0, 0.0);\n";
            }
        #endif
            else if (!isPixelShader && i == 0)
            {
                // The vertex index the hardware preloads. See the SV_VertexID
                // declaration above for why this is not optional.
                out += "float4(iVertexId, 0.0, 0.0, 0.0);\n";
            }
            else
            {
                out += "0.0;\n";
            }
        }
    }

    out += "\tint a0 = 0;\n";
    out += "\tint aL = 0;\n";
    out += "\tbool p0 = false;\n";
    out += "\tfloat ps = 0.0;\n";
    // Pre-instruction copy of a GPR the vector half is about to overwrite, so the
    // scalar half of the same instruction still reads the value the hardware
    // would have handed it.
    out += "\tfloat4 psSrc = 0.0;\n";
    if (isPixelShader)
    {
#ifdef UNLEASHED_RECOMP
        out += "\tfloat2 pixelCoord = 0.0;\n";
#endif
        out += "\tCubeMapData cubeMapData = (CubeMapData)0;\n";
    }

    const be<uint32_t>* code = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset);

    union
    {
        ControlFlowInstruction controlFlow[2];
        struct
        {
            uint32_t code0;
            uint32_t code1;
            uint32_t code2;
            uint32_t code3;
        };
    };

    auto controlFlowCode = code;
    uint32_t instrAddress = 0;
    uint32_t instrSize = shader->size;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            uint32_t address = 0;
            bool isExecBlock = false;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                isExecBlock = true;
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                isExecBlock = true;
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                isExecBlock = true;
                break;
            }

            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);

            // Find out whether the shader writes the implicit texture fetch state
            // registers before any sample is emitted, so fetches that reference
            // them can pick the right sampling form.
            if (isExecBlock)
            {
                uint32_t sequence = cfInstr.exec.sequence;
                for (uint32_t i = 0; i < cfInstr.exec.count; i++, sequence >>= 2)
                {
                    if ((sequence & 0x1) == 0)
                        continue;

                    switch (FetchOpcode(code[(address + i) * 3].get() & 0x1F))
                    {
                    case FetchOpcode::SetTextureLod:
                        usesTextureLodRegister = true;
                        break;
                    case FetchOpcode::SetTextureGradientsHorz:
                    case FetchOpcode::SetTextureGradientsVert:
                        usesTextureGradientRegisters = true;
                        break;
                    default:
                        break;
                    }
                }
            }
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    // Emitted only for shaders that actually issue setTexLOD / setGradients, so
    // the generated code for everything else is unchanged.
    if (usesTextureLodRegister)
        out += "\tfloat tfetchLod = 0.0;\n";
    if (usesTextureGradientRegisters)
    {
        out += "\tfloat3 tfetchGradientH = 0.0;\n";
        out += "\tfloat3 tfetchGradientV = 0.0;\n";
    }

    std::vector<ControlFlowInstruction> controlFlowList;
    controlFlowCode = code;
    for (instrAddress = 0; instrAddress < instrSize; instrAddress += 12)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        controlFlowList.push_back(controlFlow[0]);
        controlFlowList.push_back(controlFlow[1]);
        controlFlowCode += 3;
    }

    const ControlFlowPlan controlFlowPlan = planControlFlow(controlFlowList);
    controlFlowForm = controlFlowPlan.form;
    const bool pcMachine = (controlFlowPlan.form == ControlFlowForm::PcMachine);

    // A program this translator cannot express stops the shader at compile
    // time with the reason, instead of shipping code that does something else
    // than the hardware.
    if (!controlFlowPlan.error.empty())
        println("#error XenosRecomp: {}", controlFlowPlan.error);

    if (!pcMachine)
    {
        out += '\n';
        indentation = 1;
    }
    else
    {
        out += "\n\tuint pc = 0;\n";
        out += "\twhile (true)\n";
        out += "\t{\n";
        out += "\t\tswitch (pc)\n";
        out += "\t\t{\n";
    }

    // The blocks the structured form closes before the instruction at an
    // index, innermost first; see planControlFlow.
    auto closeBlocks = [&](uint32_t index)
    {
        for (ControlFlowPlan::BlockAction action : controlFlowPlan.blockActionsBefore[index])
        {
            --indentation;
            indent();
            out += "}\n";

            if (action == ControlFlowPlan::BlockAction::CloseThenOpenElse)
            {
                indent();
                out += "else\n";
                indent();
                out += "{\n";
                ++indentation;
            }
        }
    };

    controlFlowCode = code;
    instrAddress = 0;
    uint32_t pc = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            if (pcMachine)
            {
                indentation = 3;
                println("\t\tcase {}:", pc);
            }
            else
            {
                closeBlocks(pc);
            }

            const bool isElseJump = controlFlowPlan.isElseJump[pc];
            ++pc;

            uint32_t address = 0;
            uint32_t count = 0;
            uint32_t sequence = 0;
            bool shouldReturn = false;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                count = cfInstr.exec.count;
                sequence = cfInstr.exec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::ExecEnd);
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                count = cfInstr.condExec.count;
                sequence = cfInstr.condExec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecEnd || cfInstr.opcode == ControlFlowOpcode::CondExecPredCleanEnd);
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                count = cfInstr.condExecPred.count;
                sequence = cfInstr.condExecPred.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd);
                break;

            case ControlFlowOpcode::LoopStart:
            {
                // Both forms count aL from 0 in steps of 1 up to the loop count,
                // which is the hardware's loop only for a constant whose start is 0
                // and step is 1, and they read the count from the i{N} literal the
                // shader defines. A loop that is not like that has to fail.
                auto loopConstant = definedLoopConstants.find(cfInstr.loopStart.loopId);
                if (loopConstant == definedLoopConstants.end())
                    println("#error XenosRecomp: loop over runtime loop constant i{}", uint32_t(cfInstr.loopStart.loopId));
                else if (loopConstant->second.start != 0 || loopConstant->second.step != 1)
                    println("#error XenosRecomp: loop constant i{} starts at {} with step {}", uint32_t(cfInstr.loopStart.loopId),
                        loopConstant->second.start, loopConstant->second.step);
                if (cfInstr.loopStart.isRepeat)
                    println("#error XenosRecomp: repeat loop (iterator kept across LoopStart) at {}", pc - 1);

                if (!pcMachine)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    print("[unroll] ");
                #endif
                    println("for (aL = 0; aL < i{}.x; aL++)", uint32_t(cfInstr.loopStart.loopId));
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                else
                {
                    out += "\t\t\taL = 0;\n";
                    // A loop count of 0 skips the body, as the for statement does.
                    println("\t\t\tif (aL >= i{}.x)", uint32_t(cfInstr.loopStart.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopStart.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;
            }

            case ControlFlowOpcode::LoopEnd:
                if (cfInstr.loopEnd.isPredicatedBreak)
                    println("#error XenosRecomp: predicated loop break at {}", pc - 1);

                if (!pcMachine)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                }
                else
                {
                    out += "\t\t\t++aL;\n";
                    println("\t\t\tif (aL < i{}.x)", uint32_t(cfInstr.loopEnd.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopEnd.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (isElseJump)
                {
                    // The end of an if's then-block: the "else" closeBlocks opens
                    // at the jump's target is this jump.
                }
                else if (cfInstr.condJmp.isUnconditional)
                {
                    assert(pcMachine);
                    println("\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                    out += "\t\t\tcontinue;\n";
                }
                else
                {
                    // Structured, the block runs when the jump is NOT taken; the pc
                    // machine tests for the jump itself.
                    const bool skipsBlock = !pcMachine;

                    indent();
                    if (cfInstr.condJmp.isPredicated)
                    {
                        println("if ({}p0)", (cfInstr.condJmp.condition ^ skipsBlock) ? "" : "!");
                    }
                    else
                    {
                        auto findResult = boolConstants.find(cfInstr.condJmp.boolAddress);
                        if (findResult != boolConstants.end())
                            println("if ((g_Booleans & {}) {}= 0)", findResult->second, (cfInstr.condJmp.condition ^ skipsBlock) ? "!" : "=");
                        else
                            println("if (b{} {}= 0)", uint32_t(cfInstr.condJmp.boolAddress), (cfInstr.condJmp.condition ^ skipsBlock) ? "!" : "=");
                    }

                    if (!pcMachine)
                    {
                        indent();
                        out += "{\n";
                        ++indentation;
                    }
                    else
                    {
                        out += "\t\t\t{\n";
                        println("\t\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\t\tcontinue;\n";
                        out += "\t\t\t}\n";
                    }
                }
                break;
            }
            }

            // Conditional execution. The hardware tests the block's condition once,
            // before the block, and skips all of it -- including the end of the
            // program, for the End variants -- when the test fails. In a predicated
            // exec the compiler predicates the instructions too, because the
            // hardware enters the block when ANY of its 64 invocations passes, so
            // the block's own test is usually redundant (see below). The
            // bool-constant form is not: it guards enableVertDamage in 28 car
            // vertex shaders, which the game draws with the bool both set and clear
            // in one frame, and without the test both sides of it ran.
            bool isConditionalBlock = false;
            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
            {
                // The constant table numbers a pixel shader's bools from 0 like a
                // vertex shader's; where they sit in the bool address space this
                // test reads is not established, and no Midnight Club pixel shader
                // has the case.
                auto findResult = boolConstants.find(cfInstr.condExec.boolAddress);
                if (isPixelShader || findResult == boolConstants.end())
                {
                    println("#error XenosRecomp: exec conditioned on bool constant {} in a {} shader", uint32_t(cfInstr.condExec.boolAddress),
                        isPixelShader ? "pixel" : "vertex (not in the constant table)");
                }
                else
                {
                    indent();
                    println("if ((g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condExec.condition ? "!" : "=");
                    isConditionalBlock = true;
                }
                break;
            }

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
            {
                // When every instruction of the block is predicated on the block's
                // own condition and the block does not end the program, the test is
                // redundant and is left out: if p0 fails it at entry, the first
                // instruction is skipped, so it cannot change p0, and neither can
                // any after it. The ALU no-op (max with no write mask and no export,
                // scalar retain_prev) changes nothing and does not count. That covers
                // all 1219 of Midnight Club's predicated blocks -- the 9 that hold an
                // unpredicated instruction hold exactly that no-op -- so the test is
                // emitted for none of them; it exists for the shaders where it is not
                // redundant.
                bool everyInstructionPredicated = true;
                uint32_t blockSequence = cfInstr.condExecPred.sequence;
                for (uint32_t i = 0; i < count; i++, blockSequence >>= 2)
                {
                    union
                    {
                        VertexFetchInstruction vertexFetch;
                        TextureFetchInstruction textureFetch;
                        AluInstruction alu;
                        struct
                        {
                            uint32_t code0;
                            uint32_t code1;
                            uint32_t code2;
                        };
                    };

                    const be<uint32_t>* blockCode = code + (address + i) * 3;
                    code0 = blockCode[0];
                    code1 = blockCode[1];
                    code2 = blockCode[2];

                    bool isPredicated;
                    uint32_t predicateCondition;
                    if ((blockSequence & 0x1) == 0)
                    {
                        if (alu.vectorWriteMask == 0 && alu.scalarWriteMask == 0 && !alu.exportData &&
                            alu.vectorOpcode == AluVectorOpcode::Max && alu.scalarOpcode == AluScalarOpcode::RetainPrev)
                        {
                            continue;
                        }

                        isPredicated = alu.isPredicated;
                        predicateCondition = alu.predicateCondition;
                    }
                    else if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        isPredicated = vertexFetch.isPredicated;
                        predicateCondition = vertexFetch.predicateCondition;
                    }
                    else
                    {
                        isPredicated = textureFetch.isPredicated;
                        predicateCondition = textureFetch.predCondition;
                    }

                    if (!isPredicated || predicateCondition != cfInstr.condExecPred.condition)
                        everyInstructionPredicated = false;
                }

                if (shouldReturn || !everyInstructionPredicated)
                {
                    indent();
                    println("if ({}p0)", cfInstr.condExecPred.condition ? "" : "!");
                    isConditionalBlock = true;
                }
                break;
            }

            default:
                break;
            }

            if (isConditionalBlock)
            {
                indent();
                out += "{\n";
                ++indentation;
            }

            auto instructionCode = code + address * 3;

            for (uint32_t i = 0; i < count; i++)
            {
                union
                {
                    VertexFetchInstruction vertexFetch;
                    TextureFetchInstruction textureFetch;
                    AluInstruction alu;
                    struct
                    {
                        uint32_t code0;
                        uint32_t code1;
                        uint32_t code2;
                    };
                };
            
                code0 = instructionCode[0];
                code1 = instructionCode[1];
                code2 = instructionCode[2];
            
                if ((sequence & 0x1) != 0)
                {
                    if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        recompile(vertexFetch, address + i);
                    }
                    else
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (textureFetch.constIndex == 10) // g_GISampler
                        {
                            specConstantsMask |= SPEC_CONSTANT_BICUBIC_GI_FILTER;

                            indent();
                            out += "if (g_SpecConstants() & SPEC_CONSTANT_BICUBIC_GI_FILTER)";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, true);
                            --indentation;

                            indent();
                            out += "}";
                            indent();
                            out += "else";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, false);
                            --indentation;

                            indent();
                            out += '}';
                        }
                        else
                    #endif
                        {
                            recompile(textureFetch, false);
                        }
                    }
                }
                else
                {
                    recompile(alu);
                }
            
                sequence >>= 2;
                instructionCode += 3;
            }

            if (shouldReturn)
            {
                if (isPixelShader)
                {
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST;

                    indent();
                    out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)";
                    indent();
                    out += '{';

                    indent();
                    out += "\tclip(oC0.w - g_AlphaThreshold);\n";

                #ifndef UNLEASHED_RECOMP
                    // Alpha to coverage after the alpha test and before the exponent
                    // bias (in applyBlendPremult), as the hardware orders them; no
                    // covered sample = nothing written. The runtime runs this
                    // variant for the alpha test, alpha to coverage or both: a
                    // threshold of -infinity discards nothing, g_AlphaToMask 0
                    // covers every sample.
                    if (reinterpret_cast<const PixelShader*>(shader)->outputs & PIXEL_SHADER_OUTPUT_COLOR0)
                    {
                        out += "#if XENOS_ALPHA_COVERAGE\n";
                        indent();
                        out += "\toCoverage = computeAlphaToMask(oC0.w, iPos.xy);\n";
                        indent();
                        out += "\tif (oCoverage == 0u) discard;\n";
                        out += "#endif\n";
                    }
                #endif

                    indent();
                    out += "}";

                #ifdef UNLEASHED_RECOMP
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TO_COVERAGE;

                    indent();
                    out += "else if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_COVERAGE)";
                    indent();
                    out += '{';

                    indent();
                    out += "\toC0.w *= 1.0 + computeMipLevel(pixelCoord) * 0.25;\n";
                    indent();
                    out += "\toC0.w = 0.5 + (oC0.w - g_AlphaThreshold) / max(fwidth(oC0.w), 1e-6);\n";

                    indent();
                    out += '}';
                #endif

                    // Xenos applies the source blend factor before a MIN/MAX blend op;
                    // D3D12 and Vulkan ignore the factors there, so the factor has to
                    // be folded into the output instead. Emitted after the alpha test on
                    // purpose: the test compares this shader's own alpha, and scaling it
                    // first would change which pixels are discarded. Inert -- one uniform
                    // compare -- unless the runtime sets a mode for the draw.
                    indent();
                    out += "\toC0 = applyBlendPremult(oC0);\n";
                }
                else
                {
                #ifdef UNLEASHED_RECOMP
                    if (!hasMtxProjection)
                    {
                        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
                    }
                #else
                    // Xenos clip space to the host's; see g_PosScale in
                    // shader_common.h.
                    indent();
                    out += "oPos.xyz = oPos.xyz * g_PosScale.xyz + g_PosOffset.xyz * oPos.w;\n";
                #endif
                }

                if (!pcMachine)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    if (hasMtxProjection)
                    {
                        out += "continue;\n";
                    }
                    else
                #endif
                    {
                        out += "return;\n";
                    }
                }
                else
                {
                    out += "\t\t\tbreak;\n";
                }
            }

            if (isConditionalBlock)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (!pcMachine)
    {
        // Blocks whose jump targets the end of the program.
        closeBlocks(pc);
    }
    else
    {
        out += "\t\t\tbreak;\n";
        out += "\t\t}\n";
        out += "\t\tbreak;\n";
        out += "\t}\n";
    }

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
        out += "\t}\n";

    if (!isPixelShader && hasMtxProjection)
        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
#endif

    out += "}";
}

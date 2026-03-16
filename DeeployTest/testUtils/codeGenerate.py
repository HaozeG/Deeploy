# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from typing import List, Optional, Sequence, Tuple

import numpy as np

from Deeploy.DeeployTypes import CodeGenVerbosity, ConstantBuffer, NetworkDeployer, VariableBuffer
from Deeploy.Targets.MemPool.Platform import MemPoolPlatform
from Deeploy.Targets.PULPOpen.Platform import MemoryPULPPlatform, MemoryPULPPlatformWrapper, PULPPlatform

_TEXT_ALIGN = 30


def _shapeBroadcast(ctxt, value, name):
    if ctxt.is_global(f"{name}"):
        broadcastShape = ctxt.lookup(f"{name}").shape
        repeat = np.prod(broadcastShape) / np.prod(value.shape)
        # Raise error if repeat is not an integer
        if repeat % 1 != 0:
            raise ValueError(f"Input {name} has to be broadcastable to shape {broadcastShape}!")
        repeatNum = np.tile(value, int(repeat))
        broadcastNum = repeatNum.reshape(-1)
        ctxt.lookup(f"{name}").shape = broadcastNum.shape
    else:
        broadcastNum = value

    return broadcastNum


def generateTestInputsHeader(deployer: NetworkDeployer, test_inputs: List) -> str:
    vectors = []
    retStr = ""
    for index, values in enumerate(test_inputs):
        # WIESEP: Correctly handle empty arrays
        if np.prod(values.shape) == 0:
            continue

        bufferName = f"input_{index}"

        #LMACAN: We have some tests which have extra inputs and this is a hack to circumvent that
        if not deployer.ctxt.is_buffer(bufferName):
            continue

        values = _shapeBroadcast(deployer.ctxt, values, bufferName)

        buffer = deployer.ctxt.lookup(bufferName)
        typeName = buffer._type.referencedType.typeName
        typeWidth = buffer._type.referencedType.typeWidth

        vectorName = f"testInputVector{index}"
        vectors.append(vectorName)

        retStr += f"{typeName} {vectorName}[] ="
        retStr += "{"
        if typeName == 'float32_t':
            list_str = (", ").join([f'{x}f' if not (np.isinf(x) or np.isnan(x)) else str(x) for x in values])
        else:
            list_str = (", ").join([str(x) for x in values])

        # WIESEP: Arrays have to be 4 byte aligned (at least in banshee)
        total_bytes = (values.size * typeWidth) // 8
        pad_bytes = (-total_bytes) % 4
        if pad_bytes:
            paddingElements = (pad_bytes * 8 + typeWidth - 1) // typeWidth
            list_str += ", " + (", ").join("0" for _ in range(paddingElements))

        retStr += list_str
        retStr += "};\n"

    retStr += f"void* testInputVector[{len(vectors)}] = {{"
    retStr += ", ".join(vectors)
    retStr += "};\n"

    return retStr


def generateTestOutputsHeader(deployer: NetworkDeployer, test_outputs: List[np.ndarray]) -> str:
    retStr = ""
    for index, values in enumerate(test_outputs):
        typeName = deployer.ctxt.lookup(f'output_{index}')._type.referencedType.typeName
        typeWidth = deployer.ctxt.lookup(f'output_{index}')._type.referencedType.typeWidth

        retStr += f"#define OUTPUTTYPE {typeName}\n"
        if deployer.Platform.engines[0].name == "SoftHier":
            # inputs, outputs and constants on SoftHier are in float32, would be converted to customized float16 in the kernels 
            if typeName == "float32_t":
                retStr += f"#define ISFLOAT32 1\n"
            else:
                retStr += f"#define ISFLOAT32 0\n"
        retStr += f"#define ISOUTPUTFLOAT {int(typeName == 'float32_t')}\n"
        retStr += f"{typeName} testOutputVector{index}[] ="
        retStr += "{"

        values = values.flatten()

        if typeName == "float32_t":
            list_str = (", ").join([f'{x}f' if not (np.isinf(x) or np.isnan(x)) else str(x) for x in values])
        else:
            list_str = (", ").join([str(x) for x in values])

        # WIESEP: Arrays have to be 4 byte aligned (at least in banshee)
        total_bytes = (len(values) * typeWidth) // 8
        pad_bytes = (-total_bytes) % 4
        if pad_bytes:
            paddingElements = (pad_bytes * 8 + typeWidth - 1) // typeWidth
            list_str += ", " + (", ").join("0" for _ in range(paddingElements))

        retStr += list_str
        retStr += "};\n"

    retStr += f"void* testOutputVector[{len(test_outputs)}] = " + "{"
    retStr += ", ".join([f"testOutputVector{idx}" for idx, _ in enumerate(test_outputs)])
    retStr += "};\n"

    return retStr


def generateTestNetworkHeader(deployer: NetworkDeployer) -> str:

    retStr = ""

    retStr += """
    #ifndef __DEEPLOY_HEADER__
    #define __DEEPLOY_HEADER__
    #include <stdio.h>
    #include <stdint.h>
    #include <stdlib.h>
    """
    retStr += deployer.generateIncludeString()
    if isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
        void RunNetwork();
        void InitNetwork();

        """
    else:
        retStr += """
        void RunNetwork(uint32_t core_id, uint32_t numThreads);
        void InitNetwork(uint32_t core_id, uint32_t numThread);

        """

    retStr += deployer.generateIOBufferInitializationCode()
    retStr += """
    #endif
    """

    return retStr


def generateTestNetworkImplementation(deployer: NetworkDeployer, verbosityCfg: CodeGenVerbosity) -> str:
    retStr = ""

    retStr += """#include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    """
    retStr += deployer.generateIncludeString()
    retStr += """

    #include "Network.h"

    """

    retStr += deployer.generateBufferInitializationCode()
    retStr += deployer.generateGlobalDefinitionCode()

    # WIESEP: Mempool assigns section attributes to intermediate buffers to allow .
    if isinstance(deployer.Platform, MemPoolPlatform):
        retStr += deployer.generateInferenceInitializationCode()
        retStr += """
        void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
        """
    elif isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
        void RunNetwork(){
        """
        retStr += deployer.generateInferenceInitializationCode()
    else:
        retStr += """
        void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
        """
        retStr += deployer.generateInferenceInitializationCode()

    retStr += deployer.generateFunction(verbosityCfg)
    if isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
        }

        void InitNetwork(){
        """
    else:
        retStr += """
        }

        void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
        """
    retStr += deployer.generateEngineInitializationCode()
    retStr += deployer.generateBufferAllocationCode()
    retStr += """
    }
    """

    return retStr


def generateL3HexDump(deployer: NetworkDeployer, path: str, test_inputs: List, test_outputs: List):

    def type2TypeStr(dataType) -> Tuple[str, int]:
        if dataType.referencedType.typeName == "float32_t":
            retStr = "float32"
            width = 32
        else:
            width = dataType.referencedType.typeWidth
            signed = (dataType.referencedType.typeMin < 0)

            retStr = ""

            if signed:
                retStr += "int"
            else:
                retStr += "uint"

            retStr += str(width)

        return retStr, width

    def dumpBuffer(buf: VariableBuffer, path: str):

        # Check if buffer name matches exactly "input_N" or "output_N" pattern
        parts = buf.name.split("_")
        if len(parts) == 2 and parts[0] == "input" and parts[1].isdigit():
            idx = int(parts[1])
            array = _shapeBroadcast(deployer.ctxt, test_inputs[idx], f"input_{idx}")

        elif len(parts) == 2 and parts[0] == "output" and parts[1].isdigit():
            idx = int(parts[1])
            array = _shapeBroadcast(deployer.ctxt, test_outputs[idx], f"output_{idx}")

        elif isinstance(buf, ConstantBuffer):
            array = buf.values
        else:
            raise Exception(f"Unexpected buffer {buf}!")

        typeStr, width = type2TypeStr(buf._type)

        # Word alignment
        mod = (32 // width)
        paddingLength = (mod - (array.size % mod)) % mod
        paddedArray = np.pad(array.flatten(), (0, paddingLength), 'constant')

        paddedArray.astype(typeStr).tofile(path)

    # LMACAN: Dump all global buffers with the "extName" attribute
    os.makedirs(path, exist_ok = True)
    for buf in deployer.ctxt.globalObjects.values():
        if hasattr(buf, "extName"):
            pathName = os.path.join(path, f"{buf.extName}.hex")
            dumpBuffer(buf, pathName)


def generateTestNetwork(deployer: NetworkDeployer, test_inputs: List[np.ndarray], test_outputs: List[np.ndarray],
                        dumpdir: str, verbosityCfg: CodeGenVerbosity) -> None:
    assert deployer.prepared, "An unprepared deployer was given"

    # Create input and output vectors
    os.makedirs(dumpdir, exist_ok = True)

    testInputStr = generateTestInputsHeader(deployer, test_inputs)
    with open(f'{dumpdir}/testinputs.h', "w") as f:
        f.write(testInputStr)

    testOutputStr = generateTestOutputsHeader(deployer, test_outputs)
    with open(f'{dumpdir}/testoutputs.h', "w") as f:
        f.write(testOutputStr)

    # Generate code for Network
    testNetworkHeaderStr = generateTestNetworkHeader(deployer)
    with open(f'{dumpdir}/Network.h', "w") as f:
        f.write(testNetworkHeaderStr)

    testNetworkImplementationStr = generateTestNetworkImplementation(deployer, verbosityCfg)
    with open(f'{dumpdir}/Network.c', "w") as f:
        f.write(testNetworkImplementationStr)

    generateL3HexDump(deployer, os.path.join(f'{dumpdir}', 'hex'), test_inputs, test_outputs)

    clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.c')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.h')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testoutputs.h')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testinputs.h')


def _tilelang_numpy_to_c_literal(value) -> str:
    if isinstance(value, np.floating):
        return f"{float(value)}f"
    return str(int(value))


def _tilelang_numpy_dtype_to_ctype(dtype: np.dtype) -> str:
    if dtype == np.float16:
        return "fp16"
    if dtype == np.float32:
        return "float32_t"
    if dtype == np.int8:
        return "int8_t"
    if dtype == np.uint8:
        return "uint8_t"
    if dtype == np.int16:
        return "int16_t"
    if dtype == np.uint16:
        return "uint16_t"
    if dtype == np.int32:
        return "int32_t"
    if dtype == np.uint32:
        return "uint32_t"
    raise ValueError(f"Unsupported TileLang vector dtype for C header generation: {dtype}")


def _generate_tilelang_vectors_header(var_prefix: str, arrays: Sequence[np.ndarray]) -> str:
    if not arrays:
        return f"void* {var_prefix}Vector[0] = {{}};\n"

    retStr = ""
    names = []
    for idx, arr in enumerate(arrays):
        flat = np.asarray(arr).reshape(-1)
        ctype = _tilelang_numpy_dtype_to_ctype(flat.dtype)
        var_name = f"{var_prefix}Vector{idx}"
        names.append(var_name)
        elems = ", ".join(_tilelang_numpy_to_c_literal(v) for v in flat)
        retStr += f"{ctype} {var_name}[] = {{{elems}}};\n"

    retStr += f"void* {var_prefix}Vector[{len(names)}] = " + "{" + ", ".join(names) + "};\n"
    return retStr


def generateTilelangSoftHierNetworkHeader(
    functionSignature: str = "void tilelang_main(fp16* A, fp16* B, fp16* C)"
) -> str:
    return f"""
#ifndef __DEEPLOY_TILELANG_SOFTHIER_HEADER__
#define __DEEPLOY_TILELANG_SOFTHIER_HEADER__

#include <stdint.h>
#include "flex_types.h"

{functionSignature};

#endif
"""


def generateTilelangSoftHierNetworkImplementation(
    tilelangBody: str,
    functionSignature: str = "void tilelang_main(fp16* A, fp16* B, fp16* C)",
    includeList: Optional[Sequence[str]] = None,
    deployer: Optional[NetworkDeployer] = None,
    bufferInitializationCode: Optional[str] = None,
    globalDefinitionCode: Optional[str] = None,
) -> str:
    resolved_includes = list(includeList) if includeList is not None else [
        "flex_alloc_api.h",
        "flex_runtime_api.h",
        "flex_redmule_api.h",
        "flex_dma_api.h",
        "flex_group_barrier_api.h",
        "flex_types.h",
        "flex_printf_api.h",
        "DeeploySoftHierMath.h",
    ]

    includeStr = ""
    for include in resolved_includes:
        includeStr += f'#include "{include}"\n'
    includeStr += "#include \"Network.h\"\n"
    includeStr += "#include <stdint.h>\n"
    includeStr += "#include <string.h>\n"

    # If snippets are not provided explicitly, derive them from the deployer.
    resolved_buffer_init = bufferInitializationCode
    resolved_global_defs = globalDefinitionCode

    if deployer is not None:
        if resolved_buffer_init is None:
            resolved_buffer_init = deployer.generateBufferInitializationCode()
        if resolved_global_defs is None:
            resolved_global_defs = deployer.generateGlobalDefinitionCode()

    if resolved_buffer_init is None:
        resolved_buffer_init = ""
    if resolved_global_defs is None:
        resolved_global_defs = ""

    return f"""{includeStr}
{resolved_buffer_init}
{resolved_global_defs}
{functionSignature} {{
{tilelangBody}
}}
"""


def generateTilelangSoftHierTestNetwork(
    tilelangBody: str,
    dumpdir: str,
    functionSignature: str = "void tilelang_main(fp16* A, fp16* B, fp16* C)",
    test_inputs: Optional[Sequence[np.ndarray]] = None,
    test_outputs: Optional[Sequence[np.ndarray]] = None,
    includeList: Optional[Sequence[str]] = None,
    deployer: Optional[NetworkDeployer] = None,
    bufferInitializationCode: Optional[str] = None,
    globalDefinitionCode: Optional[str] = None,
) -> None:
    os.makedirs(dumpdir, exist_ok = True)

    networkHeader = generateTilelangSoftHierNetworkHeader(functionSignature)
    with open(f"{dumpdir}/Network.h", "w", encoding = "utf-8") as f:
        f.write(networkHeader)

    networkImpl = generateTilelangSoftHierNetworkImplementation(
        tilelangBody,
        functionSignature = functionSignature,
        includeList = includeList,
        deployer = deployer,
        bufferInitializationCode = bufferInitializationCode,
        globalDefinitionCode = globalDefinitionCode,
    )
    with open(f"{dumpdir}/Network.c", "w", encoding = "utf-8") as f:
        f.write(networkImpl)

    input_header = _generate_tilelang_vectors_header("testInput", list(test_inputs or []))
    with open(f"{dumpdir}/testinputs.h", "w", encoding = "utf-8") as f:
        f.write(input_header)

    output_header = _generate_tilelang_vectors_header("testOutput", list(test_outputs or []))
    with open(f"{dumpdir}/testoutputs.h", "w", encoding = "utf-8") as f:
        f.write(output_header)

    if shutil.which("clang-format"):
        clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.c')
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.h')
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testoutputs.h')
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testinputs.h')

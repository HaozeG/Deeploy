# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import shutil
import subprocess
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


_HBM_START_BASE = 0xC0000000
_HBM_ALIGNMENT = 64
# Offset preloaded data past the main ELF's .data and .hbm sections (which start
# at HBM_BASE).  Without this the ELF loader overwrites preloaded values with the
# .hbm section's PROGBITS contents.
_PRELOAD_HBM_OFFSET = 0x100000


def _compute_preload_addresses(arrays: Sequence[np.ndarray]) -> List[int]:
    """Compute HBM addresses for preloading arrays, returning one address per array."""
    addrs = []
    next_addr = _HBM_START_BASE + _PRELOAD_HBM_OFFSET
    for arr in arrays:
        flat = np.asarray(arr)
        # Match the upcast convention: fp16 arrays are written as float32 in binary
        elem_size = 4 if flat.dtype == np.float16 else flat.dtype.itemsize
        nbytes = flat.size * elem_size
        next_addr = (next_addr + _HBM_ALIGNMENT - 1) & ~(_HBM_ALIGNMENT - 1)
        addrs.append(next_addr)
        next_addr += nbytes
    return addrs


def _resolve_toolchain_prefix() -> str:
    """Resolve the riscv32 toolchain prefix from env or PATH."""
    # Check explicit toolchain env vars first
    for env_var in ("TOOLCHAIN_INSTALL_DIR", "LLVM_INSTALL_DIR"):
        toolchain_dir = os.environ.get(env_var, "")
        if toolchain_dir:
            prefix = os.path.join(toolchain_dir, "bin", "riscv32-unknown-elf")
            if os.path.isfile(f"{prefix}-gcc"):
                return prefix
    # Check SOFTHIER_INSTALL_DIR for the bundled toolchain
    softhier_dir = os.environ.get("SOFTHIER_INSTALL_DIR", "")
    if softhier_dir:
        prefix = os.path.join(softhier_dir, "third_party", "toolchain", "install", "bin",
                              "riscv32-unknown-elf")
        if os.path.isfile(f"{prefix}-gcc"):
            return prefix
    # Fallback: search PATH
    gcc_path = shutil.which("riscv32-unknown-elf-gcc")
    if gcc_path:
        return gcc_path[:-4]  # strip "-gcc" suffix
    return "riscv32-unknown-elf"  # last resort


def _write_preload_elf(all_arrays: Sequence[np.ndarray],
                       output_elf_path: str,
                       toolchain_prefix: Optional[str] = None) -> Optional[List[int]]:
    """Generate a preload ELF from numpy arrays, placed at computed HBM addresses.

    Writes each array as a raw binary file (with fp16 -> float32 upcast), converts to
    ELF objects via objcopy (bypassing the C compiler), and links them at fixed HBM
    addresses. Returns the list of assigned addresses, or None if there are no arrays.
    """
    if toolchain_prefix is None:
        toolchain_prefix = _resolve_toolchain_prefix()

    if not all_arrays:
        return None

    addrs = _compute_preload_addresses(all_arrays)

    obj_files = []
    bin_files = []
    try:
        for i, (arr, addr) in enumerate(zip(all_arrays, addrs)):
            flat = np.asarray(arr)
            if flat.dtype == np.float16:
                data = flat.astype(np.float32)
            else:
                data = flat

            bin_path = f"{output_elf_path}.{i}.bin"
            obj_path = f"{output_elf_path}.{i}.o"
            data.tofile(bin_path)
            bin_files.append(bin_path)

            subprocess.run(
                [f"{toolchain_prefix}-objcopy", "-I", "binary", "-O", "elf32-littleriscv",
                 "-B", "riscv", bin_path, obj_path],
                check=True, capture_output=True)
            obj_files.append(obj_path)

        # Linker script: place each object's .data section at its assigned address
        ld_path = f"{output_elf_path}.ld"
        with open(ld_path, "w") as f:
            f.write("SECTIONS {\n")
            for i, (obj, addr) in enumerate(zip(obj_files, addrs)):
                f.write(f"    . = 0x{addr:X};\n")
                f.write(f"    .section_{i} : {{ {obj}(.data) }}\n")
            f.write("}\n")

        subprocess.run(
            [f"{toolchain_prefix}-ld", "-T", ld_path] + obj_files + ["-o", output_elf_path],
            check=True, capture_output=True)

        # Strip non-essential sections (mirrors preload.py convention)
        subprocess.run(
            [f"{toolchain_prefix}-strip", "--remove-section=.comment",
             "--remove-section=.Pulp_Chip.Info", output_elf_path],
            check=True, capture_output=True)

        return addrs
    finally:
        for path in bin_files + obj_files:
            if os.path.exists(path):
                os.remove(path)
        ld_path = f"{output_elf_path}.ld"
        if os.path.exists(ld_path):
            os.remove(ld_path)


def _generate_tilelang_preload_header(var_prefix: str, addrs: Sequence[int]) -> str:
    """Generate a header that defines a pointer table to preloaded HBM addresses."""
    if not addrs:
        return f"void* {var_prefix}Vector[0] = {{}};\n"

    lines = []
    for i, addr in enumerate(addrs):
        lines.append(f"#define {var_prefix.upper()}_{i}_ADDR ((uint64_t)0x{addr:X})")
    ptrs = ", ".join(f"(void*)(uintptr_t)0x{addr:X}" for addr in addrs)
    lines.append(f"void* {var_prefix}Vector[{len(addrs)}] = {{{ptrs}}};")
    return "\n".join(lines) + "\n"


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
        if flat.dtype == np.float16:
            # Upcast to float32 and emit float32_t literals so main.c uses the
            # ISFLOAT32=1 path, which checks whether the source is in L1 or HBM
            # before DMAing.  The ISFLOAT32=0 DMA path does not perform this
            # check and causes a L1->L1 DMA that leaks memory on the simulator.
            ctype = "float32_t"
            elems = ", ".join(f"{float(v)}f" for v in flat.astype(np.float32))
        else:
            elems = ", ".join(_tilelang_numpy_to_c_literal(v) for v in flat)
        retStr += f"{ctype} {var_name}[] = {{{elems}}};\n"

    retStr += f"void* {var_prefix}Vector[{len(names)}] = " + "{" + ", ".join(names) + "};\n"
    return retStr


@dataclasses.dataclass
class TilelangIOBuffer:
    """Descriptor for one I/O buffer in a TileLang network."""
    name: str     # C variable name matching the TVM PrimFunc param
    c_dtype: str  # C type string, e.g. "fp16"
    nbytes: int   # allocation size in bytes
    is_input: bool  # True → input, False → output


_DEFAULT_TILELANG_INCLUDES = [
    "flex_alloc_api.h",
    "flex_runtime_api.h",
    "flex_redmule_api.h",
    "flex_dma_api.h",
    "flex_group_barrier_api.h",
    "flex_types.h",
    "flex_printf_api.h",
    "DeeploySoftHierMath.h",
]


def generateTilelangSoftHierNetworkHeader(
    input_bufs: Optional[Sequence["TilelangIOBuffer"]] = None,
    output_bufs: Optional[Sequence["TilelangIOBuffer"]] = None,
    functionSignature: Optional[str] = None,
) -> str:
    in_bufs = list(input_bufs or [])
    out_bufs = list(output_bufs or [])
    all_bufs = in_bufs + out_bufs

    # Metadata arrays
    n_in = len(in_bufs)
    n_out = len(out_bufs)
    in_bytes_list = ", ".join(str(b.nbytes) for b in in_bufs) if in_bufs else ""
    out_bytes_list = ", ".join(str(b.nbytes) for b in out_bufs) if out_bufs else ""
    # Legacy fallback: if functionSignature is provided without buf descriptors
    if functionSignature is not None and input_bufs is None and output_bufs is None:
        return f"""
#ifndef __DEEPLOY_TILELANG_SOFTHIER_HEADER__
#define __DEEPLOY_TILELANG_SOFTHIER_HEADER__

#include <stdint.h>
#include "flex_types.h"

{functionSignature};

#endif
"""

    return f"""\
#ifndef __DEEPLOY_TILELANG_SOFTHIER_HEADER__
#define __DEEPLOY_TILELANG_SOFTHIER_HEADER__

#include <stdint.h>
#include "flex_types.h"

void RunNetwork(uint32_t core_id, uint32_t numThreads);
void InitNetwork(uint32_t core_id, uint32_t numThreads);

static const uint32_t DeeployNetwork_num_inputs = {n_in};
extern void    *DeeployNetwork_inputs[];
static const uint32_t DeeployNetwork_inputs_bytes[{max(n_in, 1)}] = {{{in_bytes_list}}};
static const uint32_t DeeployNetwork_num_outputs = {n_out};
extern void    *DeeployNetwork_outputs[];

static const uint32_t DeeployNetwork_outputs_bytes[{max(n_out, 1)}] = {{{out_bytes_list}}};

#endif
"""


def generateTilelangSoftHierNetworkImplementation(
    tilelangBody: str,
    input_bufs: Optional[Sequence["TilelangIOBuffer"]] = None,
    output_bufs: Optional[Sequence["TilelangIOBuffer"]] = None,
    functionSignature: Optional[str] = None,
    includeList: Optional[Sequence[str]] = None,
    deployer: Optional[NetworkDeployer] = None,
    bufferInitializationCode: Optional[str] = None,
    globalDefinitionCode: Optional[str] = None,
) -> str:
    resolved_includes = list(includeList) if includeList is not None else list(_DEFAULT_TILELANG_INCLUDES)

    includeStr = ""
    for include in resolved_includes:
        includeStr += f'#include "{include}"\n'
    includeStr += '#include "Network.h"\n'
    includeStr += "#include <stdint.h>\n"
    includeStr += "#include <string.h>\n"

    # Legacy path: no buf descriptors provided
    if input_bufs is None and output_bufs is None:
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

        sig = functionSignature or "void tilelang_main(void)"
        return f"""{includeStr}
{resolved_buffer_init}
{resolved_global_defs}
{sig} {{
{tilelangBody}
}}
"""

    # New path: generate SoftHier-convention Network.c from buf descriptors
    in_bufs = list(input_bufs or [])
    out_bufs = list(output_bufs or [])
    all_bufs = in_bufs + out_bufs

    # HBM global pointer declarations
    hbm_decls = "\n".join(
        f'{b.c_dtype}* {b.name} __attribute__((section(".hbm")));' for b in all_bufs)

    # Metadata arrays
    n_in = len(in_bufs)
    n_out = len(out_bufs)

    in_bytes_list = ", ".join(str(b.nbytes) for b in in_bufs) if in_bufs else ""
    out_bytes_list = ", ".join(str(b.nbytes) for b in out_bufs) if out_bufs else ""

    in_zeros = ", ".join(["0"] * n_in) if n_in else ""
    out_zeros = ", ".join(["0"] * n_out) if n_out else ""

    meta = f"""\
void    *DeeployNetwork_inputs[{max(n_in, 1)}];

void    *DeeployNetwork_outputs[{max(n_out, 1)}];"""

    # InitNetwork: cluster 0 / dm_core allocates HBM buffers
    hbm_allocs = "\n        ".join(
        f'{b.name} = ({b.c_dtype}*)flex_hbm_malloc({b.nbytes});' for b in all_bufs)
    in_assigns = "\n    ".join(
        f"DeeployNetwork_inputs[{i}] = (void*){b.name};" for i, b in enumerate(in_bufs))
    out_assigns = "\n    ".join(
        f"DeeployNetwork_outputs[{i}] = (void*){b.name};" for i, b in enumerate(out_bufs))

    init_fn = f"""\
void InitNetwork(__attribute__((unused)) uint32_t core_id,
                 __attribute__((unused)) uint32_t numThreads) {{
  if (flex_get_cluster_id() == 0) {{
    if (flex_is_dm_core()) {{
      {hbm_allocs}
    }}
    flex_intra_cluster_sync();
    {in_assigns}
    {out_assigns}
  }}
}}"""

    run_fn = f"""\
void RunNetwork(__attribute__((unused)) uint32_t core_id,
                __attribute__((unused)) uint32_t numThreads) {{
{tilelangBody}
}}"""

    return f"""{includeStr}

{hbm_decls}

{meta}

{init_fn}

{run_fn}
"""


def _generate_tilelang_outputs_header(arrays: Sequence[np.ndarray], c_dtype: str = "fp16") -> str:
    """Generate testoutputs.h with OUTPUTTYPE macros and testOutputVector arrays."""
    retStr = f"#define OUTPUTTYPE {c_dtype}\n"
    retStr += "#define ISFLOAT32 1\n"
    retStr += "#define ISOUTPUTFLOAT 0\n"

    names = []
    for idx, arr in enumerate(arrays):
        flat = np.asarray(arr).reshape(-1)
        ctype = _tilelang_numpy_dtype_to_ctype(flat.dtype)
        var_name = f"testOutputVector{idx}"
        names.append(var_name)
        if flat.dtype == np.float16:
            # Upcast to float32_t so main.c reads expected values as float32
            # (ISFLOAT32=1 verification branch: expected = ((float32_t*)...)[i]).
            ctype = "float32_t"
            elems = ", ".join(f"{float(v)}f" for v in flat.astype(np.float32))
        else:
            elems = ", ".join(_tilelang_numpy_to_c_literal(v) for v in flat)
        retStr += f"{ctype} {var_name}[] = {{{elems}}};\n"

    retStr += f"void* testOutputVector[{max(len(names), 1)}] = " + "{"
    retStr += ", ".join(names) if names else "0"
    retStr += "};\n"
    return retStr


def generateTilelangSoftHierTestNetwork(
    tilelangBody: str,
    dumpdir: str,
    input_bufs: Optional[Sequence["TilelangIOBuffer"]] = None,
    output_bufs: Optional[Sequence["TilelangIOBuffer"]] = None,
    functionSignature: Optional[str] = None,
    test_inputs: Optional[Sequence[np.ndarray]] = None,
    test_outputs: Optional[Sequence[np.ndarray]] = None,
    includeList: Optional[Sequence[str]] = None,
    deployer: Optional[NetworkDeployer] = None,
    bufferInitializationCode: Optional[str] = None,
    globalDefinitionCode: Optional[str] = None,
) -> None:
    os.makedirs(dumpdir, exist_ok = True)

    networkHeader = generateTilelangSoftHierNetworkHeader(
        input_bufs = input_bufs,
        output_bufs = output_bufs,
        functionSignature = functionSignature,
    )
    with open(f"{dumpdir}/Network.h", "w", encoding = "utf-8") as f:
        f.write(networkHeader)

    networkImpl = generateTilelangSoftHierNetworkImplementation(
        tilelangBody,
        input_bufs = input_bufs,
        output_bufs = output_bufs,
        functionSignature = functionSignature,
        includeList = includeList,
        deployer = deployer,
        bufferInitializationCode = bufferInitializationCode,
        globalDefinitionCode = globalDefinitionCode,
    )
    with open(f"{dumpdir}/Network.c", "w", encoding = "utf-8") as f:
        f.write(networkImpl)

    # Collect all arrays for preload address computation (inputs first, then outputs)
    inp_list = list(test_inputs or [])
    out_list = list(test_outputs or [])
    all_arrays = inp_list + out_list
    n_inputs = len(inp_list)

    # Generate preload ELF and get HBM addresses for each array
    preload_elf_path = f"{dumpdir}/preload.elf"
    all_addrs = _write_preload_elf(all_arrays, preload_elf_path)
    if all_addrs is None:
        all_addrs = []
        # Remove stale preload files from a previous run (when test switches from
        # preloading to no-preloading).
        for _stale in (preload_elf_path,
                       f"{dumpdir}/preload_reserve.S",
                       f"{dumpdir}/preload_heap_start.txt"):
            if os.path.exists(_stale):
                os.remove(_stale)
    input_addrs = all_addrs[:n_inputs]
    output_addrs = all_addrs[n_inputs:]

    # Write the heap-start address so CMake can define __hbm_heap_start past the
    # preloaded region via --defsym.  This avoids putting a .hbm_reserved NOBITS
    # section in the main ELF whose PROGBITS zero-fill would overwrite the
    # preloaded data.
    if all_arrays:
        last_arr = all_arrays[-1]
        flat = np.asarray(last_arr)
        elem_bytes = 4 if flat.dtype == np.float16 else flat.dtype.itemsize
        heap_start = all_addrs[-1] + flat.size * elem_bytes
        heap_start = (heap_start + 1023) & ~1023  # align 1024
        with open(f"{dumpdir}/preload_heap_start.txt", "w") as f:
            f.write(f"0x{heap_start:X}\n")

    input_header = _generate_tilelang_preload_header("testInput", input_addrs)
    with open(f"{dumpdir}/testinputs.h", "w", encoding = "utf-8") as f:
        f.write(input_header)

    # Generate testoutputs.h with OUTPUTTYPE macros and address-based pointer table
    out_dtype = output_bufs[0].c_dtype if output_bufs else "fp16"
    macros = "#define OUTPUTTYPE {}\n".format(out_dtype)
    macros += "#define ISFLOAT32 1\n"
    macros += "#define ISOUTPUTFLOAT 0\n"
    output_header = macros + _generate_tilelang_preload_header("testOutput", output_addrs)
    with open(f"{dumpdir}/testoutputs.h", "w", encoding = "utf-8") as f:
        f.write(output_header)

    if shutil.which("clang-format"):
        clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.c')
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.h')
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testoutputs.h')
        os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testinputs.h')

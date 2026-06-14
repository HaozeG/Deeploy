# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

macro(add_gvsoc_emulation name)
  set(BINARY_PATH ${CMAKE_BINARY_DIR}/bin/${name})

  set(GVSOC_EXECUTABLE $ENV{SOFTHIER_INSTALL_DIR}/install/bin/gvsoc)

  # Preload ELF support: data arrays are injected into HBM at known addresses
  # to avoid compiling massive C array literals through riscv32-unknown-elf-gcc.
  set(PRELOAD_ARG "")
  if(DEFINED PRELOAD_ELF_PATH)
    if(EXISTS "${PRELOAD_ELF_PATH}")
      set(PRELOAD_ARG --preload ${PRELOAD_ELF_PATH})
    endif()
  endif()

  add_custom_target(gvsoc_${name}
    DEPENDS ${name}
    COMMAND env LD_LIBRARY_PATH=$ENV{SOFTHIER_INSTALL_DIR}/third_party/DRAMSys:$ENV{SOFTHIER_INSTALL_DIR}/third_party/systemc_install/lib64:$ENV{LD_LIBRARY_PATH}
            ${GVSOC_EXECUTABLE}
            --target=pulp.chips.flex_cluster.flex_cluster
            --binary ${BINARY_PATH}
            run
            ${PRELOAD_ARG}
            # --trace-level=6 --trace=/chip/cluster_0/pe0/insn
            --trace=redmule --trace=idma --trace=cluster_registers
            # --trace=redmule --trace=idma --trace=spatz --trace=cluster_registers
            # --trace=/chip/cluster_0/redmule
            # --trace=/chip/cluster_0/idma
            # --trace=/chip/cluster_0/pe0/insn
            # --trace=/chip/cluster_0/pe2/insn
            | tee $ENV{SOFTHIER_INSTALL_DIR}/gvsoc_${name}.log
    COMMENT "Simulating deeploytest with GVSOC"
    USES_TERMINAL
    VERBATIM
  )
endmacro()
# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

enable_language(ASM)

file(GLOB_RECURSE SOURCES
    main.c
)

link_directories(${ProjectId}/../../${GENERATED_SOURCE})

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES} )
target_link_libraries(${ProjectId} PRIVATE network deeploylib)
# RUN WANG: Link math Lib to Generic Target
target_link_libraries(${ProjectId} PRIVATE m)

# Push __hbm_heap_start past preloaded data so the runtime allocator does not
# overlap preloaded arrays.  The heap-start address is written by the Python
# code generator and applied via --defsym to avoid putting a NOBITS section
# in the main ELF (whose PROGBITS zero-fill would overwrite preloaded data).
set(_preload_heap_start_file "${GENERATED_SOURCE}/preload_heap_start.txt")
if(EXISTS "${_preload_heap_start_file}")
  file(READ "${_preload_heap_start_file}" _preload_heap_start)
  string(STRIP "${_preload_heap_start}" _preload_heap_start)
  target_link_options(${ProjectId} PRIVATE "-Wl,--defsym=__hbm_heap_start=${_preload_heap_start}")
endif()

# Preload ELF: test data injected into HBM to avoid compiling large C array literals
set(PRELOAD_ELF_PATH "${GENERATED_SOURCE}/preload.elf")

add_gvsoc_emulation(${ProjectId})

link_compile_dump(${TESTNAME})

# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

from .core import DeeployTestConfig, build_binary, configure_cmake, get_test_paths, run_complete_test, run_simulation
from .core.output_parser import TestResult, parse_numeric_outputs

__all__ = [
    'get_worker_id',
    'create_test_config',
    'run_and_assert_test',
    'verify_numeric_outputs',
    'build_binary',
    'configure_cmake',
    'run_simulation',
]


def get_worker_id() -> str:
    """
    Get the pytest-xdist worker ID for parallel test execution.

    Returns:
        Worker ID string (e.g., 'gw0', 'gw1', 'master' for non-parallel)
    """
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


def create_test_config(
    test_name: str,
    platform: str,
    simulator: Literal['gvsoc', 'banshee', 'qemu', 'vsim', 'vsim.gui', 'host', 'none'],
    deeploy_test_dir: str,
    toolchain: str,
    toolchain_dir: Optional[str],
    cmake_args: List[str],
    tiling: bool = False,
    cores: Optional[int] = None,
    l1: Optional[int] = None,
    l2: int = 1024000,
    default_mem_level: str = "L2",
    double_buffer: bool = False,
    mem_alloc_strategy: str = "MiniMalloc",
    search_strategy: str = "random-max",
    profile_tiling: bool = False,
    plot_mem_alloc: bool = False,
    randomized_mem_scheduler: bool = False,
    profile_untiled: bool = False,
    gen_args: Optional[List[str]] = None,
) -> DeeployTestConfig:

    test_dir = f"Tests/{test_name}"

    gen_dir, test_dir_abs, test_name_clean = get_test_paths(test_dir, platform, base_dir = deeploy_test_dir)

    worker_id = get_worker_id()

    if worker_id == "master":
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / "build_master")
    else:
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / f"build_{worker_id}")

    cmake_args_list = list(cmake_args) if cmake_args else []
    if cores is not None:
        cmake_args_list.append(f"NUM_CORES={cores}")

    gen_args_list = list(gen_args) if gen_args else []

    if cores is not None and platform in ["Siracusa", "Siracusa_w_neureka"]:
        gen_args_list.append(f"--cores={cores}")

    if tiling:
        if l1 is not None:
            gen_args_list.append(f"--l1={l1}")
        if l2 != 1024000:
            gen_args_list.append(f"--l2={l2}")
        if default_mem_level != "L2":
            gen_args_list.append(f"--defaultMemLevel={default_mem_level}")
        if double_buffer:
            gen_args_list.append("--doublebuffer")
        if mem_alloc_strategy != "MiniMalloc":
            gen_args_list.append(f"--memAllocStrategy={mem_alloc_strategy}")
        if search_strategy != "random-max":
            gen_args_list.append(f"--searchStrategy={search_strategy}")
        if profile_tiling:
            gen_args_list.append("--profileTiling")
        if plot_mem_alloc:
            gen_args_list.append("--plotMemAlloc")
        if randomized_mem_scheduler:
            gen_args_list.append("--randomizedMemoryScheduler")

    if profile_untiled and not tiling and platform == "Siracusa":
        gen_args_list.append("--profileUntiled")

    config = DeeployTestConfig(
        test_name = test_name_clean,
        test_dir = test_dir_abs,
        platform = platform,
        simulator = simulator,
        tiling = tiling,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = toolchain,
        toolchain_install_dir = toolchain_dir,
        cmake_args = cmake_args_list,
        gen_args = gen_args_list,
    )

    return config


def verify_numeric_outputs(
    result: TestResult,
    expected_arrays: List[np.ndarray],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    """Verify simulation output values against reference arrays via ``np.allclose``.

    Parses ``DEEPLOY_OUT[buf][idx]=value`` lines emitted by ``main.c`` from the
    simulation stdout, reconstructs the actual output arrays, and compares them
    element-wise against ``expected_arrays`` using ``np.allclose(rtol, atol)``.

    Args:
        result:          :class:`TestResult` returned by ``run_simulation`` or
                         ``run_complete_test``.
        expected_arrays: Reference arrays (one per output buffer).  Each array
                         is flattened and cast to ``float32`` before comparison.
        rtol:            Relative tolerance (default 1e-2, appropriate for fp16).
        atol:            Absolute tolerance (default 1e-2, appropriate for fp16).

    Raises:
        AssertionError: If no ``DEEPLOY_OUT`` values were found in stdout, or if
                        any output buffer fails ``np.allclose``.
    """
    actual_arrays = result.output_arrays or parse_numeric_outputs(result.stdout)
    assert actual_arrays, "No DEEPLOY_OUT values found in simulation stdout"
    for i, (actual, expected) in enumerate(zip(actual_arrays, expected_arrays)):
        exp_flat = np.asarray(expected).ravel().astype(np.float32)
        abs_err = np.abs(actual - exp_flat)
        print(f"Output buffer {i}: max_err={np.max(abs_err):.4e}, mean_err={np.mean(abs_err):.4e}")
        assert np.allclose(actual, exp_flat, rtol = rtol, atol = atol), (
            f"Output buffer {i} failed np.allclose(rtol={rtol}, atol={atol}): "
            f"max_err={np.max(abs_err):.4e}, mean_err={np.mean(abs_err):.4e}")


def run_and_assert_test(test_name: str, config: DeeployTestConfig, skipgen: bool, skipsim: bool) -> None:
    """
    Shared helper function to run a test and assert its results.

    Raises:
        AssertionError: If test fails or has errors
    """
    result = run_complete_test(config, skipgen = skipgen, skipsim = skipsim)

    assert result.success, (f"Test {test_name} failed with {result.error_count} errors out of {result.total_count}\n"
                            f"Output:\n{result.stdout}")

    if result.error_count >= 0:
        assert result.error_count == 0, (f"Found {result.error_count} errors out of {result.total_count} tests")

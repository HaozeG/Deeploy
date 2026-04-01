# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TestResult:
    success: bool
    error_count: int
    total_count: int
    stdout: str
    stderr: str = ""
    runtime_cycles: Optional[int] = None
    output_arrays: List[np.ndarray] = field(default_factory = list)


def parse_test_output(stdout: str, stderr: str = "") -> TestResult:

    output = stdout + stderr

    # Look for "Errors: X out of Y" pattern
    error_match = re.search(r'Errors:\s*(\d+)\s*out\s*of\s*(\d+)', output)

    if error_match:
        error_count = int(error_match.group(1))
        total_count = int(error_match.group(2))
        success = (error_count == 0)
    else:
        # Could not parse output - treat as failure
        error_count = -1
        total_count = -1
        success = False

    runtime_cycles = None
    cycle_match = re.search(r'Runtime:\s*(\d+)\s*cycles', output)
    if cycle_match:
        runtime_cycles = int(cycle_match.group(1))

    output_arrays = parse_numeric_outputs(output)

    return TestResult(
        success = success,
        error_count = error_count,
        total_count = total_count,
        stdout = stdout,
        stderr = stderr,
        runtime_cycles = runtime_cycles,
        output_arrays = output_arrays,
    )


def parse_numeric_outputs(stdout: str) -> List[np.ndarray]:
    """Parse ``DEEPLOY_OUT[buf][idx]=value`` lines from simulation stdout.

    Returns a list of float32 numpy arrays, one per output buffer, sorted by
    buffer index.  Elements within each buffer are ordered by element index.
    """
    pattern = re.compile(r'DEEPLOY_OUT\[(\d+)\]\[(\d+)\]=([^\s]+)')
    buffers: dict = {}
    for m in pattern.finditer(stdout):
        buf, idx, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
        buffers.setdefault(buf, {})[idx] = val
    result = []
    for buf_id in sorted(buffers):
        d = buffers[buf_id]
        arr = np.array([d[k] for k in sorted(d)], dtype = np.float32)
        result.append(arr)
    return result

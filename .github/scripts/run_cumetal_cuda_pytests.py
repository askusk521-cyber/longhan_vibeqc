"""Run existing CUDA pytest cases one-by-one with hard timeouts.

CuMetal can expose provider/runtime deadlocks that leave one pytest process
alive indefinitely.  Keep every existing CUDA test visible in CI, kill only
the hung subprocess, continue collecting the remaining results, and fail the
lane with a complete summary.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

TEST_FILES = ("tests/python/test_batch.py", "tests/python/test_calculator.py")
TIMEOUT_SECONDS = int(os.environ.get("CUMETAL_CUDA_TEST_TIMEOUT_SECONDS", "300"))


def collect_nodeids() -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        *TEST_FILES,
        "-q",
        "-k",
        "cuda",
        "--collect-only",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)
    nodeids = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    if not nodeids:
        raise SystemExit("no CUDA tests were collected")
    return nodeids


def stream_process(command: list[str]) -> tuple[int | None, str, bool]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    output: list[str] = []

    def pump() -> None:
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(
            f"\nERROR: CUDA pytest exceeded {TIMEOUT_SECONDS}s; killing process group",
            flush=True,
        )
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return_code = None
        process.wait()
    reader.join(timeout=10)
    return return_code, "".join(output), timed_out


def junit_status(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    skipped = [case for case in cases if case.find("skipped") is not None]
    return len(cases), len(skipped)


def main() -> None:
    nodeids = collect_nodeids()
    print(f"Collected {len(nodeids)} existing CUDA tests", flush=True)
    failures: list[str] = []
    combined_output: list[str] = []

    for index, nodeid in enumerate(nodeids):
        print(f"\n::group::CUDA pytest {index + 1}/{len(nodeids)}: {nodeid}", flush=True)
        junit = Path(f"/tmp/vibeqc-cuda-test-{index}.xml")
        command = [
            sys.executable,
            "-m",
            "pytest",
            nodeid,
            "-vv",
            "-s",
            f"--junitxml={junit}",
        ]
        return_code, output, timed_out = stream_process(command)
        combined_output.append(output)
        cases, skipped = junit_status(junit)
        if timed_out:
            failures.append(f"TIMEOUT: {nodeid}")
        elif return_code != 0:
            failures.append(f"FAILED: {nodeid} (exit {return_code})")
        elif cases != 1:
            failures.append(f"INVALID RESULT: {nodeid} produced {cases} testcase(s)")
        elif skipped:
            failures.append(f"SKIPPED: {nodeid}")
        print("::endgroup::", flush=True)

    provenance = "".join(combined_output)
    if "device=apple_gpu" not in provenance:
        failures.append("missing CuMetal provenance: device=apple_gpu")
    if "launch_success=true" not in provenance:
        failures.append("missing CuMetal provenance: launch_success=true")

    if failures:
        print("\nCUDA test failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(f"\nExecuted {len(nodeids)} existing CUDA tests with zero skips")


if __name__ == "__main__":
    main()

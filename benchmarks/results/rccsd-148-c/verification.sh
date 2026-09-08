#!/usr/bin/env bash
set -euo pipefail
cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-148-a
export PYTHONPATH=.:python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/build/cpu/libvibeqc.so"
mkdir -p build/cc-c-final
.venv/bin/python -c 'import json,hashlib;from pathlib import Path;d=json.loads(Path("build/c-snapshot.json").read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in d["files"].items());print("All source hashes match")' > build/cc-c-final/source-check.log
.venv/bin/python -m pytest tests/python/test_cc*.py tests/python/test_tensor_ir.py tests/python/test_tensor_execution.py tests/python/test_tensor_examples.py tests/python/test_posthf_reference.py tests/python/test_posthf_providers.py tests/python/test_validation.py -q > build/cc-c-final/pytest.log 2>&1
.venv/bin/python -m tools.validate_cc_solver --output build/cc-c-final/endpoints > build/cc-c-final/endpoints.log 2>&1
.venv/bin/ruff check tools/vibeqc_cc tools/cc_endpoint_fixtures.py tools/generate_cc_endpoints.py tools/replay_ccsd.py tools/validate_cc_solver.py tests/python/test_cc*.py > build/cc-c-final/ruff.log 2>&1
.venv/bin/ruff format --check tools/vibeqc_cc tools/cc_endpoint_fixtures.py tools/generate_cc_endpoints.py tools/replay_ccsd.py tools/validate_cc_solver.py tests/python/test_cc*.py >> build/cc-c-final/ruff.log 2>&1
printf 'PASS\n' > build/cc-c-final/status

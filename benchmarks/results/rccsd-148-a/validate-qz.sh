#!/usr/bin/env bash
set -euo pipefail
cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-148-a
export PYTHONPATH=.:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/build/cpu/libvibeqc.so"
mkdir -p build/cc-validation
sha256sum -c build/cc-SHA256SUMS > build/cc-validation/source-verification.log
.venv/bin/python -m pytest tests/python/test_cc_equations.py tests/python/test_cc_references.py tests/python/test_cc_provenance.py tests/python/test_tensor_ir.py tests/python/test_tensor_execution.py tests/python/test_tensor_examples.py tests/python/test_posthf_reference.py tests/python/test_posthf_providers.py tests/python/test_validation.py -q > build/cc-validation/pytest.log 2>&1
.venv/bin/ctest --test-dir build/cpu --output-on-failure > build/cc-validation/ctest.log 2>&1
.venv/bin/ruff check tools/vibeqc_cc tools/generate_cc_references.py tools/validate_cc.py tests/python/test_cc*.py > build/cc-validation/ruff.log 2>&1
.venv/bin/ruff format --check tools/vibeqc_cc tools/generate_cc_references.py tools/validate_cc.py tests/python/test_cc*.py >> build/cc-validation/ruff.log 2>&1
.venv/bin/python -m tools.validate_cc --output build/cc-validation/equations --references tests/reference_data/cc/rccsd-a.json > build/cc-validation/evidence.log 2>&1
.venv/bin/python -m pip check > build/cc-validation/pip-check.log 2>&1
printf 'PASS\n' > build/cc-validation/status

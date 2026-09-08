import hashlib
import json
from pathlib import Path

import numpy as np

repo = Path(
    "/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-148-a"
)
first = repo / "build/cc-endpoints-first"
second = repo / "tests/reference_data/cc/endpoints"
results = []
for name in ("h2", "he", "h2o", "nh3", "ch4"):
    meta1 = json.loads((first / f"{name}.json").read_text())
    meta2 = json.loads((second / f"{name}.json").read_text())
    assert meta1["inputs_hash"] == meta2["inputs_hash"], name
    assert meta1["arrays_hash"] == meta2["arrays_hash"], name
    assert meta1["generator_sha256"] == meta2["generator_sha256"], name
    with (
        np.load(first / f"{name}.npz", allow_pickle=False) as a,
        np.load(second / f"{name}.npz", allow_pickle=False) as b,
    ):
        assert set(a.files) == set(b.files), name
        for key in a.files:
            assert a[key].dtype == b[key].dtype, (name, key)
            assert np.array_equal(a[key], b[key]), (name, key)
        count = len(a.files)
    difference = abs(meta1["total_energy"] - meta2["total_energy"])
    assert difference <= 1e-12, name
    results.append(
        {
            "case": name,
            "arrays_identical": True,
            "array_count": count,
            "energy_error": difference,
            "inputs_hash": meta2["inputs_hash"],
            "arrays_hash": meta2["arrays_hash"],
            "first_npz_sha256": hashlib.sha256(
                (first / f"{name}.npz").read_bytes()
            ).hexdigest(),
            "second_npz_sha256": hashlib.sha256(
                (second / f"{name}.npz").read_bytes()
            ).hexdigest(),
        }
    )
report = {"first": str(first), "second": str(second), "cases": results, "passed": True}
(repo / "build/cc-c-final/reference-stability.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "cases": len(results),
            "all_arrays_identical": True,
            "maximum_energy_error": max(r["energy_error"] for r in results),
        }
    )
)

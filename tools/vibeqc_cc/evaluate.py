"""Conventional post-HF provider boundary for fixed-amplitude CPU evaluation."""

from hashlib import sha256

import numpy as np

from tools.vibeqc_posthf import MOBlock
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_tensor import execute
from tools.vibeqc_validation.schema import canonical_hash

from .equations import BLOCKS, build_program


def evaluate(snapshot, provider, t1, t2, *, max_bytes=256 << 20):
    """Return energy and physical T1 residual, never an iteration update.

    Snapshot validity is owned by #147. The provider retains integral blocks
    under its separate budget; max_bytes bounds TensorIR logical buffers, not
    total process RSS or combined provider/executor storage. No dense AO/MO N^4
    tensor is requested. This A-slice CPU reference exposes no T2 residual.
    """
    if not isinstance(provider, ConventionalProvider) or provider.backend != "cpu":
        raise ValueError("RCCSD A requires a conventional CPU integral provider")
    if provider.snapshot.identity != snapshot.identity:
        raise ValueError("provider and reference identities do not match")
    program = build_program(snapshot.nocc, snapshot.nmo - snapshot.nocc)
    f = snapshot.coefficients.T @ snapshot.fock @ snapshot.coefficients
    o = snapshot.nocc
    feeds = {"foo": f[:o, :o], "fov": f[:o, o:], "fvv": f[o:, o:], "t1": t1, "t2": t2}
    for name in BLOCKS:
        result = provider.get(MOBlock.from_spaces(snapshot, name))
        if (
            result.reference_id != snapshot.identity
            or result.hamiltonian_id != snapshot.hamiltonian_id
        ):
            raise ValueError("integral block identity does not match reference")
        feeds[name] = result.to_host()
    outputs = execute(program, feeds, max_bytes=max_bytes).outputs
    energy = float(outputs["correlation_energy"])
    return {
        "reference_energy": snapshot.reference_energy,
        "correlation_energy": energy,
        "total_energy": snapshot.reference_energy + energy,
        "singles_residual": outputs["singles_residual"],
        "diagnostics": outputs,
        "reference_id": snapshot.identity,
        "hamiltonian_id": snapshot.hamiltonian_id,
        "equation_hash": program.logical_hash,
        "inputs_hash": canonical_hash(
            {
                k: sha256(np.ascontiguousarray(v, dtype="<f8").tobytes()).hexdigest()
                for k, v in feeds.items()
            }
        ),
        "scope": "fixed-amplitude energy/T1 only; not a converged CCSD result",
    }

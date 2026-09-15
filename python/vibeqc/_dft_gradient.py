"""Internal stationary KS derivative contracts for Issue #163.

This module validates method state and derivative ownership. It does not
assemble a complete molecular gradient or enable public DFT forces.
"""

from dataclasses import dataclass, field

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash

_METHODS = ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks")
_ARRAY_TOLERANCE = 2e-10
_RESIDUAL_TOLERANCE = 1e-8


@dataclass(frozen=True)
class StationaryKsIdentity:
    """Exact method, topology, provider and successful-solve identity."""

    method: str
    model_identity: str
    geometry_identity: str
    basis_identity: str
    overlap_identity: str
    grid_identity: str
    topology_identity: str
    functional_identity: str
    regularization_identity: str
    provider_identity: str
    owner: int
    solve_epoch: int
    density_generation: int
    fock_generation: int
    orbital_generation: int

    def __post_init__(self):
        if self.method not in _METHODS:
            raise ValueError("stationary derivatives support LDA/PBE RKS/UKS only")
        for name in (
            "model_identity",
            "geometry_identity",
            "basis_identity",
            "overlap_identity",
            "grid_identity",
            "topology_identity",
            "functional_identity",
            "regularization_identity",
            "provider_identity",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"stationary identity requires nonempty {name}")
        for name in (
            "owner",
            "solve_epoch",
            "density_generation",
            "fock_generation",
            "orbital_generation",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"stationary identity requires positive {name}")

    def to_payload(self):
        return {
            "schema": "vibeqc.stationary-ks-state/v1",
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }


@dataclass(frozen=True, eq=False)
class StationaryKsState:
    """Detached physical KS state required by a stationary first derivative."""

    identity: StationaryKsIdentity
    density: np.ndarray
    fock: np.ndarray
    coefficients: np.ndarray
    orbital_energies: np.ndarray
    occupations: np.ndarray
    weighted_density: np.ndarray
    overlap: np.ndarray
    physical_residual: float
    successful: bool
    converged: bool
    physical: bool


@dataclass(frozen=True)
class StationaryDerivativeContract:
    """Validate a complete state before any partial gradient is consumed."""

    state_identity: StationaryKsIdentity
    topology_policy: str = "stable-explicit-grid-v1"
    sign: str = "gradient"
    identity: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.state_identity, StationaryKsIdentity):
            raise TypeError("stationary derivative requires a KS state identity")
        if self.topology_policy != "stable-explicit-grid-v1":
            raise NotImplementedError(
                "topology-changing DFT derivatives are unsupported"
            )
        if self.sign != "gradient":
            raise NotImplementedError("partial DFT forces are not a public capability")
        object.__setattr__(self, "identity", canonical_hash(self.to_payload()))

    @property
    def spin(self):
        return (
            "unpolarized" if self.state_identity.method.endswith("rks") else "polarized"
        )

    @property
    def family(self):
        return "lda" if self.state_identity.method.startswith("lda") else "gga"

    def to_payload(self):
        return {
            "schema": "vibeqc.stationary-dft-derivative/v1",
            "state": self.state_identity.to_payload(),
            "spin": self.spin,
            "family": self.family,
            "topology_policy": self.topology_policy,
            "sign": self.sign,
            "held_fixed": "density while evaluating explicit XC geometry partials",
            "force_capability": "unsupported",
        }

    def validate(self, state):
        """Return the exact accepted state; never repair or substitute it."""
        if not isinstance(state, StationaryKsState):
            raise TypeError("expected a stationary KS state")
        if state.identity != self.state_identity:
            raise ValueError("stationary KS identity mismatch")
        if not (state.successful and state.converged and state.physical):
            raise ValueError(
                "stationary derivative requires successful converged physical state"
            )
        if (
            not np.isfinite(state.physical_residual)
            or abs(state.physical_residual) > _RESIDUAL_TOLERANCE
        ):
            raise ValueError("physical residual exceeds the stationary derivative gate")

        spins = 1 if self.spin == "unpolarized" else 2
        overlap = _matrix(state.overlap, "overlap")
        n = len(overlap)
        if not np.allclose(overlap, overlap.T, atol=_ARRAY_TOLERANCE, rtol=0):
            raise ValueError("overlap is not symmetric")
        if np.linalg.eigvalsh(overlap)[0] <= 0:
            raise ValueError("overlap is not positive definite")

        density = _spin_matrices(state.density, spins, n, "density")
        fock = _spin_matrices(state.fock, spins, n, "Fock")
        coefficients = _spin_matrices(state.coefficients, spins, n, "coefficients")
        weighted = _spin_matrices(state.weighted_density, spins, n, "weighted density")
        energies = _spin_vectors(state.orbital_energies, spins, n, "orbital energies")
        occupations = _spin_vectors(state.occupations, spins, n, "occupations")

        maximum = 2.0 if spins == 1 else 1.0
        if np.min(occupations) < 0 or np.max(occupations) > maximum:
            raise ValueError("occupations exceed the supported spin domain")
        eye = np.eye(n)
        for block in range(spins):
            c = coefficients[block]
            if not np.allclose(
                c.T @ overlap @ c, eye, atol=_ARRAY_TOLERANCE, rtol=1e-10
            ):
                raise ValueError("orbitals are not S-orthonormal")
            expected_density = (c * occupations[block]) @ c.T
            if not np.allclose(
                density[block], expected_density, atol=_ARRAY_TOLERANCE, rtol=1e-10
            ):
                raise ValueError("density reconstruction mismatch")
            expected_weighted = (c * (occupations[block] * energies[block])) @ c.T
            if not np.allclose(
                weighted[block], expected_weighted, atol=_ARRAY_TOLERANCE, rtol=1e-10
            ):
                raise ValueError("weighted-density reconstruction mismatch")
            residual = fock[block] @ c - (overlap @ c) * energies[block]
            scale = max(1.0, float(np.max(np.abs(fock[block] @ c))))
            if float(np.max(np.abs(residual))) > _ARRAY_TOLERANCE * scale:
                raise ValueError(
                    "Fock eigen residual exceeds the stationary derivative gate"
                )
        return state


def _matrix(value, name):
    array = immutable(value)
    if np.iscomplexobj(value) or array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be one real square matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite {name}")
    return array


def _spin_matrices(value, spins, n, name):
    array = immutable(value)
    if np.iscomplexobj(value) or array.shape != (spins, n, n):
        raise ValueError(f"{name} must contain every spin matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite {name}")
    if name in ("density", "Fock", "weighted density") and not np.allclose(
        array, array.swapaxes(-1, -2), atol=_ARRAY_TOLERANCE, rtol=0
    ):
        raise ValueError(f"{name} is not symmetric")
    return array


def _spin_vectors(value, spins, n, name):
    array = immutable(value)
    if np.iscomplexobj(value) or array.shape != (spins, n):
        raise ValueError(f"{name} must contain every spin orbital")
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite {name}")
    return array

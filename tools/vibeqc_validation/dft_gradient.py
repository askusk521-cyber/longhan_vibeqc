"""Independent CPU finite differences for stationary explicit XC geometry.

This validation helper rebuilds AO collocation on displaced inputs and only
evaluates the scalar discrete XC energy. It never calls the generated geometry
pullback under test and is not a production molecular-gradient implementation.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from vibeqc._dft_gradient import StableGridMotion
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.spec import FunctionalSpec


@dataclass(frozen=True)
class DirectionalFiniteDifference:
    """Multistep central differences and their most stable adjacent region."""

    steps: tuple[float, ...]
    estimates: tuple[float, ...]
    stable_pair: tuple[int, int]

    @property
    def stable_estimate(self):
        i, j = self.stable_pair
        return 0.5 * (self.estimates[i] + self.estimates[j])

    @property
    def spread(self):
        i, j = self.stable_pair
        return abs(self.estimates[i] - self.estimates[j])


def finite_difference_xc_directional(
    functional,
    basis_arguments,
    points,
    weights,
    density,
    motion,
    *,
    steps=(1e-3, 3e-4, 1e-4),
):
    """Re-evaluate scalar XC energy on independently displaced inputs."""
    if not isinstance(functional, FunctionalSpec):
        raise TypeError("oracle requires a typed XC functional")
    if not isinstance(motion, StableGridMotion):
        raise TypeError("oracle requires stable-grid motion")
    if motion.topology_changed:
        raise ValueError("oracle cannot cross a topology change")
    if not isinstance(basis_arguments, dict) or "atoms" not in basis_arguments:
        raise ValueError("oracle requires reconstructable basis arguments")

    points = _direction_domain(points, motion.points, "point")
    weights = _direction_domain(weights, motion.weights, "weight")
    atoms = tuple(basis_arguments["atoms"])
    centers = np.asarray(motion.centers)
    if (
        np.iscomplexobj(centers)
        or centers.shape != (len(atoms), 3)
        or not np.isfinite(centers).all()
    ):
        raise ValueError("oracle center direction has incompatible domain")
    density = np.asarray(density)
    if np.iscomplexobj(density) or not np.isfinite(density).all():
        raise ValueError("oracle density must be finite and real")

    steps = tuple(float(step) for step in steps)
    if (
        len(steps) < 3
        or any(not np.isfinite(step) or step <= 0 for step in steps)
        or any(a <= b for a, b in pairwise(steps))
    ):
        raise ValueError("oracle requires at least three decreasing positive steps")

    energy = ContractionProgram(functional, "energy")
    estimates = []
    for step in steps:
        values = []
        for sign in (1.0, -1.0):
            moved_atoms = [
                (atom, np.asarray(position) + sign * step * delta)
                for (atom, position), delta in zip(atoms, centers, strict=True)
            ]
            with NativeAO(**{**basis_arguments, "atoms": moved_atoms}) as basis:
                jets = basis.evaluate(
                    points + sign * step * np.asarray(motion.points),
                    energy.contract.ao_order,
                )
            values.append(
                energy.evaluate(
                    jets,
                    density,
                    weights + sign * step * np.asarray(motion.weights),
                )["energy"]
            )
        estimates.append((values[0] - values[1]) / (2 * step))

    differences = np.abs(np.diff(estimates))
    first = int(np.argmin(differences))
    return DirectionalFiniteDifference(
        steps=steps,
        estimates=tuple(float(value) for value in estimates),
        stable_pair=(first, first + 1),
    )


def _direction_domain(value, direction, name):
    array = np.asarray(value)
    delta = np.asarray(direction)
    if (
        np.iscomplexobj(array)
        or np.iscomplexobj(delta)
        or array.shape != delta.shape
        or not np.isfinite(array).all()
        or not np.isfinite(delta).all()
    ):
        raise ValueError(f"oracle {name} direction has incompatible domain")
    return array

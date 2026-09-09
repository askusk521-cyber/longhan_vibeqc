"""Independent tests for the shared UHF response contract and operator."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_response import (
    CudaDFJKBackend,
    DenseAOResponseBackend,
    GMRESOptions,
    UHFReferenceSnapshot,
    UHFResponseOperator,
    solve_many,
)


def _symmetric_eri(seed=179, nbf=4):
    """Build a tiny chemists-ERI tensor with the required permutation symmetry."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(scale=0.2, size=(nbf, nbf, nbf + 1))
    factors = 0.5 * (factors + factors.swapaxes(0, 1))
    return np.einsum("pqP,rsP->pqrs", factors, factors, optimize=True)


def _reference():
    """Create a canonical two-spin fixture with distinct alpha/beta rotations."""
    nbf = 4
    overlap = np.eye(nbf)
    alpha_energies = np.array([-1.20, -0.35, 0.42, 0.91])
    beta_energies = np.array([-0.92, 0.15, 0.58, 1.08])
    return UHFReferenceSnapshot(
        overlap=overlap,
        hcore=np.diag(alpha_energies),
        fock_alpha=np.diag(alpha_energies),
        fock_beta=np.diag(beta_energies),
        coefficients_alpha=np.eye(nbf),
        coefficients_beta=np.eye(nbf),
        orbital_energies_alpha=alpha_energies,
        orbital_energies_beta=beta_energies,
        occupations_alpha=np.array([1.0, 1.0, 0.0, 0.0]),
        occupations_beta=np.array([1.0, 0.0, 0.0, 0.0]),
        reference_energy=-1.0,
        scf_residual=1e-12,
        geometry_hash="uhf-fixture-geometry",
        basis_hash="uhf-fixture-basis",
        generation_id="uhf-fixture-generation",
    )


def _expm_small(matrix, terms=18):
    """Evaluate a tiny rotation exponential without depending on SciPy."""
    result = np.eye(matrix.shape[0])
    term = result.copy()
    for order in range(1, terms + 1):
        term = term @ matrix / order
        result += term
    return result


def _finite_action(reference, backend, vector, step=1e-6):
    """Differentiate the UHF orbital gradient with an independent rotation.

    The fixture supplies canonical Focks.  We therefore construct a rotated
    Fock as that exact reference Fock plus the explicit J/K change, which is
    the first-order UHF stationarity equation independently of how the small
    fixture's reference was produced.
    """
    problem = UHFResponseOperator.build_problem(reference, backend)
    layout = problem.layout
    alpha_generator = layout.generator_matrix("alpha", vector)
    beta_generator = layout.generator_matrix("beta", vector)
    alpha_occ, _ = layout.spaces("alpha")
    beta_occ, _ = layout.spaces("beta")

    def density(coefficients, occupied):
        return coefficients[:, occupied] @ coefficients[:, occupied].T

    base_alpha = density(reference.coefficients_alpha, alpha_occ)
    base_beta = density(reference.coefficients_beta, beta_occ)

    def gradient(sign):
        alpha = reference.coefficients_alpha @ _expm_small(
            -sign * step * alpha_generator
        )
        beta = reference.coefficients_beta @ _expm_small(-sign * step * beta_generator)
        alpha_density = density(alpha, alpha_occ)
        beta_density = density(beta, beta_occ)
        delta_total = alpha_density + beta_density - base_alpha - base_beta
        delta_alpha = alpha_density - base_alpha
        delta_beta = beta_density - base_beta
        coulomb, _ = backend.coulomb_exchange(delta_total)
        _, alpha_exchange = backend.coulomb_exchange(delta_alpha)
        _, beta_exchange = backend.coulomb_exchange(delta_beta)
        alpha_fock = reference.fock_alpha + coulomb - alpha_exchange
        beta_fock = reference.fock_beta + coulomb - beta_exchange
        alpha_gradient = alpha.T @ alpha_fock @ alpha
        beta_gradient = beta.T @ beta_fock @ beta
        alpha_virtual = layout.alpha_virtual
        beta_virtual = layout.beta_virtual
        return layout.pack(
            alpha_gradient[np.ix_(alpha_virtual, alpha_occ)].T,
            beta_gradient[np.ix_(beta_virtual, beta_occ)].T,
        )

    return (gradient(1.0) - gradient(-1.0)) / (2.0 * step)


def test_uhf_matrix_free_action_matches_finite_rotation_and_transpose():
    reference = _reference()
    backend = DenseAOResponseBackend(_symmetric_eri())
    problem = UHFResponseOperator.build_problem(reference, backend)
    operator = UHFResponseOperator(problem, backend)
    rng = np.random.default_rng(174)
    vector = rng.normal(size=problem.dimension)
    finite = _finite_action(reference, backend, vector)
    np.testing.assert_allclose(operator.apply(vector), finite, atol=3e-8, rtol=3e-8)
    left = rng.normal(size=problem.dimension)
    right = rng.normal(size=problem.dimension)
    assert operator.dot_identity(left, right) < 1e-12


def test_uhf_multirhs_and_recycling_reuse_the_shared_krylov_interface():
    reference = _reference()
    backend = DenseAOResponseBackend(_symmetric_eri())
    problem = UHFResponseOperator.build_problem(reference, backend)
    operator = UHFResponseOperator(problem, backend)
    rng = np.random.default_rng(175)
    rhs = rng.normal(size=(problem.dimension, 2))
    options = GMRESOptions(rtol=1e-11, atol=1e-12, restart=6, max_iterations=80)
    sequential = solve_many(operator, rhs, strategy="sequential", options=options)
    recycled = solve_many(operator, rhs, strategy="recycled", options=options)
    assert sequential.converged and recycled.converged
    np.testing.assert_allclose(
        recycled.solution, sequential.solution, atol=2e-9, rtol=2e-9
    )
    assert recycled.results[1].recycled_vectors > 0


def test_uhf_response_rejects_an_unvalidated_cuda_df_backend():
    """The RHF-only CUDA DF plan cannot be promoted as UHF evidence."""
    reference = _reference()
    backend = DenseAOResponseBackend(_symmetric_eri())
    problem = UHFResponseOperator.build_problem(reference, backend)

    cuda_backend = object.__new__(CudaDFJKBackend)
    with pytest.raises(NotImplementedError, match="spin-resolved CUDA"):
        UHFResponseOperator(problem, cuda_backend)


def test_uhf_problem_rejects_stale_spin_reference_even_at_matching_dimension():
    reference = _reference()
    backend = DenseAOResponseBackend(_symmetric_eri())
    problem = UHFResponseOperator.build_problem(reference, backend)
    changed = replace(reference, generation_id="different-uhf-generation")
    other = UHFResponseOperator.build_problem(changed, backend)
    with pytest.raises(ValueError, match="compatibility"):
        problem.assert_compatible(other)

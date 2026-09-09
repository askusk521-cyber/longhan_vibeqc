# Range-separated four-center integrals

The integral compiler distinguishes ordinary `1/r`, long-range
`erf(omega*r)/r`, and short-range `erfc(omega*r)/r` operators. `omega` is a
finite, nonnegative parameter in inverse Bohr. At zero omega, long-range
integrals vanish and short-range integrals become full Coulomb integrals.
Ordinary Coulomb requests require omega zero.

`CoulombKernel` and `four_center_eri_operator` construct the mathematical
identity. Range requests use IntegralIR schema version 2, including the exact
normalized FP64 parameter. Ordinary requests retain schema version 1 and
their existing generated-source identity. Changing either the family or
omega changes the range request's IR and emitted source.

The existing external-weight Hermite DAG generates a scalar and twelve
shell-center derivatives. Omega and Gaussian exponents are held fixed during
nuclear differentiation. Independent shell slots remain separate even when
two slots belong to one physical atom; the consumer accumulates them after
differentiation. Weights are fixed cotangents, including any explicitly
requested exchange coefficient or sign. Their electronic response belongs
to the caller.

The modified moments obey the same chain rule as ordinary Boys moments:

```text
b = omega / hypot(omega, sqrt(rho))
LR_n(T) = integral from 0 to b of u^(2n) exp(-T*u^2) du
SR_n(T) = integral from b to 1 of u^(2n) exp(-T*u^2) du
d moment_n / dT = -moment_(n+1)
```

The native evaluator uses positive interval quadrature through order 13,
which includes the additional derivative moment for public f/f/f/f shells.
Short-range evaluation uses a rational expression for the interval width
and never subtracts full and long-range values. Factored decay and powers
resolve narrow intervals and large Boys arguments. The independent Python
validation oracle uses adaptive SciPy quadrature; SciPy is not a generation
or execution dependency.

`emit_weighted_eri_primitive_header` binds that evaluator to the existing
generated CPU/CUDA scalar arithmetic. Its input weights follow the explicit
component subset order, so a selected f/f/f/f component does not require a
full shell weight array. Subsets contain at most 64 components. Primitive
contraction and normalized public-basis pullbacks remain consumer operations.
These callables are experimental and unscreened. Existing native weighted
record streams and legacy HF emitters reject range-separated requests until
their execution boundary explicitly consumes the range identity.

Native CPU/CUDA tests compare LR and SR separately against Libcint values
and all four center derivatives for psss, dpsp, fsss, and selected f/f/f/f
components. Moment tests cover zero and enormous omega, small and enormous
Boys arguments, translation, and multiple finite-difference steps. No RSH
functional, range-separated DF, Hessian, omega derivative, or performance
promotion follows from this primitive/compiler capability.

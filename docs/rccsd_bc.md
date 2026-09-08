# Complete RCCSD residuals and CPU solver

The A baseline and its reviewed commit remain documented in `rccsd.md`.
Slice B adds `tools.vibeqc_cc.doubles.build_ccsd_program(o,v,form=...)`.
All inputs/energy/singles conventions from A continue unchanged. This remains
an internal conventional all-electron real restricted CPU implementation.

## B: physical doubles and auditable algebra

The doubles projection is the normalized opposite-spin determinant

```text
R2[i,j,a,b] = <Phi_i(alpha),j(beta)^a(alpha),b(beta)| exp(-T) H_N exp(T) |Phi>
|Phi_i(alpha),j(beta)^a(alpha),b(beta)>
    = a†_(a alpha) a_(i alpha) a†_(b beta) a_(j beta) |Phi>
```

Thus R2 has the same simultaneous `ijab↔jiba` symmetry as spatial T2;
it is not separately antisymmetric. At T=0, `R2[i,j,a,b]=(ia|jb)`.
The A coordinate metric applies unchanged to the full R1/R2 vector. Lambda
implementations must map their physical dual projectors to this convention.

`doubles.DEFINITIONS` is an ordered exact-rational inventory traced to pinned
PySCF 2.14.0 `rccsd.update_amps` (CCSD branch, not CC2) and `rintermediates`
`cc_Foo`, `cc_Fvv`, `Loo`, `Lvv`, `cc_Woooo`, `cc_Wvvvv`, `cc_Wvoov`, `cc_Wvovo`.
Source bytes, Apache-2.0 license and attribution remain in A's manifest/NOTICE.
VibeQC expands those definitions with full Fock diagonals restored. No orbital
energy, denominator, level shift or update formula occurs in the inventory.

| Diagnostic | Role |
| --- | --- |
| D01_driving | Bare ovov integral |
| D02_singles_v / D03_singles_o | Singles dressing and simultaneous exchange |
| D04_oo_ladder / D05_vv_ladder | Occupied/virtual ladders contracted with tau |
| D06_virtual / D07_occupied | Full Lvv/Loo contributions and pair permutations |
| D08_ring / D09_exchange / D10_cross | Mixed ring/exchange contractions |

All intermediate definitions and group outputs are replayable TensorIR nodes.
`form='expanded'` distributes every product of sums and explicitly alpha-renames
dummy indices to produce input-only contractions. This is the higher-memory
reference DAG, with no reused intermediate inside residual contractions.
`form='shared'` retains reusable L/W/tau nodes; `form='optimized'` applies #145's
conservative optimizer. Tests compare every diagnostic, not just the final
residual, across forms. `diagnostics=False` retains only energy and full R1/R2
outputs without changing their equations.

With PySCF's actual `D1_ia=eps_i-eps_a-level_shift`, its doubles denominator is
`D2_ijab=D1_ia+D1_jb`. Restoring removed diagonal Fock terms gives
`R2=D2*(updated_t2-t2)`, including **two** virtual level shifts in D2. Reference
generation uses both zero and nonzero shift, with supplied eps different from
diag(F). Production evaluation never calls this PySCF update.

Independent validation includes explicit determinant projections over five
occupied/virtual shapes, per-intermediate pinned PySCF references, and missing
ladder/ring/exchange mutations. All forms preserve the #138 per-element gate
and absolute energy/residual limits of 1e-8/1e-9. Two reference generations
must have identical case hashes. Reproduce with:

```bash
PYTHONPATH=.:python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m tools.generate_cc_references --full --output /tmp/cc-b-1.json
PYTHONPATH=.:python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m tools.generate_cc_references --full --output /tmp/cc-b-2.json \
  --compare /tmp/cc-b-1.json
PYTHONPATH=.:python python -m pytest tests/python/test_cc_doubles.py \
  tests/python/test_cc_doubles_references.py -q
PYTHONPATH=.:python python -m tools.validate_ccsd --output /tmp/cc-b-evidence
```

Slice C solver/endpoints are a separate acceptance step. B alone does not
establish a converged CCSD result, and neither slice introduces GPU/(T)/Lambda
or gradients.

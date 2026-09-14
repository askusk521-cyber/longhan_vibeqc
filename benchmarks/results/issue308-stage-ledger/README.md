# Issue #308 follow-up stage ledger

This increment follows the September 14 analysis in
[issue comment 5661246967](https://github.com/jinzhezenggroup/vibeqc/issues/308#issuecomment-5661246967).
It adds crash-visible phase observations and bounded fixed-density experiments.
It does not establish the full 768-AO VibeQC energy/force endpoint or close
#308/#310/#311/#206.

`historical-followup.zip` retains selected original evidence from Slurm jobs
9531/9534/9535/9536/9537: exact JSON measurements, reproduction sources and
geometry, original hashes, and the first 24-pair sample of the complete raw
three-center generation. Its manifest preserves all original file identities,
source qualifications, and the selected-member restoration audit. Every member
was reread and compared byte-for-byte. Bulk exploratory arrays, binaries and
routine logs remain in the original debug archive, copied from `/tmp` into the
workspace's ignored `.artifacts/issue308-historical/` directory; its path and
SHA-256 are recorded, and reproduction does not depend on that debugging copy.

The complete raw generation in job 9536 took 103.113819448 seconds and generated
452,984,832 values. Only its first 18,432 values were retained and compared to
the earlier row probe. This is neither a complete independent tensor-validation
result nor timing of metric transformation, SCF or forces. The temporary
register interposition in 9531 remains an unpromoted diagnostic. Its four
complete 589,824-value row outputs and the 9535 row-zero output were rechecked
for exact equality during retention.

Current campaign results will be added after the unified build and bounded
experiments finish. The stable reproduction interface is documented in
[the component-trace contract](../../../docs/df_component_trace.md).

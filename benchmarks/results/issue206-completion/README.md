# Issue #206 completion audit

This directory retains compact summaries for the current completion audit. The
raw CUDA/component traces remain in the ignored n5 artifact directory recorded
in `summary.json`; the JSON here is the reviewable, identity-pinned result.

Current native identity:

- source identity: `47d4c417aaa8f3ecaf61e62d96f9c0c54b5900cb20d99938abccac764d68dff3`
- library SHA-256: `8d40c060e65dd0f3bb78209149e91de7da2b68f85ed16d349c58650b0365ad65`
- target: NVIDIA GeForce RTX 5090, sm_120, CUDA 12.9.1

## Acceptance mapping

| Requirement | Current evidence | Result |
| --- | --- | --- |
| 96/192-AO DF energy+force parity and warm endpoint matrix | `summary.json`, current 7/5-repeat endpoint records | Pass |
| 384-AO larger DF endpoint | `summary.json` | Numerical pass; ordinary warm speedup 1.34x |
| 768-AO larger DF endpoint | `summary.json` | Numerical pass; ordinary warm ratio 1.23x, so not a no-slower claim |
| Resident response admission and transfer/work invariants | `resident-768.json` | Pass: resident JK scratch, zero raw panel gather, zero raw upload, packed exact work |
| Intentional constrained/fallback route | `fallback-192.json` | Pass: explicit host-weight control is classified as fallback and completes |
| Direct analytic-force floor | `direct-current.json` | Pass for the retained four-point 7-repeat floor |
| Unequal-auxiliary practical cases | Historical raw records used an older native identity and are not counted as current evidence | Not counted |

This audit does not close #206: the 768-AO endpoint remains slower than the
stock comparator, and the 384-AO automatic streamed low-memory experiment did
not complete within its finite 30-minute allocation. Those negative results are
retained in `summary.json` rather than omitted.

The resident sentinel and timeline runner are exercised by the hardware-free
tests `tests/python/test_issue206_resident_sentinel.py` and
`tests/python/test_issue308_response_timeline.py`.

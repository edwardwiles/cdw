# CM_WINNER_BIN_CROSS = justified_future_task

Task §10 requires this release to preserve (not implement) the winner-bin cross-block (`H_EC`)
optimization opportunity the final-gate continuation session measured and to formally close it as
a disclosed follow-up, not a blocker.

## What was measured (real D=20 data, `bench_cm_bintable_decomposition.jl`, `docs/key_results/cm_bintable_decomposition_2026-07-25.txt`)

D=20, NCORE=382, L=50, W=80,000:

| Component | Time | Share of `build_bin_tables!` | Share of FULL CM Hessian callback |
|---|---|---|---|
| `Stab` (core-CM cross ingredient, O(W·D·NCORE)) | 1.7156s | 80.7% | 68.2% |
| `Ttab` (CM-CM ingredient, O(W·D²)) | 0.1067s | 5.0% | 4.2% |
| combined `build_bin_tables!` | 2.1251s | — | 84.4% |
| `prefix_sum_tables!` | 0.0135s | — | 0.5% |
| full Hessian callback (dense-reference core) | 2.5169s | — | 100% |

`Stab` alone is **68.2% of the entire CM Hessian callback** — well above the 10% materiality bar
that would justify a dedicated redesign. This corrects the first port session's speculative
"small and unquantified" characterization (see [[shared-winner-pair-core-hessian-port-2026-07-25]]).

## Why it is not implemented in this release

This release's mandate is the shared exact winner-pair **H_EE** backend across all four families.
`Stab`/`Ttab` are the **H_EC** (core-CM cross) ingredient, a structurally separate accumulation
(per-draw bin-table construction, not the winner-pair kernel), computed identically regardless of
which H_EE backend is active — replacing dense H_EE with winner-pair H_EE does not touch this cost
at all, which is exactly why CM's whole-callback speedup (task §1 family coverage) is smaller than
unrestricted's: H_EE is a much smaller share of CM's total callback cost.

A winner-bin cross-block redesign is a materially larger, separate engineering effort: it needs
its own accumulation-shape design (how to exploit the bin/winner structure for `Stab` specifically,
since `Ttab` is already cheap), plus its own D=4 exact-correctness gates and D=20 real-scale
validation cycle, mirroring the winner-pair H_EE port's own multi-session validation path. Bundling
it into this release would reopen exactly the kind of larger redesign task §0's instructions
direct this release to avoid ("do not reopen the winner-pair derivation or redesign the shared
interface unless one of the remaining gates exposes a concrete bug").

## Disposition

**`CM_WINNER_BIN_CROSS = justified_future_task`** — quantitatively justified (68.2%, not a rough
guess), not implemented, not required for `SHARED_WINNER_PAIR_H_EE` to reach
`MERGED_ALL_FAMILIES`. A future session should treat the numbers above as its starting budget
(what fraction of the callback is actually addressable) rather than re-deriving them.

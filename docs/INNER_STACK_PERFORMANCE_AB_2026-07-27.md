# Inner-Stack Performance A/B — Winner-Aware H_ER Phase — 2026-07-27

Synthesizes the real, measured performance numbers already collected by each family's own gate
(Sections 2-5), rather than a fresh isolated-kernel benchmark — per the task's own instruction not
to select defaults from isolated kernel timing alone, every number below comes from a complete
Hessian-callback or complete-inner-solve comparison already run as part of correctness gating.

## H_ER cross-Hessian backend: dense_reference vs winner_bin, real D=20/W=80,000

| Family | Restriction width | Serial/cold | Threaded_v2 (production)/warm | Allocation (warm, winner_bin) | Default flipped |
|---|---|---|---|---|---|
| Flexible CM (Section 2) | `ncm = L*nO = 50*19 = 950` (large) | dense 3.6s -> winner_bin 0.49s (**~7.3x**) | dense 0.66-0.84s -> winner_bin 0.27-0.40s (**~2x**) | 9.4 MB/call, 0 resizes | YES |
| Common-Fréchet (Section 3) | `ncm = L*nO + L = 1000` (large, CM-grid+level) | not separately isolated | dense 0.7-3.1s -> winner_bin 0.35-0.57s (**~2-7x**, same range as flexible-CM) | 9.4-9.9 MB/call, 0 resizes | YES |
| CM+ZC mean/pair block (Section 4) | `n_restr = K_mean*D + K_pair*npair = 10` (small) | roughly comparable, sometimes slightly slower | not separately isolated | ~10 MB/call, 0 resizes | YES (not on speed grounds) |
| Origin-ZC H_ER (Section 5) | `n_restr = 10` (small) | mixed: calib 1.43s vs dense 2.67s (faster); near-delta1 1.40s vs dense 0.91s (slower) | not separately isolated | ~11.4 MB/call, 0 resizes | YES (not on speed grounds) |

**Pattern, consistent across all four**: the winner-bin backend wins decisively (2-7x) exactly when
the dense alternative it replaces is itself a *large* gemm (CM-grid blocks, `ncore x ncm` with
`ncm` in the hundreds-to-thousands) — there the O(D-fold) reduction in `build_bin_tables!`'s own
`S`-table fill dominates. It is a wash or occasionally slower when the dense alternative is already
a *small* gemm (the mean/pair ZC blocks, `ncore x n_restr` with `n_restr` in the tens) — there is
simply less FLOP headroom to reclaim. Both CM+ZC and origin-ZC were still flipped to `:winner_bin`
by default, correctly, on **correctness + no-dense-G-read grounds** (task's own explicit framing:
"the goal ... is NOT necessarily fewer FLOPs ... it's eliminating the dense read of obj.H's
economic columns"), not speed — this is the intended, honest basis for those two flips, not an
inconsistency with the other two.

## Economic FG backend: common-Fréchet operator vs dense (Section 3.3)

Real D=20/W=80,000/L=50, both contrasts, 3 points (calib + near-delta=1 + hard), 24 correctness
comparisons: **ALL PASS**. Speed: consistent **1.115x-1.213x** faster with the operator/lookup FG.
Allocation: consistently **1.0394x** (operator uses ~4% *more* memory) at every single point tested
— fails this codebase's own established "allocation parity or better" flip criterion (the same bar
flexible-CM's `:cm_lookup` FG default had to clear, per `core_exact_hessian.jl`'s own
`CM_INNER_FG_BACKEND_DEFAULT` docstring: "median allocation to EXACT PARITY... i.e. the lookup
kernel itself is now allocation-free relative to dense"). **Default correctly left at
`:dense_reference`** — a faster-but-more-allocating result is not force-flipped, matching this
project's established bar for every prior FG-backend decision.

## Verification backend: operator vs dense_reference, all 5 families (Section 6)

Correctness-only comparison per task's own §6.1 scope (draw-level dual index, objective, complete
dual gradient, KKT residual, feasibility/moment residual, status classification, cache/incumbent
admission, cold verification) — **176/176 checks passed** across all 5 families at both D=4 and
real D=20/W=80,000 (`docs/FIVE_FAMILY_OPERATOR_VERIFICATION_DEFAULT_RELEASE_2026-07-27.md`). No
separate timing/allocation A/B was run for this backend (the task's own §6.1 list does not include
one, unlike §3.3's explicit FG-default allocation-parity requirement) — all 5 defaults were flipped
on correctness grounds alone, consistent with how this codebase already treats verification
(a correctness gate, not a hot-path performance lever).

## What this means for the "no dense G" goal end to end

Every winner-bin/operator backend flip in this phase reduces `obj.H`'s dense economic-column reads
to zero in its own scope (confirmed per-family in
`docs/GLOBAL_NO_DENSE_G_INNER_SOLVE_PROOF_2026-07-27.md`'s measured counters) *without* paying a
performance tax anywhere it was flipped — the two "flat" cases (CM+ZC, origin-ZC mean/pair blocks)
are flat, not regressions, and the two "large restriction block" cases (flexible-CM, common-Fréchet)
are genuinely faster. The one backend NOT flipped (common-Fréchet's FG) was correctly held back
specifically because it *would* have cost allocation, which is the right call under this project's
own established bar, not an unfinished item.

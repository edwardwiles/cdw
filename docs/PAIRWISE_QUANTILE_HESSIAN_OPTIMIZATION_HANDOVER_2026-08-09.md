# Handover: optimize the pairwise-quantile-independence restriction's inner Hessian

## Orientation

Worktree: `/bbkinghome/edav/cdw_worktrees/pairwise-quantile-independence-2026-08-09`
Branch: `prototype/pairwise-quantile-independence-2026-08-09` (forked from
`fix/zc-cmzc-exclude-row-k2-k3-2026-08-07` @ `c22d831`)
KNITRO env: `source .knitro_env.sh` from the worktree root before running Julia (pins KNITRO
13.0.1, sets `ZIENA_LICENSE`; this host, `demand.mit.edu`, is the licensed one — confirmed
live). Julia via juliaup (`export PATH="$HOME/.juliaup/bin:$PATH"`), run with
`--project=<worktree root>`.

**Read first, in order:**
1. `docs/PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md` — the restriction's math
   (equivalence proofs, why `:all_cross` not the draft's diagonal-only condition).
2. `docs/PAIRWISE_QUANTILE_STATUS_2026-08-09.md` — full status: what's built, what's validated
   (real KNITRO D4 solve converges, verifier KKT residual ~1e-14, D20/W=100k profiling numbers),
   what bugs were found and fixed getting the real solve working, what's still missing
   (checkpoint/resume entry point).

**This document is the task**: the restriction is CORRECT and VALIDATED end-to-end (real D4
KNITRO solve, real D20/W=100k profiling run) but its inner Hessian callback is slow — 45.7s per
callback at D=20/W=100,000, and the inner solve needs ~9 of them (407s total, dominating the
whole inner solve, consistent with every other restriction family in this codebase). The task now
is to make that Hessian callback fast, without ever reintroducing a dense `W×n_rows` moment
matrix (the one hard constraint carried over from the original task — still absolute).

## Current profile (real D20/W=100,000 numbers, one Hessian callback)

| sub-block | wall-clock | share |
|---|---|---|
| H_EE (unchanged shared winner-pair backend) | 0.125s | 0.3% |
| **T1/T2/T3/T4 raw table build** | **35.80s** | **78.4%** |
| H_MM/MP/PP raw block-fill (reading the tables) | 0.573s | 1.3% |
| centering correction | 0.087s | 0.2% |
| H_E,R cross-block | 3.128s | 6.9% |
| final dense assembly + packing (3502×3502) | 5.950s | 13.0% |

The T1–T4 table build dominates, exactly as the original task anticipated. A user asked ChatGPT
to review this profile and the `pairwise_quantile_hessian.jl` implementation; I've verified the
factual claims below directly against the code (not just trusting the external review) and they
check out. Treat the analysis below as accurate, verified guidance — not as an unvetted external
opinion.

## Verified finding #1: T3/T4 store each unique table 3× redundantly

`PairwiseQuantileOperator`'s `triple_combos`/`quad_combos` (`pairwise_quantile_bin_context.jl`)
are built as:
- `triple_combos`: for every unordered pair `(p,q)` (190 of them) and every other origin `o` (18
  choices), push `(o, pidx)`. Total = `190*18 = 3420`.
- `quad_combos`: for every pair of disjoint unordered pairs `(pidx1<pidx2)`. Total = `3*C(20,4) =
  14535`.

But for a fixed unordered TRIPLE `{o,p,q}`, the three combos `(o;p,q)`, `(p;o,q)`, `(q;o,p)` are
the SAME underlying joint distribution `T_opq(a,b,c) = Σ_w h_w·1{b_o=a,b_p=b,b_q=c}` — just with
axes permuted. There are only `C(20,3)=1140` genuinely distinct triples, not 3420. Same story for
quads: for a fixed unordered quadruple `{a,b,c,d}`, the three pairings `(ab,cd)`, `(ac,bd)`,
`(ad,bc)` used by `quad_combos` are the same `T_abcd(i,j,k,l)`, axis-permuted. Only `C(20,4)=4845`
distinct quadruples exist, not 14535.

`build_pairwise_quantile_hessian_tables!` (`pairwise_quantile_hessian.jl`) loops over the FULL
(redundant) `triple_opq`/`quad_oooo` lists and does the per-draw scatter for every one of them —
i.e. it is currently doing ~3× more scatter work than the information content requires.

**Fix direction**: store tables keyed by the CANONICAL (e.g. sorted) unordered triple/quad only
(`1140`/`4845` entries), built once per Hessian callback exactly as now but over the deduplicated
list. At READ time (`fill_pairwise_quantile_hessian_raw!`'s `H_MP`/`H_PP` block-fill, which already
has to figure out which axis is which origin via `origin_slot`-style logic), permute the axes of
the SAME canonical table depending on which of the 3 (or for pairs, 2) roles the specific
`(marginal-origin, pair)` or `(pidx1,pidx2)` combo needs — this is O(1) index arithmetic per read,
not a rebuild. This alone is a plausible ~3× reduction on the dominant 78% cost.

## Verified finding #2: T3/T4 build is single-threaded

Confirmed directly in `build_pairwise_quantile_hessian_tables!`: T1/T2 go through
`build_pairwise_quantile_tables_threaded!` (the shared `Threads.@threads :static` builder,
`pairwise_quantile_operator.jl`), but T3/T4 are plain serial `for w in 1:W` loops — explicitly
documented in that function's own docstring as a "baseline, correctness-first" choice, deferred
to real profiling data before optimizing. That data now exists. Thread T3/T4 the same way T1/T2
already are (static draw-chunk partition, per-thread scratch, fixed-order `1:nt` reduction — never
atomics, matching every other threaded accumulator in this codebase,
`cm_hessian_threaded.jl::build_bin_tables_threaded!` is the direct precedent).

## Verified finding #3: bin-5 sparsity is not yet exploited for combo enumeration

Only bins 1–4 are active (bin 5's dual is the implicit zero). At the calibration point roughly
80% of each marginal's mass is in bins 1–4 (by construction — 4/5 active bins). A given
triple/quad of origins only contributes a nonzero increment for a draw if ALL of its origins are
in an active bin for that draw. Currently the code loops over every combo for every draw and
checks `bin<=4` per origin inside the combo (correct, but doesn't skip combos where an origin is
already known-inactive before entering the inner index arithmetic). Precomputing, once per outer
point, each draw's list of active origins (`bin[w,o]<=4`) and enumerating only combos WITHIN that
per-draw active set would cut expected work roughly `0.8^3 ≈ 0.51×` for triples and `0.8^4 ≈
0.41×` for quads ON TOP OF the dedup in finding #1 (i.e. combined with dedup: expected quad work
≈ `4845*0.41 ≈ 1985` vs today's `14535`, roughly a 7× reduction). **Verify this 0.8 empirically
against the real calibration point's actual marginal fractions before relying on the estimate** —
it's a reasonable a priori assumption (bins are exactly 20/80 by construction of the marginal
moments) but the ACTIVE-ORIGIN-SET size distribution across draws is worth checking directly
rather than assuming independence gives exactly binomial behavior.

## The big idea: try an exact Hessian-vector product (HVP) instead of building H at all

This is the most promising direction and should be tried FIRST, before investing in T3/T4
micro-optimization — if it works, T3/T4 stop mattering entirely for the inner solve.

**Why it fits this restriction unusually well**: `Hv = (1/M)·G'·diag(h)·(G·v)` for any direction
`v`, where `G`'s columns are this restriction's centered marginal/pair indicator features. But
`G·v` and `G'·w` are EXACTLY `pairwise_quantile_forward!` and `pairwise_quantile_transpose!`
(`pairwise_quantile_operator.jl`) — already built, already validated to machine precision against
dense references (`test_pairwise_quantile_d4_dense_oracle.jl`) and against real finite differences
of the real gradient (`debug_pq_fg_check.jl`). An HVP for the restriction block needs NO T1/T2/T3/T4
tables at all — just one forward pass + one weight multiply + one transpose pass, `O(D+npair)`
per draw, the SAME complexity class as the FG evaluation itself. This should be dramatically
cheaper than 35.8s.

**This is not a novel idea for this codebase** — HVP mode is already a real, working, validated
pattern here for OTHER restriction families. Study these before writing anything new:
- `full_aod_diag/d4_exact/compressed_inner_alt_solvers.jl` — `_callbackEvalHV_inner_compressed!`,
  `inner_loop_KNITRO_compressed_hvp`. This is the cleanest template: `evalRequest.vec` is the
  direction `v` KNITRO wants multiplied; `evalResult.hessVec` is where you write `Hv`; registered
  via the SAME `KN_set_cb_hess` call as a dense Hessian callback — which mode KNITRO actually
  invokes is controlled entirely by the `.opt` file's `hessopt` setting (`1`=dense, `5`=product).
- `full_aod_diag/d4_exact/cm_hessian_architectures.jl:1707` — `_callbackEvalHV_inner_dense!`,
  and the `hvp::Bool` branch in `inner_loop_KNITRO_archgeneric` (~line 1750) showing how an
  existing family switches between dense-Hessian and HVP registration.
- `ek_inner_hvp.opt` (already exists in this repo) — sets `algorithm=cg` (or similar), since
  Hessian-vector-product mode is only compatible with KNITRO algorithms that operate on products
  (CG/Interior-CG), NOT Direct/SQP. This is exactly the algorithm-change tradeoff to benchmark —
  see "what to actually test" below.
- `c9_phase3c_correctness_d4.jl` / `c9_phase3c_d20_bench.jl` — existing A/B harnesses comparing
  dense-Hessian vs HVP for another family, at both D4 (correctness) and D20 (performance). Use
  these as the TEMPLATE for this restriction's own A/B, not a from-scratch design.

**What still needs building, and it's the harder half**: an HVP for the restriction block alone is
not sufficient — KNITRO needs `H·v` for the FULL inner-dual Hessian, `H = [[H_EE, H_E,R];[H_E,R',
H_RR]]`. So `Hv` needs:
- `H_RR·v_R` — the easy part, exactly `(1/M)·g'·diag(h)·(g·v_R)` via
  `pairwise_quantile_forward!`/`transpose!` as above.
- `H_EE·v_E` — needs an HVP-compatible form of the EXISTING winner-pair CES kernel
  (`WinnerPairHessCtx`/`winner_pair_hessian!`, `core_exact_hessian.jl`). This is genuinely new
  research, not a known-good port: check whether `compressed_cc_hvp`
  (`compressed_inner_alt_solvers.jl`, used by the ALREADY-WORKING HVP path for another family) is
  actually THE SAME economic kernel this restriction's `H_EE` already reuses unchanged — if so,
  this may be a very short piece of work (reuse `compressed_cc_hvp` directly); if the winner-pair
  formulation and the "compressed" formulation `compressed_cc_hvp` was built for are different
  code paths that happen to compute the same H_EE, this needs its own derivation from
  `WinnerPairHessCtx`'s fields (`kappa0`, `pi_vec`, `nu`, `y`, `winner`, `target_slot`,
  `Lam_homog`) mirroring however `winner_pair_hessian!` derives the EXPLICIT entries, but for a
  product instead. Do not assume this is trivial — verify by reading `compressed_cc_hvp`'s
  definition and comparing to `winner_pair_hessian!`'s math before promising it's a quick port.
- `H_E,R·v_R + H_E,R'·v_E` — the cross term. Given `H_E,R` is itself `(1/M)·E'·diag(h)·G` (no
  materialized dense form even in the CURRENT explicit-Hessian code — see
  `pairwise_quantile_cross_hessian_block!`), an HVP form should also be reachable by composing the
  economic side's own forward/transpose-style primitives with `pairwise_quantile_forward!`/
  `transpose!` — but this also needs deriving, not assuming.

**What to actually test (per the user's own instinct — this is the real open question, not
whether HVP is "possible")**: build the HVP path, then run a CONTROLLED A/B at the SAME outer
point(s), comparing:
1. Current explicit-Hessian approach (dense-Hessian `hessopt=1`, whatever inner algorithm
   `ek_inner.opt` currently uses).
2. HVP approach (`hessopt=5`, CG/Interior algorithm per `ek_inner_hvp.opt`).

Report, for both: total inner-solve wall-clock, number of FG calls, number of Hessian(-vector)
calls, KNITRO's own iteration count, final `nStatus`, and — critically — pass the SAME converged
point through `verify_inner_solution_operator_pairwisequantile!` for BOTH and confirm the KKT
residual is small for both (not just that one is faster). The real risk (the user's own framing,
worth repeating exactly): switching to a CG-based inner algorithm could need MORE iterations or
converge less reliably than the current direct/SQP-compatible algorithm, even though each
individual callback is far cheaper — only a real, controlled comparison settles whether the net
effect is a win. Do not conclude "HVP is faster" from per-callback cost alone; compare total
solve wall-clock and robustness (does it still reach `nStatus in (0,-100,-101,-103)` at the same
points that currently converge?).

CLAUDE.md's own repo-wide warning is directly relevant here and should govern how findings get
written up either way: **the inner solve's warm/cold start affects speed, never whether it
converges** — if HVP+CG fails to converge at some point where explicit-Hessian succeeded, do not
attribute that to "HVP gives a worse starting point" or similar; find what's actually different
about the algorithm/problem. Likewise Δ*/KKT-residual claims should be verified via the real
verifier, not inferred from `nStatus` alone.

## Two smaller, lower-risk fixes (worth doing regardless of the HVP outcome)

1. **Dense assembly + packing (5.95s, 13% of the callback) is unnecessary overhead even in the
   explicit-Hessian world.** `pairwisequantile_hess_cb_builder` (`pairwise_quantile_production.jl`)
   builds a full `(NCORE+n_rows)×(NCORE+n_rows)` dense `Hfull`, mirrors both triangles, then loops
   again to pack the upper triangle into `evalResult.hess`. This is not the forbidden dense G (no
   `W` dimension), but it's still ~98MB of avoidable memory traffic per callback at D=20. The
   structured blocks (`H_EE` packed, `H_E,R` dense-but-small, `H_MM/MP/PP` dense-but-small) could
   be written DIRECTLY into the packed upper-triangular output in their correct positions, skipping
   the intermediate `Hfull` build/mirror/repack entirely.
2. **`pairwise_quantile_cross_hessian_block!` allocates ~900KB/call** (confirmed:
   `Mtab_S`/`Ptab_S`/`Mtab_Snu`/`Ptab_Snu`/`Mtab_cf`/`Ptab_cf`/`v`/`v_winner_sum` are all built
   fresh every call instead of reusing persistent scratch, unlike every other block in this
   codebase which is zero- or near-zero-allocation). Give it a persistent scratch struct
   (campaign-lifetime shape, like `PairwiseQuantileHessianTables`/`PairwiseQuantileThreadScratch`
   already are for the other blocks). Small in absolute terms at today's scale, but free to fix
   and consistent with the rest of the codebase's discipline.

## Suggested order of work

1. Read the HVP precedent files listed above; determine concretely whether `compressed_cc_hvp` (or
   an equivalent already-existing routine) gives `H_EE·v` for the SAME `WinnerPairHessCtx`
   formulation this restriction's `H_EE` already uses, or whether that needs new derivation.
2. If HVP looks tractable within reasonable effort: build it (restriction block first — trivial,
   reuses existing validated primitives — then the economic block and cross term), validate at D4
   against the D4 dense oracle AND against finite differences of the real gradient (same discipline
   already used for the explicit Hessian: `debug_pq_hess_check.jl` is the template), then run the
   controlled A/B at D20/W=100k described above.
3. If HVP turns out not to pay off (worse net wall-clock, or convergence robustness regresses),
   fall back to fixing T3/T4 directly: dedup (finding #1), thread (finding #2), exploit bin-5
   sparsity (finding #3, but verify the 0.8 assumption empirically first). Re-run
   `profile_pairwise_quantile_d20.jl 100000` (already exists, takes W as a required arg) after each
   fix to confirm real improvement, not assumed improvement.
4. Either way, apply the two smaller fixes (dense packing, cross-block scratch) since they're
   low-risk and independent of the HVP-vs-explicit decision.
5. Update `docs/PAIRWISE_QUANTILE_STATUS_2026-08-09.md` (or a new dated status doc — this
   codebase's convention is to add a new doc per session rather than silently editing history) with
   the real before/after numbers, and push to Dropbox per this repo's standing `CLAUDE.md`
   instruction (`rclone copy <package> "dropbox:Gravity robustness/Analysis/Server Output/<new
   subfolder>"`).

## What NOT to do

- Do not reintroduce a dense `W×n_rows` (or `W×anything-restriction-sized`) matrix anywhere,
  including as HVP scratch — the whole point of the HVP approach is that `pairwise_quantile_
  forward!`/`transpose!` never materialize one, and that property must be preserved.
- Do not conclude HVP is a win from a single per-callback timing comparison — compare total solve
  wall-clock and convergence outcomes across multiple real points, per the discussion above.
- Do not skip the D4 dense-oracle/finite-difference validation step for any new HVP code before
  trusting it at D20 — this is exactly the discipline that caught 4 real bugs in the explicit-
  Hessian path earlier this session (see the status doc's "Bugs found via real KNITRO testing"
  section) and should be expected to catch similar bugs in new HVP code.

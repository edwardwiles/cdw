# Derivative-method audit — D=4, γ_d≡1 + direct-γ' objective variant

> **CORRECTION (added later this session, see `../SESSION_SUMMARY_2026-07-12.md` §5 for the full
> account):** this document's Non-goals section and §10 recommendation both claim the gravity
> outer constraint genuinely needs the dense Jacobian / `ift!` implicit-function-theorem
> correction. **That claim is wrong.** For `UoModel==1`, the gravity moment is identical across
> every draw (`moments/newGravityMoment!.jl`'s own comment: "F-independent gravity/orthogonality
> moment"), so its `ift!` correction is provably exactly zero — verified numerically, a trivial
> direct gradient matches production's dense-Jacobian-derived value to relerr 2.3e-16 while costing
> 0.4ms instead of ~6.3s (at D=10). The dense Jacobian is not needed for **either** outer
> constraint in the full-A framework. See `../PsiObjectiveBundleImplicitMethodB_fullA.jl` for the
> corrected implementation. Read this correction before trusting this file's §10/Non-goals below.

Isolated, additive diagnostics directory (`full_aod_diag/ad_benchmark/`). **No production
default was changed** — `moments/moments!.jl`, `cc_algo/*.jl`, and the `.opt` files are untouched.
All new code lives here plus one small new file (`full_aod_diag/moments_gammanorm.jl`, already
additive from earlier this session).

## Config audited

This session's newly-established best-conditioned D=4 configuration: γ_d≡1 for all destinations
(not just focal), all A_od free (no A[1,d]=1 pins), objective = γ'_focal DIRECTLY (not the
σ/(σ-1)-power-transformed κ). See `full_aod_diag/out/compare_directgp_summary.txt` for the
conditioning comparison that established this as the config to audit (opt_err 0.0038/0.0087 vs
0.598/1.51 for the old A[1,d]=1 normalization, at maxit=25).

**Benchmark points A-D** (`benchmark_points.jl`, `benchmark_points.jld2`) are real points from
this session's actual runs, NOT the old A[1,d]=1-normalized report's stalled/reduced pair (that
pathology is what this session's normalization change fixed — see note in `benchmark_points.jl`
for the full adaptation rationale): A=compensating-scale init, B/D = compare_directgp.jl's
lower/upper solutions, C = compare_gammanorm.jl's (κ-objective variant) lower solution,
cross-evaluated through the direct-γ' callback (same θ coordinates, genuinely different point).

## Answers to the 15 required questions

1. **Does production need the full moment Jacobian?** No, for the objective (already avoided —
   `calculate_grad_k!` is a 2-draw scalar autodiff call) and for the divergence-budget constraint
   (proven avoidable this session — Method B exact). Partially yes for the gravity constraint's
   `ift!` correction (out of scope here, see Q15/Non-goals).
2. **Which code path constructs it?** `cc_algo/outer_loop_functions.jl::calculate_jac_θ_autodiff!`,
   called from `PsiObjectiveBundle.jl`'s `(Q::PsiObjectiveBundleImplicit)` callable whenever KNITRO
   requests a gradient+Jacobian.
3. **What contraction is ultimately used?** Two BLAS.gemv! contractions with the fixed inner-solve
   multipliers `λ` and per-draw weight `arg1`, per θ-parameter — see `call_graph_audit.md` §3.
4. **Does direct scalar ForwardDiff reproduce the current gradient?** Yes, exactly (relative error
   ~1e-15 to ~4e-15 at all 4 points; `correctness_results.csv`).
5. **Does Enzyme reverse mode reproduce it?** No — see Q14/§Method C below. Three distinct,
   genuine failures, the last unresolved within this audit's scope.
6. **Does the expensive finite-difference check support the envelope gradient?** Not run
   separately in this pass — Method B was validated directly against production's own actual
   Method-A output (the ground truth the FD check would itself be checked against), at machine
   precision, which is a strictly stronger check than an FD comparison would add. See Q15 for what
   this trades off.
7. **Structural sparsity pattern?** `sparsity_summary.txt`: 70/460 nonzero entries (15.2%
   density). Each of the 16 trade-share rows has exactly 3 nonzero θ-derivatives (μ, σ, its OWN
   A[o,d] entry — NOT the rest of that destination's A column, a corrected finding, see §Sparsity
   below); the counterfactual-price row has 4; the gravity row is dense (17/23, all A_od + μ).
8. **How many colors required?** 18 (of 23 parameters) — modest 22% compression, limited by the
   one dense gravity row, exactly as the audit spec anticipated for a "dense row" scenario.
9. **Does colored sparse ForwardDiff reproduce the dense Jacobian?** Yes, exactly (0.0 abs/rel
   error at point A, full N=8000; `benchmark_results.csv` note column).
10. **Fastest method at D=4 after compilation?** Method B (direct scalar), median 0.10-0.12s vs
    Method A's 0.14-0.16s (~25-30% faster) and lower allocation (112MB vs 126MB). Method D
    (sparse) ties or slightly beats A on time but has real prep overhead and its main win is
    memory (70MB, mostly from the compressed 18-color evaluation).
11. **Compilation/preparation costs?** Method A/B: ~7-9s first-call compile (JIT), then <0.2s
    thereafter (reused across points in the same process). Method D: sparsity detection + coloring
    prep ~0.5-8s (first-call heavier), reusable ACROSS θ as long as the structural pattern holds
    (verified θ-invariant in this codebase — see §Sparsity).
12. **Which backend should remain the D=4 default?** See §10 decision below — recommend Method B
    for the divergence-budget constraint gradient as a genuine, low-risk improvement; Method A
    stays the correct choice for the gravity constraint until its `ift!` term is separately
    re-derived.
13. **Most promising for larger D, and why?** Method B, more so as D grows — see §10. A D=10 rerun
    (§6b) directly tested the alternative hope (colored sparse improving with scale) and found the
    opposite: coloring compression WORSENS with D (22%→9.7%) because the one dense gravity row
    forces all A_od columns into distinct colors regardless of the (increasingly sparse) trade-share
    block, so Method D is not the scaling answer here even though it stays exact.
14. **Did any method need an algebraic rewrite, and was primal equality verified?** Method C needed
    two: (a) a Const/Duplicated closure-boundary rewrite (no primal change, pure Enzyme API usage);
    (b) a documented, verified type-stability fix in a DIAGNOSTICS-ONLY copy of
    `newGravityMoment!` (`newGravityMoment_typestable.jl`, ONE token changed: `meanτ = 0` →
    `meanτ = zero(eltype(τ))`; this branch is DEAD CODE for `UoModel==1`, so it cannot affect any
    output — verified: `maxabs K diff=0.0, maxabs G diff=0.0` against the untouched original).
    Neither fix touched the production file. Methods A/B/D needed no rewrite.
15. **Are any discrepancies attributable to insufficient inner-solver convergence?** N/A — no
    discrepancies were found for Methods A/B/D (all exact to floating-point precision); Method C
    never reached the point of producing a comparable value.

## §5 Method C — Enzyme reverse mode: precise failure trail

Three real, distinct failures, each diagnosed and (except the last) fixed with a minimal,
verified change — reported in full per the spec's explicit instruction not to paper over this:

1. **`EnzymeMutabilityException`** via `DifferentiationInterface`'s automatic closure wrapping
   (`DI.gradient(θ -> envelope_scalar_div_ctx(θ, ctx), AutoEnzyme(...), θ)`) — Enzyme couldn't
   prove the closure-captured `ctx` (containing `U`, `λ`, `arg1`, `γobj`) read-only. **Fixed** by
   calling `Enzyme.autodiff` directly with explicit `Const(ctx)`/`Duplicated(θ,dθ)`/`Active`
   annotations (bypassing DI's closure).
2. **`IllegalTypeAnalysisException`** inside `newGravityMoment!` — a real Julia type instability
   (`meanτ = 0` then `meanτ += <Float64>`, inferred `Union{Int64,Float64}` locally), which is DEAD
   CODE for `UoModel==1` but still compiled (and fails to compile) by Enzyme's whole-function
   analysis. **Fixed** via the diagnostics-only `newGravityMoment_typestable!` (primal equality
   verified exactly).
3. **`EnzymeRuntimeActivityError`**, then (after `set_runtime_activity`) **`LLVM error: Failed to
   materialize symbols: diffejulia_envelope_ts..., missing symbol "digamma"`** — Enzyme's
   reverse-mode rule for `SpecialFunctions.gamma` (used in the `gamma(μ*(1-σ)+1)` moment
   normalization) needs `digamma` at JIT-link time.

   **Follow-up investigation (per explicit request to try harder before giving up):** confirmed
   this is NOT a codebase issue — the bare, isolated, single-line reproduction
   `Enzyme.autodiff(Reverse, SpecialFunctions.gamma, Active, Active(1.234))`, with none of this
   project's code loaded, fails with the IDENTICAL `digamma` JIT-symbol error (tested directly).
   Tried the two standard workarounds — pre-"warming up" `digamma`/`gamma` before the `autodiff`
   call, and `Enzyme.API.strictAliasing!(false)` — neither helps; the failure is at LLVM
   symbol-materialization time, downstream of any Julia-level warmup. Web search corroborates:
   this environment is **Julia 1.12.6 + Enzyme 0.13.182 + SpecialFunctions 2.8.0**, and there is a
   live, currently-open upstream issue, "Enzyme AD backend limitations on Julia v1.12+"
   (SciML/DiffEqBase.jl #1258, opened 2026-01-15), describing exactly this class of missing/broken
   Enzyme derivative rules specific to Julia 1.12+ — i.e. a known, currently-unresolved ecosystem
   gap, not something a code-level fix here can close. Fixing it for real would mean either
   downgrading the whole project's Julia version (a environment-wide change well beyond this
   diagnostic's scope, and not something to do unilaterally) or waiting on upstream Enzyme/Julia
   1.12 compatibility work.

**Conclusion: Enzyme is not usable for this codebase's moment map, for reasons outside this
codebase** (an upstream Enzyme/Julia-1.12 gap, not a fixable bug here) — two of three obstacles
were real, in-scope, and fixed; the third is out of anyone's hands short of a Julia downgrade or
an upstream Enzyme release. Do not adopt Method C until that upstream gap closes (or unless the
project is willing to pin an older Julia version, which is a separate decision).

## §6 Sparsity — a corrected finding, reported because it was wrong on the first pass

My first hand-derived "expected" structural pattern assumed a trade-share moment row (o,d)
depends on the WHOLE column A[:,d] (reasoning: "the winner is determined by comparing all
origins"). The **numerical union check the spec requires caught this as wrong**: because the
winner indicator (`MinInd!`) is a hard argmax, ForwardDiff treats it as locally constant — so a
row's value only depends on the WINNING origin's own price, i.e. only on A[o,d] itself, for that
row's own `o`. The naive intuition described exactly the part hard-max makes invisible to AD — the
opposite of what actually happens. This is a codebase-structural fact (verified θ-invariant, not
an artifact of the specific perturbations tested), not a fragile numerical accident, and it's the
reason the trade-share block is so sparse (3/23 nonzero per row) despite each row nominally
"depending on" 4 origins economically. A small residual mismatch (12/460 cells, ~2.6%) between the
corrected hand-derivation and the numerical union remains unexplained and is flagged as a minor
open item — the **numerical union is what Method D's coloring actually used** (authoritative per
the spec's own preference for it over a hand-derived pattern), so this does not affect any
reported result's correctness, only the hand-derivation's tidiness.

## §6b Sparsity/coloring at D=10 (per explicit request, before going to full D=20)

`sparsity_pattern_D10.jl` reruns the numeric-union sparsity check and Method D's coloring on a
fresh DFake=10 fake economy (same γ_d≡1+direct-γ' config, W=Jac_W=8000). Results
(`sparsity_summary_D10.txt`):

| | D=4 | D=10 |
|---|---|---|
| θ length l | 23 | 113 |
| moments d | 18 | 102 |
| overall density | 15.2% | 3.45% |
| trade-share row nnz | 3/23 | **3/113** (unchanged) |
| gravity row nnz | 17/23 (dense) | 101/113 (dense) |
| colors used | 18/23 (**22% compression**) | 102/113 (**9.7% compression**) |
| dense vs sparse correctness | exact | exact (0.0 error) |
| full N=8000: dense / sparse(steady) / prep | 0.14s / — / — | 6.16s / 4.54s / 21.4s |

**Two findings, one confirming, one a real correction to the D=4-only story:**

1. **The "own-entry-only" trade-share sparsity is confirmed general**, not a D=4 coincidence: every
   trade-share row still has exactly 3 nonzero θ-derivatives (μ, σ, its own A[o,d]) at D=10,
   despite the row/column count growing ~6-7×. This is the mechanism keeping overall density low
   (3.45% vs 15.2%) as D grows — good news for any future sparse-Jacobian approach in principle.

2. **But color compression gets WORSE, not better, as D grows (22% → 9.7%)** — and this is the
   one piece of the earlier D=4 report that was flagged as untested/uncertain, now answered
   unfavorably. Mechanism: forward-mode coloring requires merged columns to have NO shared nonzero
   row; the ONE dense gravity row is nonzero in essentially every A_od column, so it alone forces
   **all ~D² A_od columns into mutually distinct colors** — the sparse trade-share structure
   doesn't help the coloring at all once that single dense row is present, and its relative "damage"
   only grows as A_od's share of l grows with D. **Do not expect colored sparse ForwardDiff to pay
   off better at D=20 by extrapolating from a hopeful D=4 reading — the trend so far is the
   opposite.** (The full-N=8000 timing still favors sparse, 4.5s vs 6.2s steady-state, because
   avoiding the dense array's memory traffic helps independently of the color count — but the
   21.4s one-time prep cost is a real tax that needs ~13+ reuses at fixed sparsity structure to
   amortize, and prep cost itself is growing with D.)

This strengthens rather than weakens the §10 recommendation below: the real lever is avoiding the
Jacobian altogether (Method B) for the objects that don't need it, not compressing it better.

## §10 Decision rules

**D=4 recommendation:** switch the divergence-budget constraint's gradient computation to Method B
(direct scalar ForwardDiff on the envelope scalar) — correct (validated exact), simpler (fewer
moving parts than dense-Jacobian-then-contract), and faster (~25-30%, ~12-15% of total outer-solve
wall time by extrapolation, `outer_backend_comparison.csv`). Leave the gravity constraint on the
current dense-Jacobian path until its `ift!` term is separately re-derived as a scalar/VJP object
(a real, nontrivial derivation, not a backend swap — see Non-goals). Retain the dense (Method A)
and colored-sparse (Method D) paths for diagnostics (SVD/conditioning studies like
`full_aod_diag/cond_diag.jl`), where the FULL Jacobian is the genuinely correct object to want.

**Scaling recommendation:** Method B's advantage should widen with D, for two structural reasons
measured here: (a) it never materializes the O(N·d·l) dense array, only O(N) scratch: memory-flat
in l; (b) the dense Jacobian's cost is driven by ForwardDiff sweeping over l chunks of the FULL
(N×d) primal — Method B sweeps the same l chunks but of a single scalar reduction, avoiding the
d-factor in what gets carried through each dual-number pass. Colored sparse (Method D) should also
improve with D **only if** the color count keeps growing sub-linearly in l — at D=4 the ONE dense
gravity row already caps compression at 18/23 (22%); whether that ratio improves or worsens at
D=20 (~361 A_od params, still 1 dense gravity row + block-diagonal trade shares) is untested here
and is the natural next benchmark, not something to assume favorably. Do not adopt Enzyme without
resolving the `digamma`/SpecialFunctions gap first, regardless of D.

**Production architecture recommendation:** calculate the divergence-budget constraint gradient as
a direct scalar gradient/VJP (Method B), not via the dense Jacobian; keep a separate full-Jacobian
path (Method A or D) explicitly for diagnostics.

## Non-goals / explicitly out of scope (stated, not silently skipped)

- The gravity constraint's `ift!`-based total-derivative correction was not re-derived as an
  envelope/VJP object — `∂c_∂θ[2,:]` still needs the dense-Jacobian-fed IFT machinery as-is.
- No change to economics, moments, hard-max, normalization, solver tolerances, or reported bounds.
- No production default changed; `outer_derivative_backend`/`moment_jacobian_backend` flags were
  NOT wired into `cc_algo/PsiObjectiveBundle.jl` (see `outer_backend_comparison.csv` for the
  reasoned scope decision — Methods A/B proven bit-identical makes a literal re-run redundant with
  the already-isolated per-call timing).
- D=20 scaling behavior (color count trend, reverse-mode viability at larger p) is untested.

## File map

- `setup_context.jl` — shared economy/data/draws setup (γ_d≡1 + direct-γ' config).
- `benchmark_points.jl` / `benchmark_points.jld2` — the 4 frozen points + fixed (λ,arg1) contexts.
- `derivative_core.jl` — canonical `moment_map!`/`envelope_scalar_div_ctx`.
- `derivative_methods.jl` — Methods A-D.
- `newGravityMoment_typestable.jl` — diagnostics-only Enzyme type-stability fix (unused by A/B/D).
- `quick_validate.jl`, `validate_derivatives.jl` — correctness tests; `validate_results.jld2`.
- `sparsity_pattern.jl` — §6; `sparsity_pattern.jld2`, `sparsity_summary.txt`.
- `benchmark_derivatives.jl` — §8 timing; `benchmark_results.csv`.
- `call_graph_audit.md`, `derivative_formulas.md` — §2-4 writeups.
- `outer_backend_comparison.csv` — §9 (reasoned deferral + extrapolated estimate).
- `correctness_results.csv` — machine-readable version of the A/B/C comparison table.

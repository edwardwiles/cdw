# Session summary — normalization, direct-γ' objective, derivative audit, D=10 (2026-07-12)

For a future Claude picking this up cold. Everything here is additive — no production file
(`moments/moments!.jl`, `cc_algo/*.jl`, `.opt` files) was changed. Read this before re-deriving
any of it; several conclusions here **correct or supersede** earlier memory notes (flagged below).

**D=10 Method-B full-A result, and an honest correction to my own projection**: κ bounds are
**bit-identical** to the production run (`[0.056409, 0.267194]`, same status, same objective at
every iteration — exactly as expected, since Method B's gradient was validated exact so KNITRO's
search trajectory cannot differ). Wall time: **3036.4s → 2756.1s, a 9.2% reduction** — real, but
far short of the ~40x I projected in §7 below from a per-FCeval cost model. That projection was
wrong: it assumed the dense Jacobian gets paid on every one of the 214 `outer_FCevals`, but
gradient callbacks fire far less often than value-only callbacks (roughly once per outer
*iteration*, ~25 times, not once per function evaluation) — most of the 214 evaluations are cheap,
inner-solve-only value calls that Method B does nothing for. On top of that, at l=113 a large
share of KNITRO's own wall time is its *internal* interior-point linear algebra (scales with
problem size, orthogonal to which backend supplies gradients) — not our callback cost at all. See
§7 for the corrected accounting. Lesson: a per-call microbenchmark is not a substitute for
measuring the actual full run.

---

## 1. The γ_d≡1 normalization (replaces A[1,d]=1)

**Idea** (user's): instead of pinning `A[1,d]=1` for each destination d (the old normalization,
arbitrary units) and letting `γ_d` float free, pin `γ_d≡1` for *every* destination and let the
*entire* A_od matrix float free. Since γ_d enters `denom[d]=γ[d]^σ*gdp[d]`, the normalizer in
*every* trade-share moment for destination d (`moments/hFunction.jl:37,85`), this is a genuine
reparameterization of the whole moment map, not cosmetic. Same total free-parameter count as
before (γ_θ slots pinned instead of A[1,d] pinned) — verified: same κ point estimate to 16 digits
after a compensating A-column rescale (`s_d = γ0[d]^{-σ/(μHat(σ-1))}`, derived from how the
σ-transformed winning price scales under a uniform column rescale).

**Implementation**: `full_aod_diag/moments_gammanorm.jl` — `EK_moments_gammanorm!` (γ_d≡1, κ
objective unchanged) and `EK_moments_gammanorm_directgp!` (γ_d≡1 **and** direct-γ' objective, see
§2). `build_theta_gammanorm`/`theoretical_gammaprime_bounds` build the θ_initial and bounds.

**Result (D=4, maxit=25, comparing to the OLD A[1,d]=1 normalization)**: opt_err improved
**0.598→0.0039 (lower)** and **1.51→0.14 (upper)** — an 11-150x improvement in how converged the
solver gets within a fixed iteration budget, with far fewer inner solves landing infeasible
(41-54%→14-20%). See `full_aod_diag/out/compare_A1d_norm_summary.txt` vs `compare_gammad_norm_summary.txt`.

## 2. Direct-γ' objective (removes the σ/(σ-1) power transform)

Since κ = 1-(γ'/γ)^{σ/(σ-1)} is a strictly monotone (decreasing, σ>1) transform of γ'_focal alone
(once γ_focal≡1), extremizing γ'_focal *directly* as the KNITRO objective and converting to κ
afterward gives identical bounds, but K becomes **literally linear in θ** (`K = θ[3+D]`, gradient
= a unit vector) instead of a nonlinear power. `EK_moments_gammanorm_directgp!`
(full_aod_diag/moments_gammanorm.jl) and `EK_moments_focal_norm_directgp!`
(sequential_gravity/focal_moments_directgp.jl, for the profiled/reduced model) implement this —
literally one changed line each (`counterVal = γpf` instead of `1-(γpf/γf)^(σ/(σ-1))`).

**SIGN CONVENTION** (easy to get backwards): κ is *decreasing* in γ'_focal, so `find_smallest`
must be swapped relative to the old κ-objective convention to get the same bound, then converted
back via `κ = 1-γ'^(σ/(σ-1))`. See `compare_directgp.jl`/`run_fullA_D10*.jl` for the exact pattern.

**Result (D=4, on top of §1)**: further improvement, especially on the previously-weaker upper
bound: opt_err **0.14→0.0087** (16x). Both bounds now similarly well-converged. This combination
(γ_d≡1 + direct-γ') is **the best-conditioned D=4 configuration found this session** and is what
all subsequent work (derivative audit, D=10 runs) targets.

## 3. Derivative-method audit (full write-up: `full_aod_diag/ad_benchmark/README.md`)

Audited whether production's dense N×(d+2)×l ForwardDiff Jacobian (`calculate_jac_θ_autodiff!`,
`cc_algo/outer_loop_functions.jl`) is actually needed. **Short answer: no, for the objective
(already avoided) and for the divergence-budget constraint** — a direct scalar ForwardDiff
gradient of the envelope scalar (**"Method B"**) reproduces production's actual output to
relerr~1e-15, and is ~19x (D=4) to ~111x (D=10) cheaper than the dense Jacobian. See §5 below for
why gravity *also* doesn't need it — a correction to what this section originally concluded.

**Method C (Enzyme reverse-mode) and Mooncake**: see §6 (Enzyme/Mooncake status) — both compile
after real fixes but currently produce incorrect (NaN) gradients; not usable, and per §7's timing
analysis, not worth pursuing further even if fixed.

**Colored sparse ForwardDiff (Method D)**: exact but the compression ratio *worsens* with D (D=4:
22% fewer colors than dense; D=10: only 9.7%) because the gravity moment row is dense across all
A_od — a single dense row forces most A_od columns into distinct colors regardless of the
otherwise-sparse trade-share block. Not the scaling answer here.

**Sparsity structural finding** (corrected from an initially-wrong hand-derivation, caught by the
numerical check): each trade-share moment row depends on exactly 3 params (μ, σ, its own A[o,d]
entry) — NOT the whole destination column, because the hard-max winner-selection is invisible to
ForwardDiff (a well-known, documented property: away from ties, only the WINNING origin's own
price enters the gradient). Confirmed identical at D=4 and D=10 (`sparsity_summary.txt`,
`sparsity_summary_D10.txt`).

## 4. D=10 profiled/sequential vs full-A comparison (μ FIXED both)

| | free outer params | κ bounds | opt_err (lo/up) | wall |
|---|---|---|---|---|
| **Profiled/sequential** (`sequential_gravity/run_profiled_D10_methodB.jl`) | 12 | [0.0345, 0.3934] | 0.020/0.064 | 54.8 min |
| **Full-A, production Method A** (`full_aod_diag/run_fullA_D10.jl`) | 101 | [0.0564, 0.2672] | 0.142/0.207 | 50.6 min |

κ_max (theoretical ceiling, `1-λ_dd^{1/(σ-1)}`) = **0.5704** for this D=10 fake economy.

Profiled method converges 3-7x better in the same 25-iteration budget and reaches a substantially
wider (likely less biased) interval — full-A's extra unconverged A_od parameters plausibly trap
the solver into an artificially narrow interval, the SAME mechanism `full_aod_diag/report.md`
documented for the old normalization at D=4, now shown to persist at D=10 even with §1-2's fixes.

**Independent re-verification of the profiled solution** (not reusing the search's own
bookkeeping): focal trade-share moments matched to ~1e-14 (exact); divergence budget respected
(0.94, 0.82 ≤ δ=1). Gravity: the run's own *warm-started* re-check reports R_mean well inside
tolerance (4-5e-4), but an independently-written *cold*-started re-inversion gives worse residuals
(1.8e-3, 5.2e-3) — this is a real, known sensitivity (destination inversion is a nonconvex
subproblem; cold vs warm start can land in different local solutions), documented already in
`SEQUENTIAL_GRAVITY_PROGRESS.md §5`, not a new bug. Worth knowing when judging "how robust" a
profiled-method solution is.

## 5. The gravity constraint gradient is TRIVIAL — corrects an earlier wrong claim in this session

**This session initially (wrongly) claimed the gravity outer constraint needs the dense Jacobian
+ `ift!` implicit-function-theorem correction, the same way the divergence-budget constraint's
non-envelope pieces might. This is WRONG, and the user caught it by just asking "why would this
need anything complicated, it's linear in log A_od?"**

For `UoModel==1` (this whole session's config), `newGravityMoment!` computes `sumGrav` as a single
scalar from `τ` (fixed data) and `AodPow` (depends only on μ, A_od — NOT on draws, NOT on F) and
broadcasts the *same* value into every draw's row (`moments/newGravityMoment!.jl`'s own comment:
*"F-independent gravity/orthogonality moment"*). A weighted average of a constant equals that
constant for ANY weighting — so the `ift!` term (which captures how the *weights* change with θ)
is **provably exactly zero** here, not merely small. Verified numerically: a trivial
`ForwardDiff.gradient` of the bare `sumGrav(θ)` scalar (no draws loop, no λ, no arg1 — costs
**0.4ms** at D=10) matches production's dense-Jacobian+`ift!` computation to **relerr 2.3e-16**
(`full_aod_diag/ad_benchmark/verify_gravity_trivial_and_time_inner.log`). The dense Jacobian was
doing ~14,600x more work than necessary for this one constraint.

**Consequence: the full-A framework's dense Jacobian is not needed for EITHER outer constraint.**
Both can be replaced by cheap, exact, closed-form-ish gradients: Method B (divergence) + this
trivial direct gradient (gravity).

## 6. PsiObjectiveBundleImplicitMethodBFullA — the full-A Method B implementation

`full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl` — a diagnostics-only variant of
`PsiObjectiveBundleImplicit` for the case of exactly one extra outer constraint beyond the
divergence budget (`outer_constr_index == d`, i.e. gravity). Computes:
- objective gradient: unchanged (`calculate_grad_k!`, already a direct scalar autodiff call)
- divergence-budget constraint: Method B envelope scalar (same formula/validation as
  `sequential_gravity/PsiObjectiveBundleImplicitMethodB.jl`)
- gravity constraint: the trivial direct gradient from §5 (`make_gravity_grad(γobj, D)`)

**No dense Jacobian is ever constructed.** Validated exactly against production
`PsiObjectiveBundleImplicit` at D=4: inner-solve match, objective-gradient match, and constraint
Jacobian (both rows) relerr = 4.1e-15 (`full_aod_diag/ad_benchmark/test_methodB_fullA2.log`).

Injected into the `CounterfactualSensitivity` module via `CS.include(path)` (NOT a plain
top-level `include`) — this is the correct way to add methods to a module's existing generic
functions (`inner_loop_internal`, `outer_loop_constraints!`, etc. are type-dispatched inside that
module) from an external, additive file without editing any production file. One gotcha hit and
fixed: functions defined in Main (like `withinTransform`, from `misc/doubleDiff.jl`, included at
Main scope by the driver scripts) need `Main.` qualification when called from code injected into
the CS module this way.

Driver: `full_aod_diag/run_fullA_D10_methodB.jl` (same D=10, μ-fixed setup as `run_fullA_D10.jl`,
swapped struct). **Real D=10 result**: κ bounds bit-identical to production
(`[0.056409, 0.267194]`, same status/objective trajectory throughout — expected, since Method B's
gradient is exactly the same number as production's, just computed more cheaply). **Wall time:
3036.4s → 2756.1s, a 9.2% reduction** — real, but far more modest than the naive per-call
projection below suggested. See §7 for why the projection overshot and the corrected accounting.

## 7. Is reverse-mode AD (Enzyme/Mooncake) worth pursuing? — real answer: no, but not for the reason I first projected

Isolated per-call timings (D=4 vs D=10, `full_aod_diag/ad_benchmark/benchmark_scaling_D4_vs_D10.log`):

| | l | Method B (scalar ForwardDiff) | dense Jacobian |
|---|---|---|---|
| D=4 | 23 | 0.0088s | 0.166s |
| D=10 | 113 | 0.057s | 6.33s |

Method B scales roughly *linearly* in l — it does not show the blow-up that would make a
reverse-mode backend's l-independent cost necessary. From these numbers plus a single isolated
inner-solve timing (0.099s at D=10), I projected **~40x** total wall-time savings by extrapolating
"dense Jacobian cost × number of `outer_FCevals`". **The real D=10 run (§6) only showed 9.2%.**
The projection was wrong for a specific, identifiable reason, not a mysterious one: `outer_FCevals`
(214 for the lower bound) counts *value*-only callback invocations (KNITRO's own line-search
trials, each triggering just one cheap inner solve, ~0.099s), not gradient invocations — the
gradient/Jacobian callback fires far less often, roughly once per outer *iteration* (`outer_iters
= 25`, the maxit cap), not once per FCeval. So the dense Jacobian's ~6.3s cost was only ever being
paid ~25 times per bound, not 214 — savings on that piece are more like `25×6.27s ≈ 157s`, in the
right ballpark for the ~137-142s actually observed per bound. The rest of the ~1300-1450s per
bound is KNITRO's own internal interior-point linear algebra over a 113-variable, 2-constraint
problem — overhead our callback timing never captured and that no gradient backend (forward,
reverse, or otherwise) touches.

**Lesson, stated plainly**: a per-call microbenchmark measures the wrong thing when the callback
isn't called as often as you assume, and the solver's own internal cost can dwarf the callback
entirely at some problem sizes. The 9.2% is real and free (same bounds, exact same code otherwise,
no reason not to use `PsiObjectiveBundleImplicitMethodBFullA`'s approach if this framework is used
again) — but if D=10-D=20 full-A wall time needs to come down substantially, the lever is KNITRO's
own per-iteration cost (fewer outer iterations needed, e.g. via better conditioning/starting
points, or a fundamentally smaller search space as in the profiled method — see §4) or reducing
`outer_iters`/FCevals themselves, not the gradient backend.

## 8. Enzyme / Mooncake status (both attempted seriously, neither currently usable)

Full technical trail: `full_aod_diag/ad_benchmark/README.md` §5 and this conversation. Summary:

**Enzyme** (Julia 1.12.6, Enzyme 0.13.182 — the latest released version; main-branch install not
straightforward, network-heavy, abandoned after timeout): three real, distinct compile-time
failures, two fixed:
1. Closure-capture readonly-proof failure — fixed (call `Enzyme.autodiff` directly with explicit
   `Const`/`Duplicated`/`Active` annotations, bypass `DifferentiationInterface`'s auto-closure).
2. `IllegalTypeAnalysisException` from a real (but for our config, dead-code) Julia type
   instability in `newGravityMoment!` (`meanτ = 0` should be `meanτ = 0.0`) — fixed via a
   diagnostics-only one-token patch (`full_aod_diag/ad_benchmark/newGravityMoment_typestable.jl`),
   verified primal-identical.
3. `JIT session error: Symbols not found: [ digamma ]` differentiating `gamma(μ*(1-σ)+1)` —
   **confirmed to be Enzyme.jl issue #2890, currently open upstream** (reproduced the identical
   failure in total isolation, 2 lines, no project code). Fixed via a custom
   `EnzymeRules.@easy_rule` for `SpecialFunctions.gamma` (`full_aod_diag/ad_benchmark/enzyme_gamma_rule.jl`)
   computing the standard closed form `gamma(x)*digamma(x)` in plain Julia — this works because
   `EnzymeSpecialFunctionsExt.jl` handles `gamma` via a broken low-level C++/LLVM path but handles
   `beta_inc` via a working pure-Julia `@easy_rule`; our custom rule follows the `beta_inc` pattern.
   **Gotcha**: `@easy_rule`'s derivative tuples are per-OUTPUT (not "forward tuple, reverse tuple"
   as might be guessed) — `gamma` has one output, needs exactly one tuple.

After all three fixes, Enzyme compiles and runs (113s first-call, ~0.13s steady-state) but
produces **NaN gradients for most parameters** (14/23 entries at D=4). Isolated this to be
**unrelated to hard-max/branching** (identical NaN pattern on the gravity moment alone, which has
zero branching) and **unrelated to the `reshape(vcat(θ[range]),(D,D))` A_od-construction pattern**
(tested directly: rewriting it as a plain preallocated-matrix loop gives the identical NaN
indices). Root cause not identified; would need real time inside Enzyme's reverse-mode internals.

**Mooncake**: fails immediately on ANY threaded code (`Threads.@threads`, used purely for speed in
the moments functions, unrelated to differentiation) — "try/catch/finally blocks are not
supported ... in reverse mode," a documented hard limitation, not a bug. Once tested against a
non-threaded moments variant, Mooncake **also compiles and runs**, and **also produces NaN
gradients**, similar count (13/23) — two independent, actively-developed reverse-mode AD systems
hitting a similar failure pattern is fairly strong evidence of a genuine property of this code
under reverse-mode AD, not two coincidental unrelated bugs.

**Given §7's timing analysis, this was NOT pursued further** — even a fully-fixed reverse-mode
backend would not measurably help, since Method B (already correct, already fast, already
deployed) removes the actual bottleneck.

## Corrections to prior memory notes

- `full-aod-conditioning.md` (this project's earlier memory, from the OLD A[1,d]=1 normalization):
  its "full-A_od outer loop has no cheap rescue" verdict was for a *different, worse-conditioned*
  normalization. This session's γ_d≡1 + direct-γ' + Method-B-for-both-constraints combination
  substantially rescues the full-A approach's per-evaluation cost (§6-7) — though §4's D=10
  comparison still shows the profiled/sequential method converging better per outer iteration at a
  fixed iteration budget, so "which framework wins overall" now depends on total iterations needed,
  not per-iteration cost, which used to be the dominant concern.
- The original derivative audit (`full_aod_diag/ad_benchmark/README.md`, written earlier this
  session) claimed the gravity constraint genuinely needs the dense Jacobian/`ift!` machinery.
  **This is wrong per §5** — flagged there via an addendum; this document is the corrected account.

## File map (this session's additions, beyond what's referenced above)

- `full_aod_diag/moments_gammanorm.jl` — γ_d≡1 moments functions (κ-objective and direct-γ').
- `full_aod_diag/compare_gammanorm.jl`, `compare_directgp.jl` — D=4 A[1,d]=1 vs γ_d≡1 vs direct-γ' comparisons.
- `full_aod_diag/run_fullA_D10.jl`, `run_fullA_D10_methodB.jl` — D=10 full-A, production vs Method-B gradient path.
- `sequential_gravity/focal_moments_directgp.jl` — direct-γ' for the profiled/reduced model.
- `sequential_gravity/PsiObjectiveBundleImplicitMethodB.jl` — Method-B bundle for the profiled model (zero extra outer constraints case).
- `sequential_gravity/run_profiled_D10_methodB.jl` — D=10 profiled run using it.
- `sequential_gravity/verify_D10_solution.jl` — independent moments/divergence/gravity re-check.
- `full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl` — Method-B bundle for the full-A model (exactly one extra outer constraint case; the new, more general struct).
- `full_aod_diag/ad_benchmark/` — the full derivative audit (README.md is the primary write-up; many sub-scripts, see its own file map section).

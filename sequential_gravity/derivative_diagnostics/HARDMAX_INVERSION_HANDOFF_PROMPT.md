Prompt for a new Claude instance (paste this whole thing):

# Task: validate smoothed-inversion solutions against the TRUE hard-max economic model, and fix the hard-max inversion solver

## Where this lives

Repo: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf` (Julia project, branch
`feature/sequential-inversion-perf`). KNITRO only licenses on `demand.mit.edu`:

    export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
    export KNITRODIR=/opt/shared_sw/knitro/14.2.0
    export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
    export PATH="$HOME/.juliaup/bin:$PATH"
    cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
    julia --project=. <script>

D=20 real-data setup needs `FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true` and, for anything
touching the destination-inversion loop in parallel, `julia -t 19` (N >= D-1).

**Note as of this writing:** several overnight background jobs (`sequential_gravity/head_to_head/run_{lc,gc,gu}.jl`)
may still be running in this same repo and can hold the project's `--project=.` Manifest in a
precompile-locked state. If `julia --project=.` hangs or errors on a fresh instantiate, check
`ps aux | grep run_g` / `run_lc` first — you may just need to wait, or work in an isolated scratch
copy of the environment for read-only JLD2 loading.

## The economic/mathematical question

This codebase computes counterfactual trade-share equilibria for a discrete-choice (Eaton-Kortum
style) gravity model. A "focal" country's trade shares are matched by construction using a literal
**hard argmin** (no smoothing at all — see below); the other D-1 "omitted" destinations' trade
shares are matched by *inverting* for a competitiveness vector `u` via a **smoothed softmax**
approximation (temperature `ρ=2e-3`), because that smoothing is what makes the inversion's
Newton/Levenberg-Marquardt solver tractable (differentiable).

The user wants to know: do these smoothed-inversion solutions **also** satisfy the TRUE
(unsmoothed, hard-argmax) economic model — the model that's actually supposed to be governing
the counterfactual? Concretely, for a converged solution `θ = (μ, σ, γ'_focal, A_od[1:D])`:

1. **Focal trade shares**: does `A_od[focal]`, combined with the LFD draw weights `p`, reproduce
   the empirical focal trade-share vector under the TRUE hard-argmin rule? (This part is already
   hard/unsmoothed by construction in the existing code — see "Confirmed already-hard" below — so
   this check should trivially pass unless there's a real bug; it's still worth checking as a
   sanity baseline.)
2. **Non-focal trade shares**: for each of the other D-1 destinations, does the `u` vector obtained
   by **inverting under the TRUE hard-max rule** (not the existing ρ=2e-3 smoothed inversion)
   reproduce that destination's empirical trade shares? This is mechanically guaranteed to hold
   under the SMOOTHED inversion (that's literally what the existing solver enforces) — the open
   question is whether a **hard-max** inversion at the same `(A_od, p)` can be found at all, and
   if so whether it's close to the smoothed solution.
3. **Gravity moment**: does the FULL competitiveness matrix `u_mat` (focal column from step 1,
   all D-1 other columns from step 2's hard-max inversion) satisfy the gravity-equation
   orthogonality moment (`gravity_residual`, see below)? This is a genuine consequence check, not
   a re-derivation — it's evaluated on whatever `u_mat` steps 1+2 produce.

**The user already tried this once** and found that naively setting the existing solver's `ρ=0`
just stalls — makes no real progress, sits there. They strongly suspect the smoothed-inversion's
Newton/LM machinery is fundamentally the wrong tool for the hard-max limit (no usable gradient),
and that a genuinely different algorithm is needed. **Your primary job is to find and validate a
working hard-max inversion method** — this is a real open numerical-methods problem, not a "just
tweak a parameter" task.

## Confirmed already-hard: the focal side

`EK_moments_focal_norm_directgp!` (`sequential_gravity/focal_moments_directgp.jl`) defines the
per-draw winner via a literal loop:

```julia
for ω in 1:W
    best = T(Inf); bo = 1
    for o in 1:D
        price = (wHat[o] * AodPow[o] * τ[o, focal]) * U[ω, o]^μ
        if price < best; best = price; bo = o; end
    end
    ...
end
```

No `ρ`, no softmax, nothing — a strict hard-min comparison. `focal_u(θ)` (a closure defined at the
top of `sequential_gravity/run_profiled_production.jl`, duplicated verbatim across ~10 driver
scripts) is a **closed-form deterministic transform** of `θ` that directly supplies the focal
column of `u_mat` — no solver/inversion at all is needed for the focal side. So step 1 above is
really just "recompute `focal_u(θ)` and check it against the empirical focal shares under rho=0"
— should already hold to numerical precision; treat any large discrepancy as a real bug to report,
not something to "fix" by changing the model.

## The part that needs real work: `invert_destination`

**Location**: `sequential_gravity/profiled_gravity.jl`, function `invert_destination` (grep for
`function invert_destination`).

```julia
function invert_destination(log_x::AbstractMatrix, p::AbstractVector, λ̂::AbstractVector;
                            ref::Int = 1, ρ::Real = 0.0, tol::Real = 1e-10, maxit::Int = 100,
                            ls_iters::Int = 80,
                            u_init::Union{Nothing,AbstractVector} = nothing, verbose::Bool = false)
```

- `log_x[s,o]` (S×D): per-draw log-competitiveness (`build_log_x(Uσ, μ)`, also in `profiled_gravity.jl`).
- `p` (length S): LFD draw weights (sums to 1).
- `λ̂` (length D): the target empirical trade-share column for ONE destination (sums to 1).
- Returns a `DestInversion` struct: `u_full`, `model_shares`, `max_abs_share_error`,
  `objective_value`, `gradient_norm`, `iterations`, `converged::Bool`, plus `stats`.

**What it's actually solving**: for each draw `s`, define `V[s] = ρ·logsumexp_o((u[o]+log_x[s,o])/ρ)`
(a smooth max over origins `o`, temperature `ρ`) and the softmax weight
`W[s,o] = exp((u[o]+log_x[s,o]-V[s])/ρ)`. The destination's model share is
`share[o] = Σ_s p[s]·(normalized weight)·W[s,o]`, which is exactly `∂φ/∂u` of the convex potential
`φ(u) = logsumexp_s(log p[s] + V[s])`. `invert_destination` finds `u` such that `share(u) = λ̂` by
minimizing the convex objective `φ(u) − λ̂·u` (a strictly convex program in `u` for `ρ>0`, solved
via a damped/LM Newton method using the smoothed Hessian `share_jacobian_smoothed`, see the
function body around line 226 of the same file).

**Why `ρ→0` (hard-max) breaks this solver, mathematically**: at `ρ=0`, `share(u)` becomes a
**weighted count of draws currently "won" by each origin** — for any given `u`, the per-draw
winner is a strict argmax, so `share(u)` is **piecewise-constant** in `u`. Its true derivative is
zero almost everywhere and undefined (a jump) exactly at the measure-zero set of `u` where some
draw's winner switches. The code's `ρ=0` branch (same function, search for the `ρ≤0` case) uses
`share_jacobian_closed` — `diag(share) − share·share'`, the **ρ→0 limit of the smoothed Hessian
formula**, NOT the genuine (a.e.-zero) derivative — purely as a heuristic step *direction*, then
does an exact bisection line search along that direction to find a winner-switch "kink." This can
still stall if the current point sits in a wide flat region with no kink in a long stretch of the
search direction (plausible once LFD weights `p` are very concentrated/near-degenerate, which
happens at higher δ per existing findings — see memory `d20-realdata-w-sensitivity` /
`lm-damping-fix-and-gravity-seeding-eval` if you have access to this project's memory system).

**A second, structural reason this might be fundamentally hard, not just numerically fragile**:
`λ̂` (the empirical share target) was itself matched under the ρ=2e-3 SMOOTHED model. Under hard-max,
`share(u)` only takes values in a **finite set** of possible weighted-subset-sums (since it's a
weighted count over a finite, fixed set of `W` draws) — `λ̂` will generically NOT be exactly
achievable by any hard-max partition, only approximable. Confirm or refute this empirically before
assuming it — it directly determines whether you should be solving `share(u)=λ̂` exactly (likely
impossible) or a best-approximation / least-distance problem instead (a genuinely different,
better-posed objective).

**`dest_share`** (`profiled_gravity.jl`, `function dest_share(log_x, logp, u; ρ=0.0)`) already
supports a ONE-SHOT hard-max evaluation (given an already-known `u`, compute what shares it would
produce at ρ=0) — this is used today only as a diagnostic (see
`sequential_gravity/derivative_diagnostics/verify_d20_deltagrid.jl`, which independently
re-verifies saved solutions), never as part of an iterative *solve*. It's your basic building
block for checking candidate `u` vectors, and for computing objective/gradient-direction proxies
if you build a new iterative method.

**IMPORTANT gotcha already paid for once** (documented in `verify_d20_deltagrid.jl`'s own comments,
worth reading in full before writing any new check): two `ρ` conventions coexist in this codebase
and are easy to apply to the wrong side. The **focal** column must always be checked at `ρ=0`
(never smoothed — see above); the **omitted** destinations' EXISTING solutions were solved under
`ρ=2e-3` (so checking a ρ=2e-3-solved `u` against ρ=0 model shares will show an expected O(ρ) gap
that is NOT a bug — informational only, not what you're being asked to fix). Getting this backwards
produced two rounds of spurious 1e-3–3e-3 "errors" in an earlier session. Your new hard-max
inversion is different: you're being asked to find a genuinely NEW `u` that satisfies the shares
AT ρ=0 (not just re-check the existing ρ=2e-3 solution at ρ=0 and expect it to match).

## `gravity_residual` (the moment check for step 3)

`sequential_gravity/profiled_gravity.jl`, `function gravity_residual(u_mat, logτ, logw, σ)`.
`u_mat[o,d]` is the full D×D competitiveness matrix (gauge `u[ref,d]=0` per column). Computes
implied log-productivity `logA[o,d] = logw[o] + logτ[o,d] + u_mat[o,d]/(σ-1)`, two-way
(origin+destination) demeans both `logτ` and `logA`, and returns the gravity-equation
orthogonality residual in three equivalent scalings (`R_sum`, `R_mean = R_sum/D²`,
`R_beta` = elasticity-normalized). `R≈0` is the condition the whole outer-loop optimization
machinery drives toward.

## How to reconstruct (log_x, p, u_mat) from a raw θ vector

Everything lives in `sequential_gravity/run_profiled_production.jl`. The already-validated
template for exactly this kind of standalone reconstruction is
`sequential_gravity/derivative_diagnostics/verify_d20_deltagrid.jl` — read it in full before
writing your own, it already does steps 1-2 (smoothed) for you and is the right pattern to copy
for a "load a saved θ, recompute everything from scratch" script:

```julia
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))   # sets up γ, U, λData, D, W, σ, etc.
# ... load a saved θ (see "Example data" below) ...
log_x = build_log_x(Uσ, θ[1])          # profiled_gravity.jl
uf = focal_u(θ)                          # closure in run_profiled_production.jl -- focal column, no inversion
p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)   # run_profiled_production.jl line ~147
# for each omitted destination d:
inv = invert_destination(log_x, p, λData[:, d]; ref=1, ρ=2e-3, tol=1e-6, maxit=150, ls_iters=50)
umat[:, d] = inv.u_full
R = gravity_residual(umat, logτ, logw, σ).R_mean
```

`λData = Matrix(reshape(γ.P, (D, D))')` and `logτ = log.(τ)`, `logw = log.(wHat)` are all set up
inside `run_profiled_production.jl` itself (same names, module-level).

For your NEW hard-max inversion, you'll replace the `invert_destination(...; ρ=2e-3, ...)` call
with your own solver, called with the SAME `(log_x, p, λData[:,d])` inputs but targeting the hard
`ρ=0` share function instead.

## Example data: 3 real converged points to use as test cases

D=20, W=80000, real trade data (`FAKEDATA=3`). `μ=0.11432339572008998`, `σ=2.5` are IDENTICAL
across all three (same seeded data setup) — only `γ'_focal` and `A_od` differ.

**Point 1** (LC method, target T1, tight divergence budget — δ_budget=0.1):
`sequential_gravity/head_to_head/out_lc/lc_T1_Astar.jld2`, key `best_feasible_theta`
(`Vector{Float64}`, length 23 = `[μ, σ, γ'_focal, A_od(1:20)...]`):
```
[0.11432339572008998, 2.5, 0.9728658401964491, 2.3289881620851266e6, 655511.53246567,
 561181.3830098546, 733028.5149203086, 728209.1747896491, 529630.1278353404, 716766.2260757729,
 602746.505188121, 588395.2545087751, 773457.9874749887, 671494.3069447845, 627010.026157208,
 682326.5995079477, 760014.5629868247, 740214.6657528378, 638062.4123405877, 765140.3232165921,
 726000.1040264533, 689689.8142688492, 678560.9173634152]
```
This is the mildest/easiest point of the three (least A_od movement from the calibrated baseline)
— a good first test case since the hard-max inversion should be least stressed here.

**Point 2** (LU method, target T2, medium budget — δ_budget=1.0): combine `μ,σ` above with
`gammap_target=0.950259956422648` and `best_feasible_Aod` from
`sequential_gravity/head_to_head/out_lu/lu_T2_Astar.jld2`:
```
A_od = [833121.9743984039, 413950.0400661588, 834026.568978347, 835897.3854166459,
 844609.3646075707, 847522.0264575109, 873928.2875236303, 853795.5298877798, 851513.7544376147,
 832999.1932567259, 835632.6050168962, 850481.9861869174, 839246.7387248254, 834281.625021607,
 835008.9269073466, 838452.6880790766, 835525.8614903074, 836601.670586743, 856035.1051116354,
 903894.9913978953]
```

**Point 3** (LU-multistart method, target T3, the hardest/most-stressed target): combine `μ,σ`
above with `gammap_target=0.9440553861575977` and `best_feasible_Aod` from
`sequential_gravity/head_to_head/out_lu_multistart50/lu_ms_T3_pt40.jld2`. **Gotcha**: this file
also has a key literally named `"sigma"` — that is NOT the economic elasticity σ, it's an unrelated
multistart noise-scale parameter (0.1–2.0) from how this starting point was generated. Always use
the fixed `σ=2.5` from the other files/params, never this file's `"sigma"` key.

(Prefer to pull all three fresh from disk rather than retype the numbers above, in case the
underlying head-to-head jobs have overwritten/regenerated them since this was written — check the
`done` field is `true` and re-extract.)

## Suggested approach

1. **Start with ONE point (Point 1 above) and ONE destination** (pick any `d != focal`, e.g. the
   first omitted destination). Reconstruct `(log_x, p)` per the recipe above. Confirm the EXISTING
   ρ=2e-3 `invert_destination` call reproduces `λData[:,d]` to its own stated tolerance (sanity
   check that your reconstruction is correct before touching anything hard-max-related).
2. Evaluate `dest_share(log_x, logp, u; ρ=0.0)` at that same converged `u` (the ρ=2e-3 solution) to
   see how far its HARD-MAX shares already are from `λData[:,d]` — this tells you the size of the
   gap you're trying to close, and whether it's plausibly zero-able or only approximately closeable
   (see the "structural" hypothesis above about finite achievable-share sets).
3. Try alternative algorithms for the single-destination hard-max inversion. Some directions worth
   considering (not prescriptive — use your judgment, this is genuinely open):
   - **Homotopy/continuation**: solve at a sequence of decreasing `ρ` (e.g. 2e-3 → 1e-3 → 1e-4 →
     ... → 0), warm-starting each from the previous solution. The existing solver already works at
     ρ=2e-3; the question is whether it degrades gracefully as ρ→0 or hits a wall.
   - **Bundle/subgradient methods** designed for genuinely nonsmooth convex optimization (the
     ρ=0 objective `φ(u) − λ̂·u` IS still convex, just not differentiable — this is a textbook
     nonsmooth convex program, and there's a large literature/tooling for exactly this, e.g.
     bundle methods, or reformulating as an LP if the structure allows it).
   - **Direct combinatorial construction**: since `share(u)` at ρ=0 only depends on the induced
     partition of draws into "won by origin o" sets, and that partition is determined by the
     RANK ORDER of `u[o]+log_x[s,o]` across `o` for each `s`, consider whether the target partition
     (matching `λ̂`) can be constructed more directly (e.g. via sorting/assignment) rather than via
     gradient-based search over `u`-space.
   - **Canned nonsmooth/derivative-free solvers**: this repo's own convention (see project memory
     if available: "user prefers canned solvers over hand-rolled ones") — check whether
     `NLsolve.jl`, `Optim.jl` (e.g. `NelderMead`, or a proximal-bundle method if available in the
     Julia ecosystem), or a dedicated nonsmooth-optimization package already available in this
     project's `Manifest.toml` could be dropped in rather than hand-rolling a new bisection scheme.
4. Once ONE destination works reliably, generalize to all D-1 omitted destinations, reusing the
   SAME threading pattern already established in `seq_gravcol`'s `invert_all` closure
   (`run_profiled_production.jl`, `PARALLEL_INVERSION`/`Threads.@threads` over destinations) —
   don't reinvent the parallelization, just swap in your new per-destination solver.
5. Once you have a working hard-max inversion for all D-1 destinations at a given `(A_od, p)`,
   run the full 3-step validation (focal check, hard-max non-focal inversion, gravity-moment
   check) on all 3 example points above, and report: did it work? How close were the hard-max
   shares to λ̂ at convergence (if convergence is possible at all)? How far did the hard-max `u`
   end up from the ρ=2e-3 `u`? Does the resulting `u_mat` still satisfy the gravity moment closely?

## A related, possibly-useful side finding from this session (optional context, not required reading)

While producing the example data above, a **separate, unrelated** issue was found in this
session's own post-hoc audit tooling: `exact_inner_divergence_at` (a DIFFERENT function, in
`sequential_gravity/derivative_diagnostics/fixed_A_incumbent.jl`, used to compute an exact δ* for
already-converged points, not related to the hard-max inversion task above) fails with a `1e10`
sentinel specifically at some of the most-improved/most-stressed points (e.g.
`sequential_gravity/head_to_head/out_lc/lc_T2_rand1.jld2` and `..._T3_warm.jld2`,
`out_gc/gc_T2_warm.jld2`) — this is because that audit does a COLD-STARTED (no warm start) KNITRO
dual solve, which apparently struggles from far-from-A* points. This is NOT the same problem as
the hard-max inversion above (it's a different failure of a different, warm-start-less solver on
the SMOOTHED ρ=2e-3 problem), but if you find general insights about robust/warm-started dual
solving in the course of this work that would help that audit too, worth a one-line mention in
your final report (not a required deliverable).

## Deliverable

A written report (new markdown file in `sequential_gravity/derivative_diagnostics/`, matching the
style of `full_d2_correction_report.md`) covering: the method(s) you tried, which worked and which
didn't and why, the final validation results on all 3 example points (or as many as you get
working), and your honest read of whether the smoothed-inversion solutions this whole project has
been producing are trustworthy approximations of the true hard-max economic model, or whether
there's a meaningful, economically-relevant gap. Update the project's persistent memory with key
findings if you have access to that system.

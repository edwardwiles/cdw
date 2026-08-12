# TASK: make `L` mean "L equal-mass buckets" for the production CM and CM+ZC families

Worktree `/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10`, branch
`feature/pq-free-mass-reparam-2026-08-10`. Read the repo `CLAUDE.md` first — all of it applies.

## The goal, stated by the user

> `L = 50` means 50 equally sized buckets. `L = 10` means 10 equally sized buckets. The quantiles
> come in closed form from the Fréchet CDF. No empirical quantiles anywhere.

So at `L = 50` the cutpoints should be at probability levels `k/50` for `k = 1..49`, giving 50
buckets of mass exactly 0.02.

## What I claim is happening now — VERIFY THIS FIRST, DO NOT TAKE IT ON TRUST

I measured the following on 2026-08-12. Reproduce each before changing anything; if any of it is
wrong, the rest of this task is built on sand and you should stop and say so.

**1. CM's thresholds are already closed-form theoretical, and that part is correct.**
`precalc_common_marginals_cdf` (`common_marginals_moments.jl:215`) computes
`z = theoretical_u_threshold.(probs_used)` with `theoretical_u_threshold(p) = -log(1-p)`. That is
the exact Exp(1) quantile, and its docstring derives why it is also the right thing for a Fréchet
`z = U^(-μ)` quantile statement. It replaced an empirical `quantile(U[:,refIndex1], probs)` on
2026-08-05, user-directed. **Nothing to fix here.**

**2. The GRID of probability levels is the problem.** Three different grids exist:

| where | grid | buckets at L=50 |
|---|---|---|
| CM's bare default (`probs === nothing`) | `range(1/L,(L-1)/L,length=L)` | 50 levels, spacing **0.0195918**, not 0.02 |
| what the campaign actually injects | `resolve_cm_probs(L)` → `nested_grid_sequence([10,20,50])[L]` | 50 levels → **51 buckets**, masses only ever **0.015625 or 0.03125** |
| CM+PQ family #7 (`cm_pq_probs_grid`) | `k/G`, `k=1..G-1` | 49 levels → **50 buckets of exactly 0.02** ✔ |

Reproduce the middle row with:

```julia
function nested_grid_sequence(sizes::Vector{Int})   # copy of nested_quantile_grids.jl:58
    nmax = maximum(sizes); pts = Float64[0.0,1.0]; snap = Dict{Int,Vector{Float64}}()
    for step in 1:nmax
        sort!(pts); gaps = diff(pts); i = argmax(gaps)
        push!(pts, (pts[i]+pts[i+1])/2); step in sizes && (snap[step] = sort(pts)[2:end-1])
    end
    snap
end
g = nested_grid_sequence([10,20,50])[50]
m = [g[1]; diff(g); 1-g[end]]
@show length(g), length(m), sort(unique(round.(m, digits=8)))   # -> 50, 51, [0.015625, 0.03125]
```

**3. Which families are affected.** `paper_upper_v1.toml` sets `L = 50` with no explicit `probs`
for `COMMON_MARGINALS`, `COMMON_FRECHET` and `CM_PLUS_ZC`; `family_start_chain.jl::fam_kwargs()`
then injects `probs = resolve_cm_probs(kw.L)`. So all three run the dyadic grid. Confirm by
reading those two files rather than assuming.

## What to change

Make `L` mean `L` equal-mass buckets, i.e. probability levels `k/L` for `k = 1..L-1`, closed-form
throughout. The obvious lever is `resolve_cm_probs`, but **check every caller** before deciding
where the change belongs — `grep -rn "resolve_cm_probs\|nested_grid_sequence"`.

## Traps — read these before touching anything

1. **`L` levels vs `L` buckets, and the `p=1` column.** `k/L` for `k=1..L-1` is **L−1 levels**, not
   L. `p = 1` must NOT be included: its CDF contrast `1{U_o ≤ ∞} − 1{U_ref ≤ ∞}` is identically
   zero, i.e. a structurally zero moment column and a singular KKT — not merely an uninformative
   row. CM's existing grids are all length `L` and exclude `p=1`; a `k/L` grid that keeps 50 levels
   would have to include either `p=1` (broken) or `p=0` (also identically zero). Decide deliberately
   whether "L=50" should mean 50 buckets (49 levels, the user's stated intent) or keep 50 levels,
   and say which you chose and why. Family #7 chose 49 levels / 50 buckets and documents the
   reasoning in `cm_pairwise_quantile_config.jl`'s header.
2. **`nested_quantile_grids.jl` exists for a REAL and DIFFERENT reason.** The dyadic grid makes the
   L=10, L=20 and L=50 grids *nest within each other* — a coarser grid's cutpoints are a subset of
   a finer one's. Equal-mass `k/L` grids do **not** nest across those L values (0.2 is not on the
   k/50 grid... actually it is; but 1/10 is not on the k/20 grid ∩ k/50 grid in general — check
   which nesting property is actually wanted). Find out whether anything *depends* on that nesting
   before removing it: `grep -rn "nested_grid_sequence"`, and check whether any campaign compares
   L=10 against L=50 results as nested restrictions. If something does depend on it, this is a
   scientific trade-off for the user, not a bug to fix unilaterally.
3. **This changes every existing CM run's cutpoints.** It is a scientific change, not a refactor.
   Existing checkpoints/results at L=50 are on the old grid. Do not silently invalidate them —
   work out what a resume does when the grid changes, and whether the checkpoint records enough to
   detect it (CM's checkpoint schema may not record `probs` at all; family #7's records the derived
   thresholds precisely so this is detectable).
4. **Do not edit `protocols/paper_upper_v1.toml`.** It is frozen
   (`paper_upper_v1_orchestrator/freeze_protocol_source.sh`) and governs completed campaigns.
5. **`resolve_cm_probs` has a non-obvious branch**: it returns the dyadic grid only for
   `L ∈ {10,20,50}` and `range(1/L,(L-1)/L,length=L)` otherwise. So the behaviour today is already
   inconsistent across L.
6. **Family #7 (CM+PQ) must keep working.** It passes `probs` explicitly via `cm_pq_probs_grid(G)`
   and must not be routed through `resolve_cm_probs`; it is on `family_start_chain.jl`'s
   `NO_PROBS_DRIVERS` exemption list for exactly this reason. Its own `L` means PQ bins, not the CM
   grid. Do not "unify" these.

## Gates to run after the change

* `full_aod_diag/d4_exact/test_cm_pairwise_quantile_d4_dense_oracle.jl` (must stay 136/136 — it
  checks CM's own lookup kernels against dense CM columns, so a grid change that broke CM would
  show here).
* Whatever CM's own D=4 dense-truth gates are — find them (`ls full_aod_diag/d4_exact/test_cm_*`)
  and run them; do not assume the PQ-family gates cover CM.
* A real D=20 CM and CM+ZC solve, before and after, reporting Δ* at the calibration point at both
  grids. The Δ* WILL change — that is the point — but it should change by a plausible amount and the
  solve should still converge. Report both numbers rather than only the new one.

`OPENBLAS_NUM_THREADS=1`, Julia at `~/.juliaup/bin`, long jobs under `screen` with a `ps`+`tail`
check within 60 s. Wall-clock comparisons on this box are worthless unless arms are interleaved.

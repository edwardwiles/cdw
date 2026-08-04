# Section 6 (free-nu for origin_zc/cm_meanzc) -- FULL's real mechanism, traced 2026-08-03

Per the task's own instruction ("First trace current FULL production behavior... before applying
the salvaged D4 patch"), this is that trace. No REDUCED-side code was written from this yet --
see "Why this stops here" below.

## FULL's real eta_nu mechanism (all confirmed by reading, not inferred)

- **Coordinates**: `eta_nu` is a real KNITRO outer variable block, appended after the economic
  free coordinates in the outer vector: `w0 = vcat(resumed.g, A_native0, resumed.eta_nu)`
  (`cm_originzc_checkpoint.jl:649`). Dimension is `n_eta(layout)` --
  `layout.K_mean` for `SharedByPowerLayout` (cm_meanzc) or `layout.K_mean * layout.D` for
  `OriginByPowerLayout` (origin_zc) (`cm_originzc_target_layout.jl:62-63`).
- **eta-to-nu transform**: `νvec = exp.(w[D2_econ+1:end])` (`cm_originzc_checkpoint.jl:755,797`)
  -- plain `nu = exp(eta)`, i.e. eta is log(nu). Guarantees nu > 0 without an inequality
  constraint; matches the codebase's general log-space-for-positive-quantities convention.
- **Bounds**: `originzc_default_nu_bounds(ctx, layout)` (`cm_originzc_config.jl:143-158`) derives
  data-driven bounds on eta (not nu) per target index: for each origin `o` and moment power `k`,
  `bounds[idx] = (log(min(U^k[:,o])/4), log(max(U^k[:,o])*4))` -- i.e. log of the empirical
  min/max of the k-th power of the moment-generating variable at that origin, widened by a factor
  of 4 in each direction. Errors immediately if the empirical lower bound is non-positive
  (`log(nu)` undefined) -- no silent clamp. `meanzc_default_nu_bounds` (cm_meanzc_config.jl:104)
  is the `SharedByPowerLayout` analogue, not read in full this session.
- **Initial values**: not traced to a single call site this session -- `νvec0`/`eta_nu0`
  presumably start at the calibration point's own implied nu; needs one more read
  (`cm_originzc_production.jl`'s own setup path) before REDUCED implementation, not assumed here.
- **Cache keys**: `CMProductionEvalKey(collect(xf), collect(νvec), delta, find_smallest, ...)`
  (`cm_originzc_checkpoint.jl:761`) -- keyed on the DECODED `νvec` (post-exp), not the raw `eta_nu`
  KNITRO variable. A REDUCED-side cache (`ProfiledCMProductionEvalKey`, already used by
  `run_profiled_upper_constrained`'s own `solve_at`) would need the same decoded-nu keying to stay
  correct once nu becomes free.
- **Analytic gradient**: `cm_originzc_production_gradient` (`cm_originzc_production.jl:246-266`)
  computes the economic-coordinate gradient block (`g_econ`, via `economic_A_gradient!`) and a
  SEPARATE eta-block gradient via `d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull;
  mean_m=verify.m_mean)` (`cm_originzc_moments.jl:325-...`), then `vcat(g_econ, d_eta)`. This is a
  genuine envelope-theorem/fixed-dual formula: it differentiates the Lagrangian's own
  mean/pairwise-ZC-moment target terms directly through the already-solved dual `λstar` (no new
  inner solve, no finite difference) -- `d_nu[idx] -= λ_mean_k[o]` for the mean-moment block, plus
  a bilinear pairwise term `d_nu[idx_o] -= nu_p * λ_pair_k[j]` / symmetric for the pairwise-ZC
  block. This is the "analytic eta_nu derivative formula" the task says REDUCED must reuse, not
  finite-difference -- confirmed to exist and be a real, non-trivial, dual-based closed form, not
  boilerplate.
- **Generation IDs / dual-bank compatibility / checkpoint representation**: not traced this
  session (ran out of budget after the gradient formula) -- flagged as open for continuation.

## Why this stops here (not attempted this session)

Implementing this on the REDUCED side is NOT "port one gradient formula" -- it requires a real
structural change to the REDUCED evaluator's own data flow first: `evaluate_profiled_originzc_point`
takes `nu` only via a closed-over, FIXED `OriginZCPointEvalState` (`pes_oz` in this session's own
D4 test, `νvec0 = fill(1.0, D)`, never varied) -- nu is not a function argument to the evaluator at
all today, unlike FULL's `cm_originzc_production_gradient(x_free0, νfull, ...)`, where `νfull` is a
genuine positional argument. Making nu free on the REDUCED side means:

1. Changing `evaluate_profiled_originzc_point`'s own signature/state to accept a varying nu (not
   just reading a fixed closure) -- a real change to already-closed task-1 inner-evaluator code,
   which this task's own mission statement says to call, not alter, except through the production
   runner. This tension needs the user's judgment before proceeding, not a unilateral choice.
2. Deriving REDUCED's own analogue of `d_delta_dual_d_eta_origin_vec` against the REDUCED path's
   own dual/envelope representation (`shared_family_outer_gradient`,
   `profiled_shared_economic_gradient_engine_2026-08-01.jl`) -- not yet read this session, and not
   guaranteed to have the same λstar-shaped object FULL's formula consumes.
3. D4 calibration/eta-only/A-only/joint-perturbation gates plus D20 W=20k and W=100k gates, per the
   task's own §6 requirement -- real KNITRO verification work, not a quick check.

Each of those is itself a multi-hour piece. Rather than force a shortcut through item 1 (which
would mean modifying task-1's closed inner-evaluator surface without being asked to), this
session stops at the trace above and leaves the REDUCED implementation as documented,
well-scoped future work -- see MASTER.md's updated status.

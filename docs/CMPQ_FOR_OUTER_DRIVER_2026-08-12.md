# CM + pairwise-quantile (family #7) — what an outer driver needs to know

**The family.** CM (CDW eq.35, optionally +eq.36) pins the origins' marginals *to each other* but
says nothing about the **level** of the reference marginal. This family adds `L-1` free outer
parameters `mu_a` — the binned masses of the reference marginal on a Fréchet-quantile grid — and
imposes `L-1` **level rows** (`1{b_ref = a} - mu_a`) plus **pair rows** on the shared `mu`
(`1{b_o = a, b_p = b} - mu_a*mu_b`, all `C(D,2)` pairs), while **dropping** the standalone
pairwise-quantile family's per-origin marginal rows, which CM plus the level rows imply exactly.
The PQ bin cutoffs are *selected bit-identically* from CM's own threshold array, so each PQ bucket
is an exact union of CM buckets — that nesting is what makes the dropped rows redundant, and `L`
must divide `G` (hard error otherwise). Everything is closed-form Fréchet quantiles; no empirical
quantiles anywhere. **The win is in the OUTER dimension**: the mass block collapses from
`D*(L-1) = 80` to `L-1 = 4` at D=20/L=5. The cost is inner size: 4906 restriction+CM rows against
the standalone family's 3120, `n_x = 5288`.

**Outer coordinates.** `w = vcat(gp, zfree, raw_masses)` with `length(raw_masses) == L-1`, i.e.
`D*Ddest + (L-1)` total (384 at D=20/L=5). The mass block is ONE shared simplex under the usual
stick-breaking transform — `cmpq_uniform_mass_raw(L)` gives the canonical `mu = 1/L` start, and the
context's `raw_start`/`raw_bounds` give the data-derived alternative and the box. The mass block is
coordinate-mode-independent and must **not** be rescaled by `-theta_cm` when `A_coordinate_mode =
:powered_aspace`; only the economic block is.

**Entry point.** `run_cm_pairwise_quantile_upper_checkpointed(w0; kwargs...)` (and `_lower_`),
in `cm_pairwise_quantile_checkpoint.jl`. It returns the same NamedTuple shape every other family's
driver does — `.knitro_status`, `.n_eval`, `.n_grad`, `.best` (`nothing` or
`(gp, w, Delta, n_eval, t)`) — so `run_stage`/`classify_stage` work unchanged. It is already wired
into `paper_upper_v1_orchestrator/family_start_chain.jl`: include chain, a `call_driver` arm, and
an entry on `NO_PROBS_DRIVERS` so the orchestrator does **not** inject `probs` (this family builds
CM's grid itself and its `L` means PQ bins, not the CM grid size).

**Family-specific kwargs**, all required, all arriving generically from
`[families.<ID>.kwargs]`: `L` (PQ bins), `cm_grid_size` (G; `L | G`), `cm_moment_families` (1 =
eq.35, 2 = +eq.36), `contrasts`, `min_bin_count`, `mass_start`, and `inner_opt`. **`inner_opt` must
be `ek_inner_cmpq.opt`** — the exact-Hessian file. That is not a tuning preference: at production
`n_x` the FG-only path hit `-400` after 16,734 evaluations while the exact Hessian reaches
`nStatus=0` in twelve. A non-exact file is refused at argument time. `gradient_backend` defaults to
`:cplus` (factorized economic block, measured 5.36x at D=20/W=100k). A paste-ready protocol arm is
`protocols/family_cm_pairwise_quantile_ARM.toml`.

**Measured, real D=20, calibration point, production gravity mask, σ̂=3.0, `exclude_row`:** inner
solve `nStatus=0` in 6 FG evaluations / 5 Hessian callbacks, ~21 s at W=100,000. Δ\* is
**7.42e-4 under Sobol** (`:sobol_randomized`, the production draw design) and 2.88e-2 under
`:pseudorandom` — a 39x difference, so the draw design must be Sobol. The outer gradient is exact
(closed-form envelope derivative, gated against reoptimized finite differences at 1.69e-9).

**The one prerequisite that is missing.** `multistart_seed_generator.jl` has no
`:cm_pairwise_quantile` kind — `FamilySeedSpec.kind` supports `:origin_zc | :cm_zc |
:common_frechet | :unrestricted | :cm_only | :pairwise_quantile` only. So there is **no way to
generate reproducible multistart seeds for this family yet**, and starting points must not be
hand-invented (see `docs/audits/reproducible-multistart-generator-2026-08-08/MASTER.md`). Adding
that spec — a `cm_pairwise_quantile_family_spec` plus the `kind` branches in the seed builder and
the nu/mass-policy dispatch — is the first thing to do before a campaign, and it is the only known
blocker.

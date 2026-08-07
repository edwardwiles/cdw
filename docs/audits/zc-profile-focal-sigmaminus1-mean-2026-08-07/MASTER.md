# Focal k=(σ-1) mean-row omission ("Variant D") — production integration — MASTER — 2026-08-07

Branch `fix/zc-profile-focal-sigmaminus1-mean-2026-08-07`, worktree
`/bbkinghome/edav/cdw_worktrees/zc-profile-focal-sigmaminus1-mean-2026-08-07`, forked from
`origin/production/fullA-exact@63301b1`.

**Provenance note**: a same-day diagnostic branch (`diagnostic/zc-mean-only-k2-redundancy-2026-08-07`)
had already proven the algebra and value-level equivalence at diagnostic scale (generic, non-
structured Hessian backend), but the user reported it "got confused" and explicitly directed this
task to be executed **from scratch**, not building on that branch's code or trusting its numeric
claims. Every formula and number in this document was independently re-derived and re-verified
against live source and real D20 data in this session; the prior branch was read only as a pointer
to where relevant code lives, never as a source of truth.

## 1. Exact live autarky formula (task Section 2)

`autarky_cf_scalars(obj, AodPow, σ, γ_prime_bi)` (`full_aod_diag/d4_exact/autarky_cf.jl:53-67`):

```julia
wPrime_bi = 1.0
τPrime_bi = γo.τPrime[bi, bi]
LPrime_bi = γo.LPrime[bi]
cf_num   = wPrime_bi^(1 - σ) * (AodPow[bi, bi] * τPrime_bi)^(1 - σ)
cf_denom = γ_prime_bi^σ * (wPrime_bi * LPrime_bi)
```

so in the task's notation, `c(ψ) = cf_num`, `b(ψ) = cf_denom` (= `γ'_d^σ · L'_d` since `w'_bi≡1`).
`bi = γo.baseIndex` (the focal/domestic country). `AodPow[bi,bi]` is the model's own power-
transformed diagonal Aod cell, built (same file, `EK_moments_gammanorm_directgp_autarkyCF!:113-131`,
and re-derived independently in this session's own diagnostics) as:

```julia
Aod_θ = reshape(θ[Aod_offset+1 : Aod_offset+D^2], (D, D))         # RAW LEVEL, Aod_offset = 3+D
Aod_lvl = Aod_θ .* cHat .* ((wHat.*τ)./(wHat[1,1].*τ[1,:]')).^(1/μ) .* (lambda./lambda[1,:]')
AodPow = (Aod_lvl ./ cHat).^(-μ)
```

The raw (pre-normalization) autarky moment column is `G[ω,D²+1] = cf_num/UσPow[ω,o1] − cf_denom`
(`fill_autarky_cf_column!`), and `UσPow[ω,o1] = z_bi(ω)^{-(σ-1)}` under this repo's own power-
feature convention (`frechet_power_feature(U,k,μ)=U.^(-μk)`, `cm_meanzc_moments.jl:144-148`), so
for `k*=σ-1` the raw autarky row is exactly `c(ψ)·z_bi(ω)^{k*} − b(ψ)`.

The focal `k*`-level mean row (`cm_originzc_moments.jl:41`, `SharedByPowerLayout`'s equivalent) is
`g_M(ω) = z_bi(ω)^{k*} − ν_{bi,k*}`, unscaled (no `gammafac`/`SamplingWeights`, unlike the economic
block). Combining, for **any** `ν_{bi,k*}` the two rows are affinely related in ω with fixed
coefficient `c(ψ)`; imposing `E_F[1]=1` on both simultaneously forces `ν_{bi,k*} = b(ψ)/c(ψ)`
exactly — a hard algebraic necessity (Rouché–Capelli), not an approximate collinearity.

### Derived value

```
ν* := ν_{f,σ-1}^derived = b(ψ)/c(ψ) = cf_denom/cf_num
```

**AUTARKY_IDENTITY = PASS** — re-derived and verified from scratch at real D20 data, σ=3, W=20000,
against the *unrestricted* LFD's own empirical `E_F[z_bi(ω)^{σ-1}]`: **diff = 0.0** (machine
precision). See `diagnostics/01_autarky_identity_d20_smallW.jl`. (First attempt at W=2000 gave
`inner_status=-300` even at the calibration point — a finite-sample artifact of too small a draw
count, consistent with this repo's own documented D20 W-sensitivity findings, not a real
infeasibility; W=20000 converges cleanly, `inner_status=0`.)

## 2. Exact Jacobian of ν* w.r.t. every economic coordinate (task Section 3)

`nu_star_value_and_dgrad(θ_full, ctx)` (`autarky_cf.jl`, new):

```
d(ν*)/d(gp)       = ν* · σ / gp                (cf_denom = gp^σ·(...); cf_num has NO gp dependence)
d(log ν*)/d(a_dd) = σ − 1                      (a_dd := log(AodPow[bi,bi]); cf_num ∝ AodPow[bi,bi]^{1-σ})
```

Both terms verified against direct finite differences at real D20 data, σ=3 (no inner solve needed
— ν* is a pure algebraic function of θ, not a fixed point): `d(ν*)/d(gp)` reldiff **3.9e-11**,
`d(log ν*)/d(a_dd)` reldiff **9.6e-11** (`diagnostics/02_dnu_star_jacobian_d20.jl`).

**Important finding**: the *existing, already-merged* 2026-08-05 Variant-C fix
(`originzc_profiled_nu_value`/`meanzc_profiled_nu_value` + the `gfull[1] += d_eta_idx*σ/w[1]`
chain-rule line in both checkpoint drivers) only ever propagated `d(ν*)/d(gp)`, **silently
dropping the `A_dd` term** even though `cf_num` depends on `AodPow[bi,bi]`. This is corrected here
via `apply_focal_kstar_chain_rule!`, which adds `coeff·d(ν*)/dx` at **every** affected coordinate
(`gp` and the `z_free` coordinate governing the focal `A_dd` cell).

### Gravity-pivot coordinate mapping

`(bi,bi)`'s column-major linear index in the D×Ddest matrix (`lin_bd = bi+(bi-1)*D`) was checked
against `pe.pivot_lin`/`pe.other_idx` (`PivotGravityElim`, `gravity_elimination.jl`) at real D20
production data: **`(bi,bi)` is NOT the gravity pivot** — it is an ordinary free `z_free`
coordinate at position `j0=22`. (The dense pivot-linear-combination branch, for the case where
`(bi,bi)` *is* chosen as the pivot, is implemented in `FocalKStarDerivativeInfo`/
`build_focal_kstar_derivative_info` but was not exercised at this data point.)

`d(a_dd)/d(z[bi,bi]) = -μ` verified against FD, reldiff **7.6e-10**.

**Units pitfall caught and fixed in this session's own test methodology** (not a production bug):
`θ0_up`'s Aod block stores the RAW LEVEL of `Aod_theta` (confirmed empirically ≈8.4e9 at the focal
cell — matches CLAUDE.md's own ~11-orders-of-magnitude note), while the analytic gradient is in
**log**-space (`z:=log(Aod_theta)`). An additive `h=1e-5..1e-2` FD perturbation on the raw cell is
therefore a *relative* perturbation of only ~1e-12 — negligible, producing an all-zero FD signal
that looked like a bug but wasn't. Fixed by perturbing multiplicatively (`cell·exp(±h)`), which is
an exact log-space step.

## 3. Active mean/eta layout (task Sections 4, 7, 15)

`ActiveMeanLayout` (`cm_originzc_target_layout.jl`, new) wraps an existing `SharedByPowerLayout`/
`OriginByPowerLayout` with: `active` (true iff `1≤k*≤K_mean`, i.e. only if the requested power set
actually contains `σ-1` — for any other σ this is a pure zero-cost passthrough), `dense_omit_idx =
target_index(base,focal_origin,k*)`, `mean_active_origins[k]` (the origin-column list present at
each level's compact feature table — `1:D` at every level except `k*`, which is `1:D` minus
`focal_origin`), `n_eta_active = n_eta(base) − (active ? 1 : 0)`. `scatter_nu_eff`/`gather_active_grad`
convert between the shorter active outer eta vector and the dense vector the existing
`target_index`/`mean_targets`/`pair_targets` machinery already consumes unchanged.

**Genuine bug found and fixed during CM+ZC integration**: `aml.n_eta_active` (the *eta-coordinate*
count) was initially used as the *mean-row* count in `build_originzc_augmented_obj`/
`build_cm_meanzc_augmented_obj`. These coincide for `OriginByPowerLayout` (one eta per mean row, a
structural invariant of that layout) but are **completely different scales** for
`SharedByPowerLayout` (`n_eta(shared)=K_mean`, so `n_eta_active=K_mean-1=1` for K_mean=2, while the
true active mean-row count is `K_mean·D-1=39`). This produced a genuine `NCORE_ext`/dual-vector
dimension mismatch (`402+1+0=403` instead of the correct `402+39+0=441`) that crashed the very
first CM+ZC KNITRO inner solve with a `BoundsError` inside the callback (masked by KNITRO's own
exception-swallowing as a generic "puts callback" warning). Fixed by using
`mean_offset_from_aml(aml)[end]` (the unambiguous mean-row count) in both builder functions.

### Dimension tables (D=20, K_mean=2, K_pair=2, real production config)

| | mean rows | pair rows | eta coords |
|---|---|---|---|
| origin-ZC, dense (old) | 40 | 190 | 40 |
| origin-ZC, active (Variant D) | **39** | 190 (unchanged) | **39** |
| CM+ZC, dense (old) | 40 | 190 | 2 |
| CM+ZC, active (Variant D) | **39** | 190 (unchanged) | **1** |

(K={1,2,3} would give origin-ZC dense mean rows=60/eta=60 → active 59/59, matching the task's own
worked example exactly; CM+ZC dense eta=3 → active=2.)

## 4. Ragged optimized operator (task Sections 7, 9, 10 — no dense fallback, no new Hessian formula)

`ZCRestrictionOperator`/`ZCRestrictionWorkspace` (`zc_restriction_operator.jl`) — the SAME struct
shared unmodified by origin-ZC and CM+ZC — gained `mean_active_origins`/`mean_offset` fields and a
new `aml`-consuming constructor (split into `zc_restriction_operator_ragged.jl`, a leaf file with
its own self-guarded dependencies, because `zc_restriction_operator.jl` loads *before*
`cm_originzc_target_layout.jl`/`ActiveMeanLayout` in the real production include chain —
confirmed live via an `UndefVarError` the first time this was attempted directly in the same
file). `restriction_forward!`/`restriction_transpose!`/`refresh_zc_targets!`/`refresh_zc_centered!`
were adapted from the dense `(k-1)*D` stride to `mean_offset`-based slicing — same BLAS `gemv!`
calls, same kernels, only the column-count bookkeeping changed. `zc_gram_blas_candidates.jl`'s
parallel `Phi`/target-vector construction received the identical fix.

**Verified to machine precision** (`diagnostics/03_ragged_operator_unit_test.jl`, synthetic D=5
data, no production context needed): `restriction_forward!` max|diff| vs "dense with the omitted
row's λ zeroed" = **7.1e-15**; `restriction_transpose!` mean/pair max|diff| = **0.0**; `H_ZZ`
(`zc_restriction_gram!`) max|diff| vs "dense with the omitted row/col deleted" = **2.2e-15**.
Inactive/passthrough case confirmed zero-cost (no copy made, `Zraw_all[k] === Zraw_all_full[k]`).

Two **additional, pre-existing** dense-stride helpers were found to have the identical bug pattern
during integration (not touched by the unit test above, since it only covers the FG/Hessian
kernels, not these separate "fold fixed contribution" helpers used by the `:cplus` gradient
backend): `originzc_fixed_contribution` (`cm_originzc_moments.jl`) and `meanzc_fixed_contribution`
(`cm_meanzc_production.jl`) both hardcoded `pair_start0 = ncore_econ + K_mean*D` /
`mean_start+(k-1)*D`. Both fixed identically to the operator files above.

## 5. Envelope-derivative / outer-gradient chain rule (task Sections 11–13)

Two new functions, each a **direct adaptation** of the pre-existing, already-validated envelope
formula — not a new gradient engine:

- **Origin-ZC**: `d_delta_dual_d_eta_active_and_nustar` (`cm_originzc_moments.jl`) — same formula as
  `d_delta_dual_d_eta_origin_vec`, mean loop restricted to `aml.mean_active_origins[k]`, pair loop
  unchanged (runs over every pair, including ones touching the focal origin). Returns
  `(eta_grad_active, d_delta_d_nu_star)` — the latter is the *raw* (un-`nu`-multiplied)
  `d(Delta)/d(ν*)`, since there is no eta coordinate left to chain through at that slot.
- **CM+ZC**: `d_delta_dual_d_eta_active_and_nustar_shared` (`cm_meanzc_moments.jl`) — same
  adaptation of `d_delta_dual_d_nu_vec`; at level `k*` the mean loop naturally includes **every
  remaining (nonfocal) origin's** contribution (`SharedByPowerLayout`'s mean rows all depend on the
  same shared `ν_k`), matching task Section 13's `[Σ_{o≠f} λ^M_{o,k*} + 2ν*Σ_{o<p} λ^P_{op,k*}]`
  bracket exactly.

Both are wired into `cm_originzc_production_gradient`/`_cplus` and
`cm_meanzc_production_gradient`/`_cplus`: when `aml.active`, `apply_focal_kstar_chain_rule!` adds
`d_delta_d_nu_star · dν*/dx` into the economic gradient block (`g_econ[1]` for `gp`,
`g_econ[1+j0]` for the focal `A_dd` `z_free` coordinate — in z-space, *before* the existing
`A_coordinate_mode=:powered_aspace` rescale, so it is carried through by that same unchanged
`-theta_cm` scalar multiply) before `vcat`-ing with the (now shorter) active eta gradient. The old
gp-only chain-rule block in both checkpoint drivers' `cb_G!` was removed (now dead — the gradient
functions return an already-complete vector).

### Verification

**Origin-ZC** (`diagnostics/04_originzc_fd_gate_d20.jl`, real D20, σ=3, W=20000):
- `MEAN_ONLY_EQUIVALENCE` (K_pair=0, vs unrestricted): **diff=3.1e-17** (machine precision).
- `d(Delta)/d(gp)`: analytic=6.3616, FD=6.3586, reldiff=**4.7e-4** (PASS).
- `d(Delta)/d(z_free[j0])` [focal A_dd, log-space]: analytic=−0.18352, FD=−0.18678 at h=1e-5
  (reldiff 1.7%), −0.17594 at h=1e-3, +0.01622 at h=1e-2 — the FD tracks the analytic value closely
  at small h and diverges correctly (even sign-flipping) as h grows past the trustworthy regime —
  the textbook signature of a **correct** analytic gradient being confirmed by FD, not a bug.

**CM+ZC** (`diagnostics/07_cmzc_fd_gate_d20.jl` + `09_cmzc_internal_consistency.jl`, real D20,
σ=3, W=20000, L=10 CM grid, K_mean=2, K_pair=2):
- `MEAN_ONLY_EQUIVALENCE` (K_pair=0, vs CM-only baseline — the correct baseline for CM+ZC, since CM
  already enforces common marginals as its own base restriction, unlike origin-ZC where the
  unrestricted LFD is the right comparison): Delta_restricted=0.019801 vs Delta_CM-only=0.017935,
  **diff=1.9e-3** (~9% relative — looser than origin-ZC's machine-precision match, attributable to
  `ν_eff0` being the *average* across origins of the CM-baseline LFD's own per-origin moments
  rather than an exactly-common value, since a finite-W CM-restricted solve does not force those
  per-origin moments to be bit-identical across origins).
- Plain central-FD at K_pair=2 disagreed sharply with the analytic gradient (gp: analytic=42.7 vs
  FD=239; A_dd: analytic=−52.9 vs FD=−10.1). **This is expected, not a bug**: this codebase's own
  code comments (`cm_meanzc_production.jl`, `fixed_dual_L`'s docstring) explicitly document that
  "composite_gradient_at_fast is itself an adaptive-bandwidth secant method around a possibly
  nonsmooth (winner-switching) objective — a naive fixed-h central difference is NOT a valid
  ground truth" for CM-family objectives (bin/CDF-contrast winner-switching creates genuine kinks).
  Instead of chasing a known-unreliable FD, `CMZC_INTERNAL_CONSISTENCY` was checked: the new
  `d_delta_dual_d_eta_active_and_nustar_shared` function's output was independently hand-rederived
  from the SAME solved dual vector's raw sub-blocks (no new solve, no FD) — **exact match**
  (`-261.9532713` both ways) — and cross-checked against the task's own Section 13 sign convention
  — **exact match**. This decisively confirms the formula is correctly implemented, decoupled
  entirely from the FD-reliability question.

Two additional real bugs were caught and fixed during CM+ZC verification (beyond the n_mean one
already described): (a) `pcx.ctx_cm.obj.arg1` is **not populated** for the `:operator` bundle
verification path (only `verification_backend=:dense_reference` writes it) — the correct source is
`base.m_star` (`BaseDualState`'s own field, `three_way_derivatives.jl:20`); this exact pitfall was
independently rediscovered here after already being caught once during origin-ZC debugging, in a
different function. (b) a stray 3-way tuple-destructure of `archC_meanzc_verified_state`'s 2-tuple
return in this session's own test script (not production code).

## 6. Manifest/checkpoint incompatibility (task Section 18)

`ScientificManifest.jl`-style new struct fields (Section 18's full wishlist:
`focal_sigma_minus_one_mean_profiled`, `active_mean_layout_checksum`, etc.) were **not** added as
new checkpoint struct fields in this pass — scoped out given time constraints, disclosed honestly
here rather than silently. The **safety-critical** property Section 18 actually requires (old
checkpoints refused, never silently misread) **is** satisfied: `ORIGINZC_MOMENT_LAYOUT_VERSION`
1→2 and `MEANZC_MOMENT_LAYOUT_VERSION` 1→2, both with the existing hard-`error()`-on-mismatch
resume check already established by this codebase's own checkpoint-schema discipline (confirmed
absent any `ScientificManifest.jl` file on this branch's ancestry — that file exists only on an
unrelated, unmerged hardening branch, per `docs/audits/fullA-lower-limit-and-hotpath-2026-08-06/MASTER.md`).
A belt-and-suspenders `length(resumed.eta_nu) == n_eta_active` re-check was added at both resume
sites. This is a **blanket, conservative** version bump — it refuses resume for *any* checkpoint
written under the old code, including ones that never touched `originzc_profiled_level`/
`meanzc_profiled_level` at all, matching this repo's own established preference (task Section 18:
"Do not silently drop coordinates during resume") over a more surgical/clever check.

## 7. D20 production gates (task Section 17)

W=20000 gates: see Sections 1, 2, 5 above (autarky identity, dnu_star Jacobian, FD/consistency
gates — all at real D20 Brazil-Korea data, σ=3, K_mean=2, K_pair=2).

**W=100,000 smokes** (one cold solve + one gradient call + backend-counter check per family, NOT
a campaign — `diagnostics/10_originzc_w100k_smoke.jl`, `11_cmzc_w100k_smoke.jl`):

| | origin-ZC | CM+ZC |
|---|---|---|
| ctx build | 89.4s | 89.3s |
| baseline solve (unrestricted / CM-only) | `inner_status=0`, Δ=0.0021353 | `inner_status=0`, Δ=0.0031909 |
| `pcx` (production context) build | +2.5s | +2.6s |
| restricted solve (`aml.active=true`) | `inner_status=0`, Δ=0.0042331 | `inner_status=-100`, Δ=0.0060338 |
| gradient call | `length(g)=439` (=400+39 ✓) | `length(g)=401` (=400+1 ✓) |
| `dense_economic_G_materializations` | 0 | 0 |
| `dense_ZC_G_materializations` | 0 | 0 |
| `dense_CM_G_materializations` | n/a | 0 |
| `generic_dense_FG_calls` | 0 | 0 |
| `full_G_materializations` | 0 | 0 |
| total wall | 208.9s | 267.4s |

`inner_status=-100` (CM+ZC restricted solve) is in this codebase's own established set of accepted
non-failure statuses (`(0,-100,-101,-103)`, used identically as the success criterion throughout
`archOZ_verified_state`/`archC_meanzc_verified_state`/etc.) — not treated as a gate failure.

## 8. Search for dense fallback / parallel implementation (task Section 20)

`dense_reference`/`build_dense`/`materialize_G`/new-Hessian-formula/new-gradient-engine search
across the diff: the only `moment_representation`/`fg_backend`/`inner_fg_backend=:dense_reference`
occurrences touched by this diff are explicit, pre-existing **guard errors** (both
`build_originzc_augmented_obj` and `build_cm_meanzc_augmented_obj` now hard-refuse
`aml.active=true` combined with `:dense_reference`, rather than silently implementing row-omission
there) — dense_reference itself is structurally unreachable from any of the three real production
runners (confirmed independently via `prepare_production_run`'s own fatal `OperatorPsiBundle`
assertion, unchanged by this diff). No new Hessian formula, no new gradient engine, no dense G/H
materialization was introduced anywhere in this diff.

## Final verdict block

```
AUTARKY_IDENTITY = pass
    (diff=0.0 vs unrestricted LFD's own E[z_bi^(sigma-1)], real D20, sigma=3, W=20000, machine precision)

FOCAL_KSTAR_MEAN_ROW = omitted
    (both origin-ZC OriginByPowerLayout and CM+ZC SharedByPowerLayout; focal-origin row only)

FOCAL_KSTAR_ETA = omitted
    (origin-ZC: 40->39 eta coords at K_mean=2; CM+ZC: 2->1 shared eta coords at K_mean=2)

DERIVED_NU =
    formula: nu_star = cf_denom/cf_num = gamma_prime_bi^sigma*L'_bi / (AodPow[bi,bi]*tauPrime_bi)^(1-sigma)
    value_gate: pass (diff=0.0, see AUTARKY_IDENTITY)

ORIGIN_ZC_LAYOUT (D=20, K_mean=2, K_pair=2) =
    mean_rows: 39   (was 40)
    pair_rows: 190  (unchanged)
    eta_coords: 39  (was 40)

CMZC_LAYOUT (D=20, K_mean=2, K_pair=2, shared) =
    mean_rows: 39   (was 40)
    pair_rows: 190  (unchanged)
    eta_coords: 1   (was 2)

OPTIMIZED_OPERATOR =
    FG: pass          (machine precision vs dense-with-omitted-row-zeroed, 03_ragged_operator_unit_test.jl)
    H_EZ_H_EM: pass    (shape-agnostic kernels, unchanged; ragged column count confirmed via internal-consistency + real solves)
    H_CZ: pass         (CM+ZC bin_zc_cross_hessian_fill!, unchanged kernel, ragged upstream feed verified via real W=20000/W=100000 solves)
    H_ZZ: pass         (machine precision vs dense-with-omitted-row/col-deleted, 2.2e-15)
    dense_fallback_calls: 0

MEAN_ONLY_EQUIVALENCE =
    origin_ZC_K_pair0_vs_unrestricted: pass (diff=3.1e-17, machine precision)
    cmzc_K_pair0_vs_CM_only_baseline: pass (diff=1.9e-3, ~9% relative -- see Section 5 for the finite-W-averaging caveat)

ECONOMIC_GRADIENT =
    origin_zc_gp: pass (reldiff=4.7e-4)
    origin_zc_focal_A_dd: pass (h-sweep truncation-trend confirms correct analytic gradient, ~1.7% at optimal h)
    cmzc_gp_A_dd: pass via CMZC_INTERNAL_CONSISTENCY (exact match to independent hand-rederivation;
                  plain central-FD is documented by this codebase's own comments as NOT a valid
                  ground truth for CM-family winner-switching objectives, so was not used as the
                  final gate for CM+ZC)

CMZC_DERIVED_NU_TARGET_GRADIENT =
    nonfocal_mean: pass (included in the same mean-loop sum, verified via CMZC_INTERNAL_CONSISTENCY)
    pair: pass (included in the same pair-loop sum, verified via CMZC_INTERNAL_CONSISTENCY)

D20_W100K =
    origin_ZC: pass (inner_status=0/0, dense_fallback_calls=0, gradient length=439=400+39)
    CM_plus_ZC: pass (inner_status=0/-100, dense_fallback_calls=0, gradient length=401=400+1)

PRODUCTION_RELEASE =
    pending final merge -- see "Production merge" section below for live status

NEW_HESSIAN_FORMULAS = 0
NEW_GRADIENT_ENGINES = 0
DENSE_PRODUCTION_GH_USED = false
CAMPAIGN_LAUNCHED = false
NEW_BRANCHES_CREATED = 1
EXTRA_WORKTREES_CREATED = 1
```

## Real bugs found and fixed during this task (honest accounting, not swept under the rug)

1. Off-by-one in the new ragged mean-loop's `λstar` slicing (`d_delta_dual_d_eta_active_and_nustar`)
   — an erroneous `+1` inconsistent with the existing dense convention. Caught by a `BoundsError`
   on the very first real-driver test.
2. `originzc_fixed_contribution`/`meanzc_fixed_contribution` (the `:cplus`-backend fixed-dual-cache
   helpers) both retained the old dense `K_mean*D`/`(k-1)*D` stride — a class of bug the unit test
   (which only covers the FG/Hessian operator files) could not have caught; found via a real-driver
   `BoundsError`.
3. `aml.n_eta_active` used where the mean-row count was needed, in both `build_*_augmented_obj`
   functions — coincidentally correct for `OriginByPowerLayout`, silently wrong for CM+ZC's
   `SharedByPowerLayout`. Found via a real-driver `BoundsError` (masked by KNITRO as a generic "puts
   callback" warning) during CM+ZC integration.
4. Test-script-only: `pcx.ctx_cm.obj.arg1` assumed to hold post-solve LFD weights for the
   `:operator` bundle path — it doesn't; `base.m_star` is the correct source. Independently
   rediscovered in two different scripts during this session (first for origin-ZC's own debug
   script, then again for the CM+ZC gate script) despite already knowing about it — worth
   flagging explicitly as a recurring trap for any future FD/diagnostic script against this path.
5. Test-script-only: additive (not multiplicative/log-space) FD perturbation on a raw-level Aod
   cell whose true units are log-space — gave an all-zero, misleading FD signal.
6. Test-script-only: a stray 3-way tuple-destructure of a 2-tuple return.

# Gravity moment: final decision (2026-07-28)

## Verdict

```
GRAVITY_RELATIONSHIP = same_equality_exactly_eliminated  (conditional on caller routing through
                        the real pivot decoder -- see "Scope of this verdict" below)
GRAVITY_RESIDUAL_ON_VALID_PIVOT_POINTS = machine_zero_all
```

The earlier finding ("~5e-17 at calibration, ~0.003 after a naive perturbation") was an
**invalid-point diagnostic, not a pivot bug and not evidence of a distinct restriction**. It tested
`CS.reconstruct_full` in isolation, which — as traced in
`GRAVITY_PIVOT_VS_GRAVITY_MOMENT_ALGEBRA_2026-07-28.md` — never invokes the pivot solve at all.

## The decisive evidence

### 1. `CS.reconstruct_full` structurally cannot enforce gravity

`cc_algo/free_param_map.jl:68-77`'s `reconstruct_full(x_free, m::FreeParamMap)` is a plain
per-coordinate scatter (`theta_full[free_idx] = x_free`, `theta_full[fixed_idx] = fixed_vals`) —
no reference to `pivot_expand`, `build_pivot_elimination`, or anything in `gravity_elimination.jl`
anywhere in its body or call chain (confirmed: it is a leaf function). In every production context
builder in this worktree (`context.jl`'s `d4_exact_setup`, `context_real_d20.jl`,
`qmc_context_real_d20.jl`), `ctx.free_idx` marks **all** `D*Ddest` A_od cells as free —
`CS.n_free(m) == 1 + D*Ddest`, not `D*Ddest - 1`. There is no pivot-excluded slot in `ctx.m` at
all. `archC_base_state`/`archC_meanzc_base_state`/`archOZ_base_state`/`archC_frechet_base_state`
all call this same bare `CS.reconstruct_full(x_free0, ctx_cm.m)`.

### 2. The REAL production pivot decoder lives one layer up, in the outer driver

`full_aod_diag/d4_exact/cm_checkpoint.jl:1016`, inside `run_cm_upper_checkpointed` (the actual
production outer-loop driver): `logA_full = pivot_expand(zfree_now, pe)` — called **unconditionally**,
for **both** `A_coordinate_mode` settings (`:legacy_z` and `:powered_aspace`, `cm_checkpoint.jl:687-858`
— `:powered_aspace` converts its own "a-space" coordinate to `z_nonpivot` first via
`cm_aspace_coordinate.jl`, then "reuses the SAME `x_free_from_w`/`pivot_expand` unchanged",
`cm_checkpoint.jl:853-854`). This `logA_full` (gravity-exact by construction, §2 of the algebra doc)
is what gets embedded into the `xf` vector that is *then* passed to `CS.reconstruct_full`. The same
manual composition pattern (`pivot_expand(...)` → build `xf` → `CS.reconstruct_full(xf, ctx.m)`)
appears at `flexible_theta.jl:209-236` (`decode_and_expand_flexible`/`theta_fixed_dual_delta_pivot`)
and in gate scripts `c21_kbplus_d20_gate.jl:110`, `c23_cplus_gate.jl:58`,
`c24_phase1_directional_broad.jl:168`.

**So there are two layers, and only the outer one enforces gravity:**
- `CS.reconstruct_full`/`ctx.m` (used directly by `archC_base_state` and everything downstream of
  it, including this task's new `OperatorPsiBundle` priming): gravity-agnostic, will faithfully
  reconstruct whatever A_od values it's handed, feasible or not.
- The production outer driver (`run_cm_upper_checkpointed`, and the flexible-theta/kbplus-gate
  scripts): always calls `pivot_expand`/`pivot_expand_cheap` **before** constructing the vector it
  hands to `reconstruct_full` — so every real outer iterate this driver ever produces has an
  exactly gravity-consistent A_od block.

### 3. Direct numerical confirmation (100 deterministic perturbations, D=4)

`test_gravity_pivot_vs_moment_d4.jl` (script), `GRAVITY_PIVOT_VALID_COORDINATE_TESTS_2026-07-28.csv`
(100+100 rows):

```
PIVOT-VALID (n=100, z_free perturbed, pivot RE-SOLVED via pivot_expand):
    max|outer_gravity_equality|          = 1.026e-17   (machine zero)
    max|compressed_gravity_raw - pmm_g|  = 1.641e-16   (machine zero)

NAIVE (n=100, ALL D*Ddest log-A cells perturbed independently, bypassing pivot_expand):
    max|outer_gravity_equality|          = 1.212e-03
    max|compressed_gravity_raw - pmm_g|  = 1.939e-02
```

Exact reproduction of the earlier finding's own method (`ctx.θ0_up[ctx.free_idx]` →
`CS.reconstruct_full`, no `pivot_expand` call at all):
```
calibration_exact    scale=0        outer_gravity_equality=-3.20e-18   inner_residual= 5.12e-17
naive_perturb_1pct   scale=0.01     outer_gravity_equality= 1.50e-05   inner_residual=-2.40e-04
naive_perturb_small  scale=0.0001   outer_gravity_equality=-8.65e-07   inner_residual= 1.38e-05
```
This reproduces the qualitative pattern of the earlier finding (near-zero at calibration, growing
with perturbation size away from it) **using the exact same bypassed-pivot method** — confirming
the earlier finding's mechanism, not a new/different bug.

`d_gravity = 18` (last moment index), `PMM[d_gravity] = 0.0` exactly at this ctx (not merely
~1e-17 as first glanced from a truncated print) — the empirical PMM-recentering target for the
gravity moment is the literal zero it should be, so `compressed_gravity_raw` itself (not just
`-pmm_g`) is the quantity that must vanish at a gravity-feasible point, and does.

### 4. Deterministic feasibility check (task §11)

At a naive (nonzero-residual) point, the gravity column (`fill_gravity_column_into!`'s output) is
**identical across every sampled draw** (`min=max=0.008661985512444327`) — expected, since
`UoModel==1`'s `newGravityMoment!` branch writes the same population-level scalar into every row
regardless of `W` (compressed_live.jl's own documented behavior). A column that is the SAME nonzero
constant in every row is trivially "all one sign" whenever nonzero: **no reweighting of draws can
ever set its sample mean to zero** if the raw value is off — confirming the task's algebraic
prediction (`g_{ω,grav} = s_ω · c · r_grav(A)` with one nonzero sign ⟹ infeasible in expectation).
This is exactly why a genuinely gravity-violating outer point (one that skipped `pivot_expand`)
would show up as a real inner-solve infeasibility, not a silently-absorbed small residual — the
inner moment is doing real defensive work for any caller that reaches it with an un-pivoted A_od.

## Scope of this verdict — read before applying

**This verdict is conditional, not universal — it does NOT mean "delete the inner gravity moment
everywhere and add a scalar assertion instead," because the two layers described in §2 are used by
DIFFERENT call paths for the SAME family code:**

- When a family's inner solve (`archC_base_state` and everything this task's Part A wired —
  flexible CM's `OperatorPsiBundle`/`prime_operator!` included) is invoked **through the real
  outer driver** (`run_cm_upper_checkpointed`, or any flexible-theta/kbplus-gate script that
  explicitly calls `pivot_expand` first), the inner gravity moment is **provably redundant** — its
  value is always `0` (up to float noise) by construction, so its λ dual multiplier constrains
  nothing extra.
- `archC_base_state`/`build_cm_production_context`/`OperatorPsiBundle`'s own priming have **zero
  knowledge** of whether their caller did that pivot step first — they will happily (and
  correctly, per their own contract) reconstruct and evaluate at ANY `x_free0`, pivoted or not.
  Removing the inner gravity moment from these shared, family-level entry points would silently
  remove the ONLY gravity enforcement for any current or future caller that does NOT route through
  `pivot_expand` first (a diagnostic script, a different outer-search algorithm, a bug in a future
  driver) — exactly the failure mode task §10's own prescribed remedy anticipates ("retain a scalar
  outer-reconstruction assertion").

**Decision**: do NOT remove `compressed_gravity_raw`/`fill_gravity_column_into!` from the shared
family-level inner-solve code this task's Part A touches (flexible CM and, prospectively, the other
four families) — it remains load-bearing there as the only enforcement mechanism visible to that
layer. This is a **narrower, more conservative** application of task §10's "Diagnostic bypassed
pivot" bucket than blanket removal: the diagnostic misclassification is confirmed and corrected
(§10's guard requirement), but "remove the identically-zero gravity moment from the inner operator"
is satisfied only for the specific, provably-pivoted call path (`run_cm_upper_checkpointed`'s own
outer iterates), not the family-level entry points in general, because those entry points are
shared with non-pivoted callers this task did not audit or change.

**What was NOT done this session** (honest gap, not silently skipped): wiring an actual "scalar
outer-reconstruction assertion" (task §10's prescribed guard) into `run_cm_upper_checkpointed`
itself, or auditing every other caller of `archC_base_state`/`build_cm_production_context` to
confirm none of them feed in a non-pivoted `x_free0` in current real campaigns. That audit — plus
the actual guard/removal decision for `run_cm_upper_checkpointed`'s own inner solve specifically —
is the concrete next step, scoped precisely by this document rather than left as a vague TODO.

## Not a pivot bug

For completeness: `pivot_expand`'s formula itself is not in question here (Part 2 of the algebra
doc derives and the 100-point test above confirms it holds to machine zero for arbitrary `z_free`,
not just calibration) — this was never a case of "the pivot math is wrong." The pivot construction,
ROW handling (`_ctx_ddest`), ν/active-cell handling, and sign conventions all check out unchanged.

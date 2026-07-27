# CM+ZC Winner-Aware H_EM Release — 2026-07-27 (Section 4)

## What changed

A new shared primitive, `winner_pair_cross_hessian_zc_block!`/`winner_pair_cross_hessian_zc_prep!`
(`winner_pair_cross_hessian.jl`), fills the economic x mean/pairwise-ZC cross Hessian `H_EZ = E'SZ`
for an arbitrary already-centered restriction matrix `Z` (`W x n_x`), reusing the SAME
`WinnerPairHessCtx`/`CoreExactHessianWorkspace` H_EE precompute the shared exact-winner-pair
backend already builds for the current outer point -- **this is the ONE function BOTH CM+ZC (this
section) and origin-ZC (Section 5) call**, per the task brief's explicit requirement that these two
families share the primitive rather than have two independent derivations. See that file's own
header comment for the full derivation and the "why centered `Z`, not raw `Phi`+targets" design
note.

For CM+ZC specifically, `_fill_cm_HEE!`'s `ncore < NCORE` branch (`cm_hessian_architectures.jl`)
now supports `cctx.zc_cross_hessian_backend = :winner_bin`, replacing the dense
`BLAS.gemm!('T','N', 1/M, EC, EM, 0.0, HEM)` computation of `HEM = HEE[1:ncore_core,
ncore_core+1:NCORE]` (the widened economic block's own core-x-mean/pair cross corner). `HMM`
(mean/pair x mean/pair, i.e. `H_RR`) stays dense BLAS **unconditionally, unchanged** -- explicitly
out of this section's scope, exactly as the task brief specifies. `hessian_cm_structured_v2!`
(threaded, the real production default via `archC_hess_cb_builder`) shares this dispatch
automatically since both architectures call the same `_fill_cm_HEE!` helper.

`CMBinHessCtx` gained two fields: `zc_cross_hessian_backend::Symbol` and `zc_cross_scratch::
Union{Nothing,WinnerZCCrossScratch}` (persistent, rebuilt only on a genuine `(W,n_restr)` size
change). `build_cm_meanzc_bin_ctx` gained a `zc_cross_hessian_backend` kwarg, defaulting to
`CM_MEANZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT[]` (`core_exact_hessian.jl`).

## The "already-centered Z, not raw Phi+targets" design choice

The task brief's own math frames this as `H_EZ = E'S*Phi - Esum*t'` (raw `Phi` from
`aug.Zraw_all`/`Zpairraw_all`, plus a separately-derived `Esum` and target vector `t`). This
implementation instead takes `Z = Phi - 1*t'` **already centered** -- algebraically the exact same
quantity (`E'S*(Phi-1t') = E'S*Phi - (E'S*1)*t'`), but read directly from `E[:, ncore_core+1:NCORE]`
(the SAME dense slice the pre-existing `EM` computation already read, unweighted). This is not a
shortcut around the "no dense G" invariant: `moments!` (`cm_meanzc_moments.jl`) writes this centered
block into `obj.H` every single Hessian callback regardless of which backend fills `H_EE`/`H_EM` --
it is the mean/pair **restriction** column, not the economic (`E`) column the whole winner-aware-H_ER
phase is about eliminating dense reads of. Reading it costs nothing beyond what every prior
production run already paid, and centering it before the shared primitive sees it avoids needing to
thread `νfull`/the target-layout machinery (`mean_targets`/`pair_targets`) through the Hessian
callback, which does not otherwise have convenient access to the current outer point's raw `ν`.
`E[:, 1:ncore_core]` (the winner-conditioned economic columns) is **never** read in the `:winner_bin`
path -- only `E[:, ncore_core+1:NCORE]` is, and that slice is also unconditionally needed by `HMM`
(dense, out of scope) regardless of backend.

## Investigation: can the CM-grid `:winner_bin` backend also be enabled for CM+ZC?

**No, not without building a genuinely new cross primitive -- investigated, not assumed.**

`hessian_cm_structured!`'s CM-grid `H_EC` cross block reads `E = @view H[:, 2:1+NCORE]` where
`NCORE = cctx.NCORE` is the **widened** width (core + mean/pair) for CM+ZC -- confirmed by reading
the actual code (`cm_hessian_architectures.jl` line 741), not assumed. The dense path's
`build_bin_tables!(cctx, E, w; fill_S=true)` therefore folds mean/pair columns into the SAME
CM-grid cross-block computation the core (bilateral) columns feed into: CM+ZC's `H_EC` is really
`H_(core+meanpair, CM-grid)`, not just `H_(core, CM-grid)`.

The existing `:winner_bin` CM-grid primitive (`winner_pair_cross_hessian_fill!`/
`_cm_block!`, Section 2) is built from `wctx = build_winner_pair_ctx(cf)`, whose `ncolI` covers
**only** the core's own `ncore_core-1` bilateral(+cf) columns -- it has no representation of the
mean/pair columns at all. Relaxing `_cm_cross_hessian_wants_winner_bin`'s `ncore_core == NCORE`
guard to also accept `ncore_core < NCORE` would silently leave rows `ncore_core+1:NCORE` of the
CM-grid cross block **uncomputed** (a real correctness bug, not a knife edge) -- there is no way to
recover the mean/pair-vs-CM-grid contribution from the existing core-only `wctx`.

Making this safe would require a **third**, genuinely new cross primitive: mean/pair-restriction x
CM-grid-restriction, `Z'S*C` (`C` = the CM-grid bin-indicator restriction), which is a
restriction-vs-restriction cross (closer in spirit to the out-of-scope `H_RR` diagonal blocks than
to anything Sections 4/5 asked for) and was not attempted this session. `cctx.ncore_core ==
cctx.NCORE` therefore remains the CORRECT gate as-is, unchanged, and CM+ZC's CM-grid block
continues to fall back to dense automatically and correctly (`record_dense_cross_hessian_call!`),
exactly as it did before this section. This is a genuine scope boundary, not an oversight -- flagged
here for a future session rather than attempted under this task's time budget.

## Gates (both PASS)

**Standalone primitive** (`test_winner_pair_cross_hessian_zc_d4.jl`, shared with Section 5): D=4,
`K_mean=1/K_pair=1` and `K_mean=2/K_pair=2`, calibration + 2 perturbed points, against an
independently-built dense `(1/M)*E'*diag(S)*Z` reference (`E` read straight from a real dense-backend
`obj.H`, `Z` the same). **ALL PASS**, `max|Δ|` in `[1.8e-15, 1.4e-14]`.

**D=4 wiring** (`test_cm_meanzc_winner_bin_hez_wiring_d4.jl`): `K_mean=1/K_pair=1` and
`K_mean=2/K_pair=2`, both contrast conventions (anchored, orthonormal), calibration + 2 perturbed
points, both serial `hessian_cm_structured!` and threaded-production `hessian_cm_structured_v2!`,
plus a complete real KNITRO inner solve per (config,contrasts) pair (independent `aug`/`obj_cm` per
backend, mirroring Section 2's own methodology). **ALL PASS**, `max|ΔH|` in `[1.3e-15, 3.9e-14]`
against a Hessian scale of `3.17` to `2.6e3`. Complete inner solve status matches (`nStatus=0`
throughout) and dual point agrees to `<1e-8`.

**Real D=20/W=80,000/L=50** (`test_cm_meanzc_winner_bin_hez_wiring_d20.jl`,
`destination_sample=:exclude_row`, `K_mean=1/K_pair=1`, matching
`d20_meanzc_release_gates.jl`'s own Point A config): both contrasts, calibration + a near-delta=1
perturbed point, both architectures. **ALL PASS**, `max|ΔH|` in `[6.4e-13, 2.0e-12]` against a
Hessian scale of `~3970-5211`. Complete inner solve status matches (`nStatus=0` both backends, both
contrasts) and dual point agrees to `<7.4e-13`.

Timing (informational, not the default-flip criterion): at real D=20 with this small a restriction
width (`n_restr=10`), `:winner_bin`'s O(W*Ddest*n_restr) direct accumulation loop was **not**
faster than the dense BLAS gemm it replaces (roughly comparable wall-clock, sometimes slightly
slower for the serial architecture) -- unlike Section 2's CM-grid primitive, this block's dense
alternative is already a *small* gemm (`ncore x n_restr`, not `ncore x ncore`), so there is less
FLOP headroom to win back. This matches the task brief's own framing: the point of this backend is
eliminating the dense economic-column read, not necessarily raw speed. Warm
`hessian_cm_structured_v2!` (`:winner_bin`) allocates a stable ~10 MB/call at real D=20, with zero
persistent-workspace resizes across repeated calls (`zc_cross_scratch` object identity confirmed
stable).

## Runtime counters

Reuses the existing `dense_cross_hessian_calls`/`winner_cross_hessian_calls`/
`operator_cross_hessian_calls` counters (`no_dense_g_counters.jl`) -- no new counter names, per the
task's own "don't invent new counter names unless the existing ones genuinely don't fit" guidance.
D=4 wiring gate run: `dense_cross_hessian_calls=134` (every explicit `:dense_reference` half of
every comparison, PLUS the CM-grid block's own always-dense fallback for this family, see the
investigation above), `winner_cross_hessian_calls=50` (`=operator_cross_hessian_calls`, the H_EM
`:winner_bin` half only).

## Default backend

`CM_MEANZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT` (`core_exact_hessian.jl`) is flipped from
`:dense_reference` to `:winner_bin` in this same session, after both gates above passed to machine
precision.

## Not done / left for a future session

- The CM-grid cross-block for CM+ZC stays dense unconditionally (see investigation above) -- would
  need a new mean/pair-vs-CM-grid cross primitive.
- `H_RR` (mean/pair x mean/pair, `HMM`) stays dense BLAS, unchanged -- explicitly out of scope per
  the task brief.
- Rectangular (`D != Ddest`) D=4 configurations were not separately exercised (same gap Section 2's
  own doc already disclosed -- no rectangular D=4 CM context builder exists in this repository).
- `K_pair=0` (mean-only, no pair levels) was not separately gated for the `:winner_bin` backend --
  every gate config here used `K_pair >= 1`. The primitive's `has_cf`/pair-level handling is
  agnostic to `K_pair=0` (it would simply mean `n_restr = n_mean` and the pair-level loop in the
  wiring call site's concatenated `Z` never executes), so this is believed safe by construction but
  was not independently exercised this session.

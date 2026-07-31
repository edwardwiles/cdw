# Gravity sample/pivot call graph, 2026-07-31

Base: `remotes/cdw/production/fullA-exact` @ `81a673054551a5d91a672f1a7258655e4d868cb0`.

## The three distinct gravity-related computations in this codebase

1. **θ\* calibration regression** — `prestep/master_prestep.jl` (`thetaIn==0` branch, lines
   ~15-45). Fits `thetaHat` from raw data once, at calibration time.
2. **The pivot / actual enforced restriction** — `full_aod_diag/d4_exact/gravity_elimination.jl`
   (`build_pivot_elimination`/`build_pivot_elimination_cheap`, `pivot_reduce`/`pivot_expand`,
   `gravity_from_logz`/`gravity_offset`). Per this repo's own prior audit
   (`GRAVITY_MOMENT_FINAL_DECISION_2026-07-28.md`), **this is the actual gravity-equality
   enforcement mechanism** for every real outer iterate: the production driver
   (`cm_checkpoint.jl:1016`'s `run_cm_upper_checkpointed`, and the flexible-theta/gate scripts)
   calls `pivot_expand` before `CS.reconstruct_full`, so every real outer point it produces
   satisfies the gravity equality to machine precision by construction.
3. **The "inner" KNITRO moment** — `moments/newGravityMoment!.jl`, invoked via
   `compressed_gravity_raw`/`fill_gravity_column!` (`full_aod_diag/d4_exact/compressed_live.jl`).
   Per the same prior audit, this is **provably redundant for pivoted callers** (its value is
   ~0 by construction whenever the caller routed through `pivot_expand` first) and is
   **deliberately retained, unmodified, as a defensive check** for any caller that might bypass
   the pivot. It uses `within_transform_rect` (no masking support at all — cannot exclude the
   diagonal or an individual cell). **Out of scope for this task**: excluding Brazil→Korea here
   would require re-opening a broader architectural question this repo's own prior session
   explicitly scoped out ("What was NOT done this session" in `GRAVITY_MOMENT_FINAL_DECISION...`).
   Not touched by this task's change; noted here so it isn't mistaken for a missed site.

## Eligibility ("which (o,d) cells count") — every site, before this task's change

| # | Site | Feeds | Current logic |
|---|---|---|---|
| 1 | `full_aod_diag/gravity_tariff.jl:62` `_offdiag_mask(D,Ddest) = [o != d for o,d]` | `precompute_q_tilde` (:80,83), `gravity_value` (:113,116) | diagonal-only mask, shared helper |
| 2 | `full_aod_diag/d4_exact/context_real_d20.jl:57,114,172` | `q_tilde,N_obs = precompute_q_tilde(τ; exclude_diagonal=exclude_diagonal_gravity)` | threads a single `exclude_diagonal_gravity::Bool` flag through to `_offdiag_mask`'s consumer — **already a single shared flag**, just no per-cell exclusion capability |
| 3 | `full_aod_diag/d4_exact/context_real_d20.jl:120,147` | `row_idx = destination_sample==:exclude_row ? D20_REAL : nothing`; `Ddest = row_idx===nothing ? Dact : Dact-1` | ROW-destination drop, shared across every consumer via `ctx.D_dest`/`ctx.q_tilde` shape |
| 4 | `prestep/master_prestep.jl:25` `named_dest=filter(!=(row_idx),1:D)`, `:53` `diag_mask=[o!=d for o,d]` | θ\* regression sample | **independent duplicate re-derivation** — does NOT call `_offdiag_mask` or share code with site 1, despite being logically the same mask (confirmed: reproduces bit-identical `thetaHat` to the current manifest when both are given the same `exclude_diagonal_gravity`/`row_idx` settings) |
| 5 | `full_aod_diag/d4_exact/gravity_elimination.jl` (all functions) | pivot cell selection (`argmax(abs.(c))`, `c=μ.*q_tilde./N_obs`), pivot equality/derivative | **no eligibility logic of its own** — inherits it entirely and "for free" through `ctx.q_tilde`/`ctx.N_obs` (masked cells carry `q_tilde=0`⟹ can never be the argmax pivot, contribute 0 to the linear equality and its derivative) |
| 6 | `gravity_value` consumers: `oracle.jl:446`, `oracle_fast.jl:330`, `oracle_profiled.jl:124`, `compressed_live.jl:655`, `fast_range_screen.jl:715`, `infeasibility_screen.jl:848`, `run_smoothed_homotopy.jl:254`, `test_oracle.jl:27` | economic-adjacent gravity-value/screen computations | all already call `gravity_value(...; exclude_diagonal=get(ctx,:exclude_diagonal_gravity,false))` — automatically inherit any mask change made at site 1/2, **zero per-site changes needed** |
| 7 | `cc_algo/active_layout.jl:26,36,45` `active_origins`/`active_destinations`/`active_od_cells` | general economic layout (winner/witness/pairwise screens, `fast_range_screen.jl:925`, `cm_screen_bridge.jl:138`) | ROW-exclusion only, **deliberately does not** encode diagonal/BK exclusion — this is the *economic* layout (task §5 requires Brazil↔Korea stay active here) — confirmed not a site to touch |
| 8 | `full_aod_diag/d4_exact/oracle.jl:114` `context_fingerprint`, schema=`CONTEXT_FINGERPRINT_SCHEMA` | campaign/checkpoint cache keys | hashes `ctx.τ` and `row_idx`/`D_dest`, but **does NOT hash `exclude_diagonal_gravity` itself** — a pre-existing gap (two contexts differing only in `exclude_diagonal_gravity`, same τ, alias to the same fingerprint) that this task's mask-checksum addition also closes |
| 9 | `moments/localGravityMoment!.jl`, `localGravityCrossMoment!.jl`, `prepare_cc/nameMoments.jl:63,75` | legacy "Local ACR" moment | confirmed **inactive** in production (`AD_PARAMS.localGravityMoment=0`, `full_aod_diag/ad_benchmark/setup_context.jl:19`) — no change |

## This task's change

Add one function, `gravity_sample_mask(D, Ddest; exclude_diagonal, exclude_cells)`, in
`full_aod_diag/gravity_tariff.jl` next to (replacing) `_offdiag_mask`, taking an explicit
`exclude_cells::Vector{Tuple{Int,Int}}` (dest-space column indices) in addition to the existing
`exclude_diagonal` flag. Wire it into:
- Site 1/2 (`precompute_q_tilde`, `gravity_value`) — covers sites 5 and 6 automatically (no changes
  needed there).
- Site 4 (`master_prestep.jl`) — replace the duplicated `named_dest`/`diag_mask` derivation with a
  call to the same shared function, closing the only real duplication in the codebase.
- Site 8 (`context_fingerprint`) — add `exclude_diagonal_gravity` (closing the pre-existing gap)
  and the new `exclude_cells` mask to the hashed buffer; bump `CONTEXT_FINGERPRINT_SCHEMA`.

`bra_idx=3`, `kor_idx=14` (dest-space column also 14, since Korea's global index is < ROW's index
20 and ROW is always the omitted last column — see
`BRAZIL_KOREA_GRAVITY_EXCLUSION_PROVENANCE_2026-07-31.json`) are resolved via a new
`resolve_country_index` helper (no prior name→index resolver existed anywhere in the repo), not
hardcoded at any call site.

## Where `d`=ROW, diagonal cells, missing observations, and the pivot are currently handled
- `d`=ROW (destination index 20): dropped from the destination axis entirely (`Ddest=D-1`) whenever
  `destination_sample=:exclude_row`; never appears as a column in `q_tilde`/`N_obs`/`τ`'s dest-space
  slice. Origin index 20 (ROW-as-exporter) is retained.
- Diagonal cells (`o==d`): dropped from the gravity-identification sample only when
  `exclude_diagonal_gravity=true` (current sigma3/W500k campaign default) — `_offdiag_mask`/site 1.
  Retained in the underlying `τ`/`pi` matrices and in every economic (non-gravity) computation.
- Missing/invalid observations: none in the current `real_data/noah_D20` panel (verified: no
  NaN/Inf, `pi<=0`, or missing entries in `pi.csv`/`tau.csv`).
- The pivot cell: chosen by `argmax(abs.(c))` over the (masked) linear gravity coefficient vector
  `c` (`gravity_elimination.jl`), i.e. automatically re-selected among whatever cells the mask
  leaves eligible — no separate "pivot formula" exists to duplicate. **Before this task's change,
  the current frozen campaign's pivot cell is (origin=3=Brazil, destination=14=Korea)** — this task
  forces a pivot change, verified in `GRAVITY_PIVOT_EXCLUSION_GATE_2026-07-31.csv`.

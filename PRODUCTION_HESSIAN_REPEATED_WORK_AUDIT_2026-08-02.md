# Production Hessian Repeated-Work Audit (2026-08-02)

Scope: quantities computed more than once per Hessian callback, or duplicated across blocks with
potentially-different weighting/normalization conventions (the task brief explicitly warns against
merging such computations just because their names look similar without confirming identical
formulas).

## Confirmed NOT duplicated (shared, single-computation, by design)

- **H_EE core** (`_fill_cm_HEE!`): flexible_cm, common_frechet, and cm_meanzc all call the exact
  SAME function for their core economic block -- not three copies. Confirmed by direct read
  (`cm_hessian_architectures.jl:813`, called from `hessian_cm_structured!`/`_v2!` for all three).
- **H_CC**: extracted into a single shared `fill_cm_HCC!` (`cm_hessian_architectures.jl`), used by
  flexible_cm, common_frechet, AND cm_meanzc -- the file's own header comment documents this was
  itself a fix for a prior triplicated copy ("harmonization task 2026-07-28... previously a third
  verbatim copy of this loop").
- **Bin-contingency tables** (`Ttab`/`Stab`/`CScum`/`CT`): built ONCE per callback via
  `build_bin_tables_threaded!`/`prefix_sum_tables_threaded!`, then read by H_EC, H_CC, AND (for
  common_frechet) the level blocks -- not rebuilt per consuming block. Confirmed: `_fill_cm_HEE!`
  and `_fill_frechet_level_blocks!` both read `cctx.CScum`/`cctx.CT` without re-triggering a build.
- **`ddPsi!(arg2, arg0)`** (the Ψ'' weights, `w`): computed once at the top of
  `hessian_cm_structured_v2!`, passed as `w` to every subsequent block -- not recomputed per block.
- **`zc_restriction_operator.jl`'s `ZcS`** (S-weighted, target-centered restriction columns): the
  root-cause fix documented at `cm_hessian_architectures.jl` (`refresh_zc_centered!` call site,
  "diagnose-optimize/HZZ-BLAS-and-HCZ-prep-2026-07-29, Part B") establishes that this is computed
  ONCE and shared correctly by BOTH H_ZZ and H_CZ (cm_meanzc) -- notably, this fix's own history is
  a cautionary tale in the OTHER direction (a PRIOR version incorrectly skipped this shared
  computation for some backends, silently corrupting H_CZ) -- i.e. this codebase has direct,
  documented experience with the exact failure mode this audit category warns about, and the
  current code is the corrected state.

## New finding this audit pass: the two allocation hotspots WERE a repeated-work variant

Both allocation fixes in this pass (commit `8f1151e`) are also, precisely, repeated-work findings
under this section's own definition ("computed more than once per Hessian callback... or duplicated
across blocks"):

1. **cm_meanzc's `HEE[ncore+1:NCORE,1:ncore] .= transpose(HEM)`**: not duplicated computation
   per se, but a single necessary O(ncore*(NCORE-ncore)) COPY that was, via the aliasing-defensive
   broadcast, effectively implemented as "materialize a full temporary copy, then copy that into
   the destination" -- i.e. the data was written twice per callback where once suffices. Fixed.
2. **common_frechet's `cctx.R' * Hraw_cmlevel`**: the SAME small (`nO`-length) matrix-vector
   product recomputed via a fresh allocation on all `L*L=2500` iterations of the H_CM,level loop,
   where a single persistent buffer + `mul!` suffices (each iteration's Hraw_cmlevel differs, so
   the mul! itself must still run 2500 times -- what was duplicated was the ALLOCATION, not the
   arithmetic; this is `CALLBACK_TEMPORARY` misclassified as needing 2500 fresh buffers instead of
   1 reused one). Fixed.

## Candidates flagged, not yet fully traced (deferred -- not confirmed as real duplication)

- `cm_originzc_target_layout.jl:85/99` (`mean_targets`/`pair_targets`) and
  `cm_meanzc_moments.jl:68` (`packed_pair_index`) appear as small allocation sites in BOTH
  cm_meanzc's and origin_zc's `Profile.Allocs` breakdowns (see
  `PRODUCTION_HESSIAN_ALLOCATION_BASELINE_2026-08-02.csv` and the underlying diagnostic logs).
  Whether these represent genuinely-necessary per-callback recomputation (the nu/eta targets DO
  depend on the current dual point in general) or an opportunity to cache across calls when the
  targets haven't changed has NOT been traced to a formula-level confirmation in this pass --
  flagged as a follow-up, not proposed as a fix here, per this section's own "require exact formula
  mapping before sharing" instruction (I have not yet done that mapping for these two sites).
- Whether `_fill_cm_HEE!`'s H_ZZ dispatch (`zc_gram_dispatch!`) and the H_CZ prep step
  (`hcz_prep_dispatch!`, cm_meanzc only) share any weight/normalization computation that could be
  hoisted is not yet traced at the same resolution as the H_EE/H_CC sharing above -- deferred.

No case was found in this pass where two blocks compute what LOOKS like the same quantity but
actually use different weighting/normalization conventions (the specific trap this section warns
against) -- every shared computation found was confirmed, by reading the consuming code, to use
the identical formula and be a deliberate single-computation design, not an accidental merge.

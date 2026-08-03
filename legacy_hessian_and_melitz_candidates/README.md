# Legacy Hessian and Melitz candidate scripts (archived 2026-08-03)

These 48 files were copied out of 30 now-pruned branches during the 2026-08-03
repository cleanup, because they contain genuine Hessian-block profiling/timing
work or completed Melitz research that was never merged into any of the four
kept branches (FULL production 146b10e, REDUCED prototype 7ec5c6c, Melitz
closeout 7201752, Melitz live campaign
campaign/melitz-suffix-group-delta0p5-production-2026-08-02).

None of this is wired into any production driver. It is preserved for
reference only. SOURCE_MANIFEST.tsv records the exact source branch and SHA
for every file; the full commit history for all of it also still exists in
the pre-cleanup git bundles at /bbkinghome/edav/repo_salvage/.

## full_aod_diag/d4_exact/ -- FULL-side Hessian architecture candidates (30 files)

Exploratory/benchmark code from the path to the current production Hessian
architecture (cm_hessian_architectures.jl, chunked_hessian.jl,
cm_hessian_subblock_profiling.jl, all still in production). Two representative
examples, read in full before archiving:

- hessian_syrk.jl explicitly self-labels "BENCHMARK/CANDIDATE CODE ONLY --
  not wired into any production driver" -- it benchmarks BLAS.syrk! vs.
  BLAS.gemm! for the dense Hessian's W-contraction.
- winner_pair_hessian.jl (+ _parallel/_threaded/_wiring variants,
  validate_winner_pair_hessian_d4/d20.jl) is a "Phase 5" alternative H_EE
  construction built directly from CompressedFactual, never adopted into
  production; references a derivation doc
  (docs/UNRESTRICTED_WINNER_PAIR_HESSIAN_DERIVATION_2026-07-25.md, not
  copied here, still in the source branch history).

The hzz_*/hcz_*/hez_* files are dated bake-off/benchmark/correctness
scripts (2026-07-28 -> 2026-08-01) comparing candidate H_ZZ/H_CZ/H_EZ backends
against reference implementations -- the kind of block-by-block Hessian timing
work worth keeping as reference even though superseded by whatever backend
production ultimately adopted.

## src/melitz/ + scripts/melitz_* -- Melitz Hessian and envelope-gradient work (18 files)

- exact_hessian_hybrid_polish.jl and exact_reduced_hessian.jl are NOT
  throwaway benchmarks. exact_reduced_hessian.jl is a from-first-principles
  re-derivation of the exact Schur-complement Hessian of Melitz's fixed-q
  A-middle loop, cross-checked against the numerical adjoint identity and
  finite differences per its own docstring, and documents a real sign-
  convention bug it found in a prior task brief. exact_hessian_hybrid_polish.jl
  builds on it (rank-revealing eigendecomposition + stability report + a
  KNITRO Hessian-vector-product "polish" phase). Worth a real look before
  assuming Melitz's current production solver doesn't need this.
- profiled_envelope_gradient.jl plus the melitz_profiled_envelope_*_2026-07-31.jl
  scripts and melitz_profile38_*_2026-08-01.jl scripts are the profiling/
  diagnostic harnesses used to develop and validate that envelope-gradient work.

## What this is NOT

This is not a claim that any of this should be revived or wired into
production -- that determination needs someone who understands the current
Melitz/Hessian architecture to actually read it, which this cleanup pass did
not do exhaustively (only the largest/most-referenced files were read in full).

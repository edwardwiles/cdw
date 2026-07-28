# D=20 profiling task (cm_meanzc/origin_zc H_EC/H_EZ/H_CZ/H_ZZ), 2026-07-28.
#
# Opt-in Ref{Any} stash for live production-context handles. Per this task's own brief (and
# STRUCTURED_CROSS_HESSIAN_MASTER_REPORT_2026-07-28.md §5): calling the LOW-LEVEL inner-solve
# helpers directly (archC_meanzc_base_state/archC_meanzc_verified_state, or building a context via
# build_cm_meanzc_production_context and then invoking cm_meanzc_production_value_verified myself)
# reliably raises KNITRO's KN_RC_CALLBACK_ERR (nStatus=-500) for cm_meanzc, confirmed even fully
# isolated in its own process, at both D=4 and D=20. origin_zc is not known to have this problem,
# but the task spec still requires routing through the real public driver as the primary path.
#
# These two Refs are populated by a ONE-LINE edit at the tail of
# build_cm_meanzc_production_context (cm_meanzc_production.jl) / build_originzc_production_context
# (cm_originzc_production.jl) -- the construction functions the real drivers
# (run_cm_upper_checkpointed / run_originzc_upper_checkpointed) already call internally -- never by
# calling those constructors a second time ourselves. After a real driver run completes (or is
# stopped by maxtime_real), the stashed handle gives direct read access to the SAME live cctx/octx
# and obj (with obj.arg2 = S already filled by the real KNITRO Hessian callback's own ddPsi! call)
# that the production solve actually used -- safe to feed into the PURE, KNITRO-free standalone
# sub-block kernels (fill_core_hessian_upper!, winner_pair_cross_hessian_fill!/_threaded!,
# winner_pair_cross_hessian_zc_block!/_threaded!, bin_zc_cross_hessian_fill!/_threaded!,
# zc_gram_blas_syrk!/_gemm!/zc_gram_threaded_packed!) for timing/correctness microbenchmarks,
# exactly the same functions diag_subblock_profile_2026-07-28.jl already calls this way for
# origin_zc -- no second KNITRO inner solve is ever triggered by this profiling.
const CMZC_LIVE_PCX_STASH = Ref{Any}(nothing)
const ORIGINZC_LIVE_PCX_STASH = Ref{Any}(nothing)

"Opt-in gate -- true (default) stashes on every construction call; false is a zero-overhead opt-out (skips the single Ref assignment)."
const STASH_LIVE_PCX_ENABLED = Ref{Bool}(true)

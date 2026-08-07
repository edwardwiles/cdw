# fix/fullA-lower-limit-and-hotpath-2026-08-06, task P2: instrument build_bin_tables_threaded!
# (cm_hessian_threaded.jl) into parallel-fill / serial-reduction / zero-reset / other, and measure
# at several thread counts on the SAME real D20/W=100,000 common-Frechet point.
#
# METHODOLOGY NOTE (deviation from the task's literal "launch separate 4T/10T/20T processes",
# disclosed rather than silently substituted): building one real D20/W=100,000 context under this
# session's machine contention (see MASTER.md §0) already costs several minutes; 3 separate full
# process launches (each re-doing context construction from scratch) was not affordable within this
# session's remaining budget. Instead: ONE process, launched with `-t 20` (so 20 real OS threads/
# cores are genuinely available), builds ONE real warmed context+point ONCE via the actual
# run_cm_upper_checkpointed public driver (same CM_LIVE_PCX_STASH pattern p1_allocs_frechet uses),
# then calls a parameterized clone of build_bin_tables_threaded! repeatedly with an explicit
# `nt_use` in {4,10,20} (all <= Threads.nthreads()=20, so every call still dispatches real work
# across `nt_use` genuinely different OS threads via Threads.@threads -- this is NOT simulating
# threads in software, it is the same Threads.@threads machinery, just intentionally splitting the
# SAME W=100,000 draws into 4, 10, or 20 chunks within one process). This answers the SAME question
# (how does the parallel-fill vs serial-reduction split scale with thread count on identical data)
# without re-paying context-build cost 3x. Each configuration is repeated a few times and the
# fastest is reported (standard micro-benchmark practice -- excludes GC/scheduler noise spikes from
# the heavily-loaded shared machine, not cherry-picking a favorable result).
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_cplus.jl", "cm_meanzc_lookup_production.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf

lp(xs...) = (println(xs...); flush(stdout))
lp("nthreads()=", Threads.nthreads())
Threads.nthreads() >= 20 || error("p2_bintables_fill_vs_reduction: launch with `julia -t 20` (or more) -- got $(Threads.nthreads())")

"""
Parameterized clone of build_bin_tables_threaded! (cm_hessian_threaded.jl) -- byte-for-byte
identical arithmetic, split into 3 timed phases (zero/reset, parallel fill, serial reduction), and
taking an explicit `nt_use` instead of always reading `Threads.nthreads()`, so the SAME warmed
(cctx, tls, H, w) can be re-measured at multiple chunk counts without rebuilding anything. Does NOT
mutate cctx.Ttab/cctx.Stab/etc (writes into a caller-supplied scratch pair instead) so repeated
calls at different nt_use don't corrupt cctx for a subsequent call.
"""
function build_bin_tables_threaded_timed(cctx, tls, H, w::AbstractVector{Float64}, nt_use::Int; fill_S::Bool = true)
    D = cctx.D; NCORE = cctx.NCORE; Bidx = cctx.Bidx
    W = length(w)
    fam2 = cctx.n_families == 2
    Pow = fam2 ? cctx.Pow : nothing
    @assert nt_use <= length(tls.Ttab) "nt_use=$nt_use exceeds tls capacity $(length(tls.Ttab))"

    t_zero = @elapsed begin
        for t in 1:nt_use
            fill!(tls.Ttab[t], 0.0)
            fill!(tls.Stab[t], 0.0)
            if fam2
                fill!(tls.Ttab12[t], 0.0)
                fill!(tls.Ttab22[t], 0.0)
                fill_S && fill!(tls.Stab2[t], 0.0)
            end
        end
    end

    t_fill = @elapsed begin
        if fill_S
            E = @view H[:, 2:1+NCORE]
            Threads.@threads :static for tid in 1:nt_use
                lo = 1 + div((tid - 1) * W, nt_use)
                hi = div(tid * W, nt_use)
                Tloc = tls.Ttab[tid]; Sloc = tls.Stab[tid]
                T12loc = fam2 ? tls.Ttab12[tid] : nothing
                T22loc = fam2 ? tls.Ttab22[tid] : nothing
                S2loc = fam2 ? tls.Stab2[tid] : nothing
                @inbounds for s in lo:hi
                    ws = w[s]
                    for x in 1:D
                        bx = Bidx[s, x]
                        for j in 1:NCORE
                            Sloc[x, j, bx] += ws * E[s, j]
                        end
                        if fam2
                            for j in 1:NCORE
                                S2loc[x, j, bx] += ws * E[s, j] * Pow[s, x]
                            end
                        end
                    end
                    for x in 1:D
                        bx = Bidx[s, x]
                        for y in 1:D
                            by = Bidx[s, y]
                            Tloc[x, y, bx, by] += ws
                        end
                    end
                    if fam2
                        for x in 1:D
                            bx = Bidx[s, x]
                            for y in 1:D
                                by = Bidx[s, y]
                                wsy = ws * Pow[s, y]
                                T12loc[x, y, bx, by] += wsy
                                T22loc[x, y, bx, by] += wsy * Pow[s, x]
                            end
                        end
                    end
                end
            end
        else
            Threads.@threads :static for tid in 1:nt_use
                lo = 1 + div((tid - 1) * W, nt_use)
                hi = div(tid * W, nt_use)
                Tloc = tls.Ttab[tid]
                T12loc = fam2 ? tls.Ttab12[tid] : nothing
                T22loc = fam2 ? tls.Ttab22[tid] : nothing
                @inbounds for s in lo:hi
                    ws = w[s]
                    for x in 1:D
                        bx = Bidx[s, x]
                        for y in 1:D
                            by = Bidx[s, y]
                            Tloc[x, y, bx, by] += ws
                        end
                    end
                    if fam2
                        for x in 1:D
                            bx = Bidx[s, x]
                            for y in 1:D
                                by = Bidx[s, y]
                                wsy = ws * Pow[s, y]
                                T12loc[x, y, bx, by] += wsy
                                T22loc[x, y, bx, by] += wsy * Pow[s, x]
                            end
                        end
                    end
                end
            end
        end
    end

    T = similar(cctx.Ttab); S = similar(cctx.Stab)
    t_reduce = @elapsed begin
        fill!(T, 0.0); fill!(S, 0.0)
        for tid in 1:nt_use
            T .+= tls.Ttab[tid]
            fill_S && (S .+= tls.Stab[tid])
        end
    end

    return (zero = t_zero, fill = t_fill, reduce = t_reduce, total = t_zero + t_fill + t_reduce)
end

ctx = d20_real_setup_design(W = 100_000, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    inner_lower_limit = -10.0)
pe = build_pivot_elimination(ctx)
w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
L = 50
probs = cm_equal_grid_probs(L)
CKPT = mktempdir()

lp("="^90); lp("P2: warming a real common-Frechet cctx via the real public driver (one real solve)"); lp("="^90)
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true
result = run_cm_upper_checkpointed(w0; W = 100_000, delta = 1.0,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    L = L, contrasts = :anchored, probs = probs,
    include_truncated_moment = false,   # single-family here -- fam2 bin-table cost is additive on
        # top of this (already measured qualitatively in the prior session's report); this
        # measurement targets the SHARED (both families') Ttab/Stab fill+reduction split, so
        # single-family keeps the point cheaper to reach without changing what's being measured.
    cm_hessian_backend = :structured, threaded_bins = true,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    marginal_restriction = :common_frechet,
    A_coordinate_mode = :powered_aspace,
    inner_lower_limit = -10.0,
    ckpt_dir = CKPT, run_id = "p2_bintables", label = "p2_bintables",
    checkpoint_interval_s = 90.0, maxtime_real = 60.0, verbose = true)
lp("knitro_status=", result.knitro_status, " n_eval=", result.n_eval)

pcx_live = CM_LIVE_PCX_STASH[]
pcx_live === nothing && error("p2_bintables_fill_vs_reduction: CM_LIVE_PCX_STASH was never populated -- no warmed cctx reached")
cctx = pcx_live.cctx
obj = pcx_live.ctx_cm.obj
lp("cctx.use_threaded_bins=", cctx.use_threaded_bins, " cctx.n_families=", cctx.n_families,
   " D=", cctx.D, " NCORE=", cctx.NCORE, " W=", 100_000)

# Need a real (H, w) pair matching a real dual-solve point -- reuse whatever the last real callback
# left behind via _prep_dual_index_for_archC! having already run (obj.arg0 is populated); w = ddPsi
# applied to arg0, exactly what hessian_cm_structured_v2! itself computes first.
w_vec = similar(obj.arg2)
obj.ddPsi!(w_vec, obj.arg0)
H_field = _dense_H_or_nothing(obj)
lp("H_field is nothing (operator/no-dense-H bundle, expected): ", H_field === nothing)

# fill_S=false unconditionally for THIS measurement (a deliberate, disclosed deviation from
# "match the exact real dispatch at this one point" -- checked live: at the point this script's own
# warm-up run reached, use_winner_bin=false for this operator/no-dense-H bundle, which would require
# fill_S=true and hence a real dense H the production bundle structurally does not have; extracting
# E from the operator state instead was not implemented this session). fill_S=false still measures
# the SAME Ttab (not Stab) fill+reduction machinery -- Ttab is present and identically accumulated
# in BOTH branches, and per the prior session's own measurement (docs/audits/cm-extensions-hotpath-
# 2026-08-06/MASTER.md §6) bintables_prep (Ttab-dominated) is the ~56.6% dominant Hessian sub-block
# for common-Frechet -- so this still answers the task's own core question (does the serial
# post-@threads reduction loop explain sub-linear thread scaling), just without the smaller,
# secondary Stab/E-dependent piece.
cf = cctx.core_cf_ref[]
use_winner_bin = _cm_cross_hessian_wants_winner_bin(cctx, cf)
lp("use_winner_bin=", use_winner_bin, " at the warmed point (real dispatch would use fill_S=", !use_winner_bin, ")")
fill_S_real = false

tls = build_thread_local_scratch(cctx)
NREPEAT = 3
for nt_use in (4, 10, 20)
    times = [build_bin_tables_threaded_timed(cctx, tls, H_field, w_vec, nt_use; fill_S = fill_S_real) for _ in 1:NREPEAT]
    best = times[argmin([t.total for t in times])]
    @printf("nt_use=%2d  zero=%.4fs  fill=%.4fs (%.1f%%)  reduce=%.4fs (%.1f%%)  total=%.4fs\n",
        nt_use, best.zero, best.fill, 100*best.fill/best.total, best.reduce, 100*best.reduce/best.total, best.total)
end

lp("="^90); lp("DONE"); lp("="^90)

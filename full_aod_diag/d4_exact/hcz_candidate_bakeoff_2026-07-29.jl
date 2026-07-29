# Part C — H_CZ prep candidate bakeoff (2026-07-29).
#
# Correctness + performance comparison of the four H_CZ "prep" (raw bin-feature reduction,
# `T[o,b,j] = Σ_{w: bin_o(w)=b} S_w Z[w,j]`) candidates named in
# docs/PART_C_HCZ_FORMULA_2026-07-29.md / docs/PART_A_LABEL_MAPPING_2026-07-29.md:
#   (a) bin_zc_cross_hessian_fill!            -- serial, origin-owned reference/anchor
#   (b) bin_zc_cross_hessian_fill_threaded!   -- threaded, origin-owned (current production)
#   (c) bin_zc_cross_hessian_fill_drawchunk!  -- NEW threaded, draw-chunk thread-local (Candidate 2)
#   (d) bin_zc_cross_hessian_fill_sparse!     -- NEW sparse one-hot SpMM (Candidate 4)
#
# Uses ONLY the construction-only path (build_cm_meanzc_bin_ctx / build_cm_meanzc_augmented_obj,
# neither of which ever calls KNITRO) plus synthetic, task-sanctioned random S vectors -- the real
# archC_meanzc_base_state/_verified_state direct-call entry points are pre-existing broken on this
# clean production HEAD (see docs/PREEXISTING_DIRECT_CALL_FAILURE_2026-07-29.md); this script does
# NOT attempt to work around that, per this task's own explicit instruction. "solver-derived S" in
# the task brief is therefore substituted with a SECOND independent random S draw at every config --
# flagged explicitly in every row's `s_variant` column, never claimed to be solver-derived.
#
# D=4: only `d4_exact_setup` exists in this codebase (no destination-exclusion/rectangular concept
# at D=4 -- `context.jl` has no `destination_sample`/`exclude_row` kwarg at all, confirmed by direct
# grep, not assumed). The task asked for "rectangular AND square" layouts at D=4; this script
# instead exercises the "rectangular vs square" axis at real D=20 (`d20_real_setup`'s own
# `destination_sample = :exclude_row | :all_legacy`, the ONLY place that axis exists in this
# codebase). This is not a corner cut for convenience: the H_CZ prep formula itself
# (`T[o,b,j] = Σ_w S_w Z[w,j]` for `bin_o(w)=b`) depends only on `ctx.D` (CM origins), `Bidx` (W x D),
# and `ZcS` (W x n_z) -- `D_dest` (the axis :exclude_row/:all_legacy actually vary) never enters
# this kernel's shape or math at all, so a D=4-scale "rectangular" config would not exercise any
# code path in these four candidates that the D=20 exclude_row/all_legacy contrast doesn't already
# cover more realistically.

const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "threaded_cross_hessian.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "cm_outer_driver.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "hcz_sparse_spmm_candidate_2026-07-29.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random, Statistics, SparseArrays
using Base.Threads: nthreads

println("Threads.nthreads() = ", nthreads()); flush(stdout)

nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]

# ------------------------------------------------------------------------------------------------
# Result row accumulator + CSV writer (no CSV.jl dependency in this Project.toml -- plain DelimitedFiles-
# style manual write, matching this codebase's own existing bench_*.jl idiom of writing CSV by hand).
# ------------------------------------------------------------------------------------------------
mutable struct BakeoffRow
    config::String
    W::Int
    D::Int
    K_mean::Int
    K_pair::Int
    contrasts::String
    s_variant::String
    candidate::String
    maxdiff::Float64
    tol_kind::String       # "bit_identical" | "relative_tol"
    pass::Bool
    finite_ok::Bool
    repeat_deterministic::Bool
    aliases_input::Bool
    time_s::Float64        # NaN if not timed
    speedup_vs_ref::Float64  # NaN if not timed
    allocated_bytes::Float64  # NaN if not measured
end

const ROWS = BakeoffRow[]

function push_row!(; config, W, D, K_mean, K_pair, contrasts, s_variant, candidate, maxdiff, tol_kind, pass,
        finite_ok, repeat_deterministic, aliases_input, time_s = NaN, speedup_vs_ref = NaN, allocated_bytes = NaN)
    push!(ROWS, BakeoffRow(config, W, D, K_mean, K_pair, string(contrasts), s_variant, candidate, maxdiff, tol_kind,
        pass, finite_ok, repeat_deterministic, aliases_input, time_s, speedup_vs_ref, allocated_bytes))
end

function write_csv(path::AbstractString, rows::Vector{BakeoffRow})
    open(path, "w") do io
        println(io, "config,W,D,K_mean,K_pair,contrasts,s_variant,candidate,maxdiff,tol_kind,pass,finite_ok,repeat_deterministic,aliases_input,time_s,speedup_vs_ref,allocated_bytes")
        for r in rows
            @printf(io, "%s,%d,%d,%d,%d,%s,%s,%s,%.6e,%s,%s,%s,%s,%s,%.6g,%.6g,%.6g\n",
                r.config, r.W, r.D, r.K_mean, r.K_pair, r.contrasts, r.s_variant, r.candidate,
                r.maxdiff, r.tol_kind, r.pass, r.finite_ok, r.repeat_deterministic, r.aliases_input,
                r.time_s, r.speedup_vs_ref, r.allocated_bytes)
        end
    end
end

# ------------------------------------------------------------------------------------------------
# Construction helpers (pure, no KNITRO solve anywhere in this file).
# ------------------------------------------------------------------------------------------------

"Build ctx/aug/cctx via the construction-only path. `size_kind` in (:d4, :d20)."
function build_ctx_aug_cctx(size_kind::Symbol; W::Int = 0, L::Int, K_mean::Int, K_pair::Int,
        contrasts::Symbol, destination_sample::Symbol = :exclude_row)
    if size_kind === :d4
        ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    elseif size_kind === :d20
        ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = destination_sample)
    else
        error("build_ctx_aug_cctx: unknown size_kind=$size_kind")
    end
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    return ctx, aug, cctx
end

"""
Refresh `cctx.hzz_centered.ZcS` for a synthetic `S` (task-sanctioned random per-draw weight,
`S >= 0` is the only structural requirement -- Ψ'' is a convex-conjugate second derivative). Bypasses
the broken direct-call solver path entirely -- pure construction, matches
`_fill_cm_HEE!`'s own H_ZZ/H_CZ prep sequence (cm_hessian_architectures.jl:815-823) exactly.
"""
function refresh_synthetic_zcS!(cctx, K_mean::Int, S::Vector{Float64})
    ν0 = nu0vec(K_mean)
    cctx.nu_ref[] = collect(ν0)
    op = cctx.hzz_zc_op
    W = length(S)
    refresh_zc_targets!(cctx.hzz_zc_ws, op, cctx.hzz_zc_layout, cctx.nu_ref[])
    cctx.hzz_centered = ensure_zc_centered_scratch!(cctx.hzz_centered, op, W)
    refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, S; fill_S = true, cache_across_callbacks = false)
    return cctx.hzz_centered.ZcS
end

synthetic_S(W::Int; scale = 2.0, floor_ = 1e-3) = rand(W) .* scale .+ floor_

# ------------------------------------------------------------------------------------------------
# Core per-config correctness (+ optional timing) routine.
# ------------------------------------------------------------------------------------------------
function run_config!(config_label::String, ctx, aug, cctx, S::Vector{Float64}, s_variant::String;
        workers::Int, do_timing::Bool = false, timing_reps::Int = 5)
    D = ctx.D
    L = aug.L
    op = cctx.hzz_zc_op
    nz = n_restriction(op)
    W = length(S)
    Bidx = cctx.Bidx

    ZcS = refresh_synthetic_zcS!(cctx, aug.K_mean, S)
    @assert size(Bidx, 1) == W
    @assert size(ZcS, 1) == W && size(ZcS, 2) >= nz

    workers_eff = min(workers, nthreads(), D == 0 ? 1 : max(1, D))
    # NOTE: draw-chunk/sparse candidates chunk over W (or use a global SpMM), not D, so they do not
    # need workers<=D; only the origin-owned threaded candidate (b) needs workers<=D to have
    # non-empty chunks. Use workers_b = min(workers, nthreads(), D) for (b) specifically, and
    # workers_c = min(workers, nthreads()) for (c).
    workers_b = min(workers, nthreads(), D)
    workers_c = min(workers, nthreads())

    # ---- fresh scratch per candidate (so post-hoc comparison reads each candidate's own output,
    # never overwritten by a later candidate) ----
    ws_ref = BinZCrossScratch(D, L, nz)
    ws_thr = BinZCrossScratch(D, L, nz)
    ws_dc = BinZCrossScratch(D, L, nz)
    ws_sp = BinZCrossScratch(D, L, nz)
    dc = BinZCrossDrawChunkScratch(D, L, nz, workers_c)
    sc = build_bin_zc_sparse_scratch(Bidx, D, L, nz)

    # ---- alias checks (should never alias — flag if they do) ----
    alias_ref = Base.mightalias(ws_ref.ZBinTab, ZcS) || Base.mightalias(ws_ref.ZBinTab, Bidx)
    alias_thr = Base.mightalias(ws_thr.ZBinTab, ZcS) || Base.mightalias(ws_thr.ZBinTab, Bidx)
    alias_dc = Base.mightalias(ws_dc.ZBinTab, ZcS) || Base.mightalias(ws_dc.ZBinTab, Bidx) || Base.mightalias(dc.local_tabs, ZcS)
    alias_sp = Base.mightalias(ws_sp.ZBinTab, ZcS) || Base.mightalias(ws_sp.ZBinTab, Bidx) || Base.mightalias(sc.Tflat, ZcS)

    # ---- fill each candidate ----
    bin_zc_cross_hessian_fill!(ws_ref, Bidx, ZcS)
    bin_zc_cross_hessian_fill_threaded!(ws_thr, Bidx, ZcS; workers = workers_b)
    bin_zc_cross_hessian_fill_drawchunk!(ws_dc, dc, Bidx, ZcS; workers = workers_c)
    bin_zc_cross_hessian_fill_sparse!(ws_sp, sc, ZcS)

    # ---- repeat-call determinism (each candidate individually, on identical input) ----
    ws_ref2 = BinZCrossScratch(D, L, nz); bin_zc_cross_hessian_fill!(ws_ref2, Bidx, ZcS)
    ws_thr2 = BinZCrossScratch(D, L, nz); bin_zc_cross_hessian_fill_threaded!(ws_thr2, Bidx, ZcS; workers = workers_b)
    ws_dc2 = BinZCrossScratch(D, L, nz); dc2 = BinZCrossDrawChunkScratch(D, L, nz, workers_c)
    bin_zc_cross_hessian_fill_drawchunk!(ws_dc2, dc2, Bidx, ZcS; workers = workers_c)
    ws_sp2 = BinZCrossScratch(D, L, nz); sc2 = build_bin_zc_sparse_scratch(Bidx, D, L, nz)
    bin_zc_cross_hessian_fill_sparse!(ws_sp2, sc2, ZcS)

    det_ref = ws_ref.ZBinTab == ws_ref2.ZBinTab && ws_ref.ZBinCScum == ws_ref2.ZBinCScum
    det_thr = ws_thr.ZBinTab == ws_thr2.ZBinTab && ws_thr.ZBinCScum == ws_thr2.ZBinCScum
    det_dc = ws_dc.ZBinTab == ws_dc2.ZBinTab && ws_dc.ZBinCScum == ws_dc2.ZBinCScum
    det_sp = ws_sp.ZBinTab == ws_sp2.ZBinTab && ws_sp.ZBinCScum == ws_sp2.ZBinCScum

    # ---- finiteness ----
    fin_ref = all(isfinite, ws_ref.ZBinTab) && all(isfinite, ws_ref.ZBinCScum)
    fin_thr = all(isfinite, ws_thr.ZBinTab) && all(isfinite, ws_thr.ZBinCScum)
    fin_dc = all(isfinite, ws_dc.ZBinTab) && all(isfinite, ws_dc.ZBinCScum)
    fin_sp = all(isfinite, ws_sp.ZBinTab) && all(isfinite, ws_sp.ZBinCScum)

    # ---- correctness vs reference ----
    maxdiff_tab(a, b) = maximum(abs.(a .- b))
    md_thr = max(maxdiff_tab(ws_ref.ZBinTab, ws_thr.ZBinTab), maxdiff_tab(ws_ref.ZBinCScum, ws_thr.ZBinCScum))
    md_dc = max(maxdiff_tab(ws_ref.ZBinTab, ws_dc.ZBinTab), maxdiff_tab(ws_ref.ZBinCScum, ws_dc.ZBinCScum))
    md_sp = max(maxdiff_tab(ws_ref.ZBinTab, ws_sp.ZBinTab), maxdiff_tab(ws_ref.ZBinCScum, ws_sp.ZBinCScum))
    scale = max(1.0, maximum(abs.(ws_ref.ZBinCScum)))

    pass_thr = md_thr == 0.0
    pass_dc = md_dc < HCZ_CANDIDATE_TOL * scale
    pass_sp = md_sp < HCZ_CANDIDATE_TOL * scale

    D_, K_mean, K_pair = ctx.D, aug.K_mean, aug.K_pair
    push_row!(; config = config_label, W = W, D = D_, K_mean = K_mean, K_pair = K_pair, contrasts = aug.contrasts,
        s_variant = s_variant, candidate = "origin_owned_serial_REF", maxdiff = 0.0, tol_kind = "reference",
        pass = true, finite_ok = fin_ref, repeat_deterministic = det_ref, aliases_input = alias_ref)
    push_row!(; config = config_label, W = W, D = D_, K_mean = K_mean, K_pair = K_pair, contrasts = aug.contrasts,
        s_variant = s_variant, candidate = "origin_owned_threaded", maxdiff = md_thr, tol_kind = "bit_identical",
        pass = pass_thr, finite_ok = fin_thr, repeat_deterministic = det_thr, aliases_input = alias_thr)
    push_row!(; config = config_label, W = W, D = D_, K_mean = K_mean, K_pair = K_pair, contrasts = aug.contrasts,
        s_variant = s_variant, candidate = "draw_chunk_thread_local", maxdiff = md_dc, tol_kind = "relative_tol",
        pass = pass_dc, finite_ok = fin_dc, repeat_deterministic = det_dc, aliases_input = alias_dc)
    push_row!(; config = config_label, W = W, D = D_, K_mean = K_mean, K_pair = K_pair, contrasts = aug.contrasts,
        s_variant = s_variant, candidate = "sparse_spmm", maxdiff = md_sp, tol_kind = "relative_tol",
        pass = pass_sp, finite_ok = fin_sp, repeat_deterministic = det_sp, aliases_input = alias_sp)

    @printf("  [%s / %s]  thr maxdiff=%.3e (bit-exact expect 0.0)  dc maxdiff=%.3e (tol=%.1e*scale=%.3e)  sp maxdiff=%.3e\n",
        config_label, s_variant, md_thr, md_dc, HCZ_CANDIDATE_TOL, scale, md_sp)
    println("    PASS: thr=$pass_thr dc=$pass_dc sp=$pass_sp | finite: ref=$fin_ref thr=$fin_thr dc=$fin_dc sp=$fin_sp | ",
            "det: ref=$det_ref thr=$det_thr dc=$det_dc sp=$det_sp | alias: ref=$alias_ref thr=$alias_thr dc=$alias_dc sp=$alias_sp")
    flush(stdout)

    if do_timing
        println("  -- timing (warmup + $timing_reps reps) at W=$W, D=$D_, L=$L, n_z=$nz, workers=$workers --"); flush(stdout)
        # warmup
        bin_zc_cross_hessian_fill!(ws_ref, Bidx, ZcS)
        bin_zc_cross_hessian_fill_threaded!(ws_thr, Bidx, ZcS; workers = workers_b)
        bin_zc_cross_hessian_fill_drawchunk!(ws_dc, dc, Bidx, ZcS; workers = workers_c)
        bin_zc_cross_hessian_fill_sparse!(ws_sp, sc, ZcS)

        function timeit(f::Function)
            best = Inf
            for _ in 1:timing_reps
                t0 = time_ns()
                f()
                dt = (time_ns() - t0) / 1e9
                best = min(best, dt)
            end
            return best
        end

        t_ref = timeit(() -> bin_zc_cross_hessian_fill!(ws_ref, Bidx, ZcS))
        t_thr = timeit(() -> bin_zc_cross_hessian_fill_threaded!(ws_thr, Bidx, ZcS; workers = workers_b))
        t_dc = timeit(() -> bin_zc_cross_hessian_fill_drawchunk!(ws_dc, dc, Bidx, ZcS; workers = workers_c))
        t_sp = timeit(() -> bin_zc_cross_hessian_fill_sparse!(ws_sp, sc, ZcS))

        a_ref = @allocated bin_zc_cross_hessian_fill!(ws_ref, Bidx, ZcS)
        a_thr = @allocated bin_zc_cross_hessian_fill_threaded!(ws_thr, Bidx, ZcS; workers = workers_b)
        a_dc = @allocated bin_zc_cross_hessian_fill_drawchunk!(ws_dc, dc, Bidx, ZcS; workers = workers_c)
        a_sp = @allocated bin_zc_cross_hessian_fill_sparse!(ws_sp, sc, ZcS)

        @printf("    TIMING  ref=%.4fs  thr=%.4fs (%.2fx)  dc=%.4fs (%.2fx)  sp=%.4fs (%.2fx)\n",
            t_ref, t_thr, t_ref / t_thr, t_dc, t_ref / t_dc, t_sp, t_ref / t_sp)
        @printf("    ALLOC   ref=%.3fMB  thr=%.3fMB  dc=%.3fMB  sp=%.3fMB\n",
            a_ref / 1e6, a_thr / 1e6, a_dc / 1e6, a_sp / 1e6)
        flush(stdout)

        # overwrite the timing-relevant fields on the rows just pushed for this config/s_variant
        for r in ROWS
            if r.config == config_label && r.s_variant == s_variant
                if r.candidate == "origin_owned_serial_REF"
                    r.time_s = t_ref; r.speedup_vs_ref = 1.0; r.allocated_bytes = a_ref
                elseif r.candidate == "origin_owned_threaded"
                    r.time_s = t_thr; r.speedup_vs_ref = t_ref / t_thr; r.allocated_bytes = a_thr
                elseif r.candidate == "draw_chunk_thread_local"
                    r.time_s = t_dc; r.speedup_vs_ref = t_ref / t_dc; r.allocated_bytes = a_dc
                elseif r.candidate == "sparse_spmm"
                    r.time_s = t_sp; r.speedup_vs_ref = t_ref / t_sp; r.allocated_bytes = a_sp
                end
            end
        end
    end
    return nothing
end

# ==================================================================================================
# D=4 configs
# ==================================================================================================
println("=== D=4 configs ==="); flush(stdout)
Random.seed!(19)
ctx_d4 = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)

const D4_CONFIGS = [
    ("d4_K1P1_anchored", 1, 1, :anchored),
    ("d4_K1P1_orthonormal", 1, 1, :orthonormal),
    ("d4_K1P0_orthonormal", 1, 0, :orthonormal),   # K_pair=0
    ("d4_K2P2_orthonormal", 2, 2, :orthonormal),   # larger K
]

for (label, K_mean, K_pair, contrasts) in D4_CONFIGS
    println("--- $label ---"); flush(stdout)
    aug = build_cm_meanzc_augmented_obj(ctx_d4, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = contrasts, meanzc_basis = :direct)
    cctx = build_cm_meanzc_bin_ctx(ctx_d4, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    W = size(ctx_d4.U, 1)
    Random.seed!(1001)
    S1 = synthetic_S(W)
    S2 = synthetic_S(W)   # "solver-derived" substitute: SECOND independent random draw, see file header
    run_config!(label, ctx_d4, aug, cctx, S1, "random_S_1"; workers = 20, do_timing = false)
    run_config!(label, ctx_d4, aug, cctx, S2, "random_S_2_(solver_derived_substitute)"; workers = 20, do_timing = false)
end

# ==================================================================================================
# D=20 real (W=100,000) configs -- production scale
# ==================================================================================================
println(); println("=== real D=20 / W=100,000 configs ==="); flush(stdout)
const W20 = 100_000

const D20_CONFIGS = [
    ("d20_K1P1_exclude_row_anchored", :exclude_row, :anchored, true),      # production config, timed
    ("d20_K1P1_exclude_row_orthonormal", :exclude_row, :orthonormal, true),
    ("d20_K1P1_all_legacy_orthonormal", :all_legacy, :orthonormal, false), # square contrast, correctness only
]

for (label, destination_sample, contrasts, do_timing) in D20_CONFIGS
    println("--- $label (destination_sample=$destination_sample, contrasts=$contrasts) ---"); flush(stdout)
    println("  building ctx/aug/cctx..."); flush(stdout)
    @time ctx20, aug20, cctx20 = build_ctx_aug_cctx(:d20; W = W20, L = 50, K_mean = 1, K_pair = 1,
        contrasts = contrasts, destination_sample = destination_sample)
    Random.seed!(2027)
    S1 = synthetic_S(W20)
    S2 = synthetic_S(W20)
    run_config!(label, ctx20, aug20, cctx20, S1, "random_S_1"; workers = 20, do_timing = do_timing, timing_reps = 5)
    run_config!(label, ctx20, aug20, cctx20, S2, "random_S_2_(solver_derived_substitute)"; workers = 20, do_timing = false)
end

# ==================================================================================================
# Write results CSV + final pass/fail summary
# ==================================================================================================
csv_path = joinpath(D4X, "..", "..", "docs", "HCZ_PREP_CANDIDATE_BENCHMARK_2026-07-29.csv")
csv_path = normpath(csv_path)
write_csv(csv_path, ROWS)
println(); println("Wrote ", csv_path); flush(stdout)

n_fail = count(r -> !r.pass || !r.finite_ok || !r.repeat_deterministic || r.aliases_input, ROWS)
println(); println(n_fail == 0 ? "ALL PASS ($(length(ROWS)) rows)" : "SOME FAILURES ($n_fail / $(length(ROWS)) rows)")
n_fail == 0 || exit(1)

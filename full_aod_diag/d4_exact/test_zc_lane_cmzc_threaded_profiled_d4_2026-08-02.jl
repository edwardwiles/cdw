# Performance closeout task (2026-08-02), Section 8: D4 gate for the newly-ported profiled branch
# in `hessian_cm_structured_v2!` (cm_hessian_threaded.jl). Before this task, that threaded twin had
# NO `profiled_layout` handling at all (confirmed by grep), so every profiled/reduced cctx in this
# codebase had to pass `threaded_bins=false` -- not a performance choice, an outright missing code
# path. The fix ported the serial `hessian_cm_structured!`'s own profiled branch (unchanged math,
# same use_profiled_correction=true calls, same hcz_prep_dispatch! call this task's Section 7 wired
# in) into the threaded twin, swapping only the bin-table prep for the pre-existing
# build_bin_tables_threaded!/prefix_sum_tables_threaded! (already validated for the non-profiled
# case). This test proves that port: threaded_bins=true and threaded_bins=false, on cctxs built
# from the IDENTICAL aug/layout, must produce a bit-identical (or machine-precision) packed Hessian
# at several nonzero states and worker counts.
const D4X = @__DIR__
include(joinpath(D4X, "test_zc_lane_cmzc_d4_fg_and_solve_gate_2026-08-02.jl"))

using Printf

println()
println("=== Section 8 gate: threaded_bins=true vs false on the SAME profiled/reduced CM+ZC cctx ===")

nt = Threads.nthreads()
println("Threads.nthreads()=$nt")

cctx_reduced_threaded = build_cm_meanzc_bin_ctx(ctx, aug_reduced; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = true, inner_fg_backend = :dense_reference,
    profiled_layout = layout)
@printf("cctx_reduced_threaded: use_threaded_bins=%s  tls===nothing? %s\n",
    cctx_reduced_threaded.use_threaded_bins, cctx_reduced_threaded.tls === nothing)
# archC_meanzc_base_state publishes cctx.nu_ref[] = collect(νvec) before ever calling the Hessian
# callback (cm_meanzc_production.jl:224) -- this test calls hessian_cm_structured!/_v2! directly, so
# it must do the same for cctx_reduced_threaded (a freshly built cctx, never routed through that
# base-state helper).
cctx_reduced_threaded.nu_ref[] = collect(νvec0)

n = obj_reduced.outer_constr_index
Random.seed!(9200)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n), -0.03 .* randn(n)]

h_len = (cctx_reduced.NCORE + cctx_reduced.ncm) * (cctx_reduced.NCORE + cctx_reduced.ncm + 1) ÷ 2
maxdiff_overall = 0.0
for (pi_, x) in enumerate(xs)
    # Independent prep per cctx: each owns its own core_cf_ref/profiled_full_wctx/etc, so both must
    # be freshly prepped at the SAME x for a fair comparison, mirroring _prep_dual_index_for_archC!'s
    # own precondition (documented at that function's call sites throughout this file).
    _prep_dual_index_for_archC!(cctx_reduced, obj_reduced, x)
    h_serial = zeros(h_len)
    hessian_cm_structured!(h_serial, obj_reduced, cctx_reduced)

    _prep_dual_index_for_archC!(cctx_reduced_threaded, obj_reduced, x)
    h_threaded = zeros(h_len)
    hessian_cm_structured_v2!(h_threaded, obj_reduced, cctx_reduced_threaded; threaded_bins = true, tls = cctx_reduced_threaded.tls)

    maxdiff = maximum(abs.(h_serial .- h_threaded))
    global maxdiff_overall = max(maxdiff_overall, maxdiff)
    check("pt$pi_: threaded_bins=true profiled Hessian matches threaded_bins=false profiled Hessian (max|Δ|=$(maxdiff))",
          maxdiff < 1e-10)
    @printf("  pt%d: max|Δ(threaded_bins=true - false)|=%.3e\n", pi_, maxdiff)
end

println()
println("Overall max|Δ(threaded-serial, profiled CM+ZC)|: ", maxdiff_overall)
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)

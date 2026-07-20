# Continuation (branch diag/fullA-d4-exact-cm-hessian-arch), step 2: full
# KNITRO inner-solve benchmark for Hessian Architectures A/B/C/D, D=4,
# L in {10,20,50}, calibration point. Reports: Hessian-callback wall time
# (from @prof), cold + warm full inner-solve wall time, inner iteration
# count, FG/Hessian callback counts, and total bytes allocated (@allocated
# around the whole solve). See c13_validate_hessian_archs.jl for the
# preceding pure-numerical agreement check (all four architectures agree with
# Architecture A to ~1e-15, after fixing a real H_EC symmetrization bug in
# Architecture C -- see cm_hessian_architectures.jl's inline comment).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
using Printf, Statistics, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)

Random.seed!(9091)
x_free_perturbed = copy(x_free_calib)
x_free_perturbed[2:end] .*= exp.(0.01 .* randn(length(x_free_perturbed) - 1))
θ_full_perturbed = CS.reconstruct_full(x_free_perturbed, ctx.m)

const HVP_OPT = joinpath(D4X_ROOT, "full_aod_diag", "d4_exact", "ek_inner_hvp.opt")

function run_solve(label, obj, θ; hess_cb_builder = nothing, hvp = false)
    prof_reset!()
    iters0 = CS.INNER_ITERS_TOTAL[]
    bytes = @allocated begin
        t0 = time()
        res = inner_loop_internal_archgeneric(obj, θ; hess_cb_builder = hess_cb_builder, hvp = hvp)
        t1 = time()
    end
    iters = CS.INNER_ITERS_TOTAL[] - iters0
    K_hard, x, nStatus, n_fg, n_hess = res
    hess_label = hvp ? "inner_dual_hvp_callback_dense" :
                 (hess_cb_builder === archA_hess_cb_builder ? "inner_dual_hessian_callback" : "inner_dual_hessian_callback_archC")
    hess_times = get(PROF_TIMES, hess_label, Float64[])
    hess_total = sum(hess_times)
    hess_mean = isempty(hess_times) ? NaN : mean(hess_times)
    solve_times = get(PROF_TIMES, "inner_knitro_dual_solve_arch", Float64[])
    return (label = label, nStatus = nStatus, wall_s = t1 - t0, iters = iters,
            n_fg = n_fg, n_hess = n_hess, hess_total_s = hess_total, hess_mean_s = hess_mean,
            knitro_solve_s = isempty(solve_times) ? NaN : sum(solve_times),
            bytes_alloc = bytes, x = x)
end

function bench_L(L::Int; contrasts::Symbol = :anchored, n_warm_reps::Int = 3)
    println("\n"); println("="^100); println("L=$L  contrasts=$contrasts  (D=4, W=$(size(ctx.U,1)))")
    println("="^100)

    # ---- build one obj per architecture (independent instances, no shared warm-state) ----
    augA = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    augB = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts)
    augC = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)   # own instance, same math as A's obj
    augD = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    augD.obj_cm.inner_loop_opt = HVP_OPT

    cctx = build_cm_bin_ctx(ctx, augC)
    cctx_callback = archC_hess_cb_builder(cctx)   # the actual 5-arg KNITRO callback
    hessC_builder = (obj_arg) -> cctx_callback    # hess_cb_builder(obj) -> callback, matching archA's convention

    variants = [
        ("A_dense_BLAS", augA.obj_cm, archA_hess_cb_builder, false),
        ("B_chunked_moments+BLAS", augB.obj_cm, archA_hess_cb_builder, false),
        ("C_structured_bintables", augC.obj_cm, hessC_builder, false),
        ("D_matrixfree_HVP", augD.obj_cm, nothing, true),
    ]

    rows = NamedTuple[]
    for (label, obj, hcb, hvp) in variants
        # median of 3 independent cold solves (each preceded by an obj.x reset) --
        # single-shot cold timing showed real run-to-run noise (GC pauses, first-touch
        # page faults) large enough to distort the cross-architecture comparison at
        # this problem size; median-of-3 is cheap here (D=4 solves are fast) and
        # removes the worst of it.
        cold_reps = map(1:3) do _
            obj.x .= NaN
            run_solve(label * "_cold", obj, θ_full; hess_cb_builder = hcb, hvp = hvp)
        end
        # last rep already leaves obj.x at the calib optimum for the warm tests below;
        # report median wall/hess/alloc across the 3 reps, keep the last rep's x/nStatus/iters
        cold = merge(cold_reps[end], (wall_s = median([r.wall_s for r in cold_reps]),
                                       hess_total_s = median([r.hess_total_s for r in cold_reps]),
                                       bytes_alloc = median([r.bytes_alloc for r in cold_reps])))
        # "same-point" warm: re-solve at the IDENTICAL theta -- obj.x is already at the
        # optimum, so KNITRO takes 0 Newton iterations; this measures pure KNITRO
        # context setup/teardown overhead, NOT the Hessian architecture (see report).
        warms = [run_solve(label * "_warm_samepoint", obj, θ_full; hess_cb_builder = hcb, hvp = hvp) for _ in 1:n_warm_reps]
        warm_wall = median([w.wall_s for w in warms])
        warm_hess_total = median([w.hess_total_s for w in warms])
        warm_iters = warms[1].iters
        # "realistic" warm: re-solve at a NEARBY perturbed theta, warm-started from the
        # calibration solution -- exercises a genuine (small) number of Newton
        # iterations, representative of an outer-loop line search / nearby-theta re-solve.
        wp = [run_solve(label * "_warm_perturbed", obj, θ_full_perturbed; hess_cb_builder = hcb, hvp = hvp) for _ in 1:n_warm_reps]
        obj.x .= NaN   # reset so the NEXT rep starts from the true calib optimum again, not the perturbed one
        cold2 = run_solve(label * "_recold_calib", obj, θ_full; hess_cb_builder = hcb, hvp = hvp)  # restore obj.x to calib optimum for future reps
        wp_wall = median([w.wall_s for w in wp])
        wp_hess_total = median([w.hess_total_s for w in wp])
        wp_iters = wp[1].iters
        @printf("%-24s nStatus=%-5d  cold: wall=%.4fs iters=%3d n_hess=%3d hess_total=%.4fs (mean/call=%.2e s)  alloc=%.1fMB\n",
                label, cold.nStatus, cold.wall_s, cold.iters, cold.n_hess, cold.hess_total_s, cold.hess_mean_s, cold.bytes_alloc/1e6)
        @printf("%-24s %-13s warm(samept): wall=%.4fs(med of %d) iters=%3d hess_total=%.4fs(med)  alloc=%.1fMB(med)\n",
                "", "", warm_wall, n_warm_reps, warm_iters, warm_hess_total, median([w.bytes_alloc for w in warms])/1e6)
        @printf("%-24s %-13s warm(perturbed): wall=%.4fs(med of %d) iters=%3d hess_total=%.4fs(med) n_hess=%3d\n",
                "", "", wp_wall, n_warm_reps, wp_iters, wp_hess_total, wp[1].n_hess)
        push!(rows, (L = L, contrasts = contrasts, arch = label,
                      cold_wall_s = cold.wall_s, cold_iters = cold.iters, cold_nStatus = cold.nStatus,
                      cold_hess_total_s = cold.hess_total_s, cold_hess_mean_s = cold.hess_mean_s,
                      cold_n_hess = cold.n_hess, cold_n_fg = cold.n_fg, cold_alloc_bytes = cold.bytes_alloc,
                      warm_wall_s = warm_wall, warm_iters = warm_iters, warm_hess_total_s = warm_hess_total,
                      warm_alloc_bytes = median([w.bytes_alloc for w in warms]),
                      wp_wall_s = wp_wall, wp_iters = wp_iters, wp_hess_total_s = wp_hess_total, wp_n_hess = wp[1].n_hess,
                      x_cold = collect(cold.x)))
    end

    # ---- cross-architecture x-agreement check (all should converge to the SAME dual point) ----
    xref = rows[1].x_cold
    for r in rows[2:end]
        if length(r.x_cold) == length(xref)
            d = maximum(abs.(r.x_cold .- xref))
            @printf("   x agreement vs A: %-24s max|Δx| = %.3e\n", r.arch, d)
        end
    end

    return rows
end

# ---- JIT warmup pass (discarded): first call of each closure type pays Julia
# compilation cost, which would otherwise contaminate the FIRST architecture's
# "cold" timing (observed directly: an uncontrolled first run showed A_dense_BLAS
# at L=10 taking 3.97s cold vs 0.03s at L=20 -- almost entirely JIT, not a real
# architecture cost; see docs/fullA_cm_hessian_architecture_report.md sec 4). ----
println("\n--- JIT warmup pass (results discarded) ---")
bench_L(5; contrasts = :anchored)
println("--- warmup done, starting real measurements ---\n")

all_rows = NamedTuple[]
for L in (10, 20, 50)
    append!(all_rows, bench_L(L; contrasts = :anchored))
end
# One orthonormal-contrasts spot check at the largest L (R-transform congruence adds
# extra small nO x nO matmuls to Architecture C's H_EC/H_CC assembly that Architecture
# A never pays -- confirms the timing story is not an artifact of the anchored-only R=I case).
append!(all_rows, bench_L(50; contrasts = :orthonormal))

println("\n\n" * "="^100)
println("SUMMARY TABLE (anchored contrasts, calibration point)")
println("="^100)
@printf("%-6s %-24s %8s %6s %10s %10s %10s %10s %10s %6s\n", "L", "arch", "cold_s", "iters", "hess_tot_s", "warm_s", "warm_hess", "alloc_MB", "wpert_s", "wp_it")
for r in all_rows
    @printf("%-6d %-24s %8.4f %6d %10.4f %10.4f %10.4f %10.1f %10.4f %6d\n",
            r.L, r.arch, r.cold_wall_s, r.cold_iters, r.cold_hess_total_s, r.warm_wall_s, r.warm_hess_total_s, r.cold_alloc_bytes/1e6, r.wp_wall_s, r.wp_iters)
end

using DelimitedFiles
open(joinpath(@__DIR__, "..", "..", "docs", "fullA_cm_hessian_bench_raw.csv"), "w") do io
    println(io, "L,contrasts,arch,cold_wall_s,cold_iters,cold_nStatus,cold_hess_total_s,cold_hess_mean_s,cold_n_hess,cold_n_fg,cold_alloc_bytes,warm_wall_s,warm_iters,warm_hess_total_s,warm_alloc_bytes,wp_wall_s,wp_iters,wp_hess_total_s,wp_n_hess")
    for r in all_rows
        println(io, join([r.L, r.contrasts, r.arch, r.cold_wall_s, r.cold_iters, r.cold_nStatus, r.cold_hess_total_s, r.cold_hess_mean_s, r.cold_n_hess, r.cold_n_fg, r.cold_alloc_bytes, r.warm_wall_s, r.warm_iters, r.warm_hess_total_s, r.warm_alloc_bytes, r.wp_wall_s, r.wp_iters, r.wp_hess_total_s, r.wp_n_hess], ","))
    end
end
println("\nRaw CSV written to docs/fullA_cm_hessian_bench_raw.csv")

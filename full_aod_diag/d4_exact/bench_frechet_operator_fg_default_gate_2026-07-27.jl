# Winner-aware H_ER phase (2026-07-27), Section 3.3: common-Frechet operator-FG DEFAULT gate.
#
# docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md found and fixed the skip_cm_fill_ref bug that
# was causing :cm_frechet_lookup to crash at a perturbed D=20 point, then measured ONE D=20 point
# (the previously-crashing calibration point, L=50, contrasts=:anchored only) and explicitly declined
# to flip CM_FRECHET_INNER_FG_BACKEND_DEFAULT, citing insufficient breadth ("this session's own
# gates cover exactly two D=20 points ... not the breadth of configurations a genuine default-flip
# decision should rest on"). This script is the broader gate that doc asked for: both contrasts,
# THREE points (calibration, a small near-delta=1 random perturbation, and the "hard point"
# x_free0.*1.01 that originally exposed the crash), complete-solve time/allocation (repeated,
# median-of-N), n_fg/n_hess counts (iteration-count proxy), correctness (zeta*/status agreement),
# and isolated per-FG-callback allocation for the lookup state (same methodology
# docs/COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md used to find the historical 14MB/call
# regression -- reused here to confirm whether it is still present post-fix).
#
# NOTE on cm_cross_hessian_backend: both sides of this A/B use whatever
# CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[] is AT THE TIME THIS SCRIPT RUNS (i.e. the real
# production default after this session's own Section 3 Hessian-backend decision, run and committed
# separately) -- this isolates the FG-backend question from the Hessian cross-block question, but
# measures it against the actual production configuration rather than an artificially-fixed one.
#
# Usage: julia --project=. -t N full_aod_diag/d4_exact/bench_frechet_operator_fg_default_gate_2026-07-27.jl
const D4X = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_cplus.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Printf, Statistics, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

function timed_median(f, nreps)
    f()   # warm-up
    times = Float64[]; allocs = Int[]
    for _ in 1:nreps
        s = @timed f()
        push!(times, s.time); push!(allocs, s.bytes)
    end
    k = cld(nreps, 2)
    return (time = sort(times)[k], bytes = sort(allocs)[k])
end

println("Building real D=20 context (W=80000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(2026)
L = 50
NREPS = 5

println("cm_cross_hessian_backend in use for this gate: ", CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[]); flush(stdout)

results = NamedTuple[]

for contrasts in (:anchored, :orthonormal)
    println("="^90); println("contrasts=$contrasts L=$L"); println("="^90); flush(stdout)
    pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, inner_fg_backend = :dense_reference,
        cm_cross_hessian_backend = CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[])
    pcx_lookup = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
        cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup,
        cm_cross_hessian_backend = CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[])

    # near-delta1 perturbation: small random nudge of every free coordinate (matches the Hessian
    # gate's own "near_delta1_perturbed" point construction philosophy).
    x_free_near = x_free_calib .+ vcat(0.005, 0.01 .* randn(length(x_free_calib) - 1))
    # hard point: the exact x_free0 .* 1.01 perturbation that exposed the skip_cm_fill_ref crash
    # (docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md).
    x_free_hard = x_free_calib .* 1.01

    for (label, x_free) in (("calib", x_free_calib), ("near_delta1_perturbed", x_free_near), ("hard_point_x1.01", x_free_hard))
        println("-- $label --"); flush(stdout)
        θ_full0 = CS.reconstruct_full(x_free, pcx_dense.ctx_cm.m)

        print("  dense complete-solve:  ")
        sd = timed_median(() -> inner_loop_internal_archgeneric(pcx_dense.ctx_cm.obj, θ_full0; hess_cb_builder = pcx_dense.hess_cb_builder), NREPS)
        @printf "median=%.4fs alloc=%.3fMB\n" sd.time sd.bytes/1e6

        print("  lookup complete-solve: ")
        sl = timed_median(() -> inner_loop_internal_cmfrechetlookup_production(pcx_lookup.ctx_cm.obj, θ_full0, pcx_lookup.cctx, pcx_lookup.aug.level_targets; hess_cb_builder = pcx_lookup.hess_cb_builder), NREPS)
        @printf "median=%.4fs alloc=%.3fMB\n" sl.time sl.bytes/1e6

        Kd, xd, nsd, nfgd, nhd = inner_loop_internal_archgeneric(pcx_dense.ctx_cm.obj, θ_full0; hess_cb_builder = pcx_dense.hess_cb_builder)
        Kl, xl, nsl, nfgl, nhl = inner_loop_internal_cmfrechetlookup_production(pcx_lookup.ctx_cm.obj, θ_full0, pcx_lookup.cctx, pcx_lookup.aug.level_targets; hess_cb_builder = pcx_lookup.hess_cb_builder)

        feasd = nsd in (0, -100, -101, -102, -103, -400, -401, -402)
        feasl = nsl in (0, -100, -101, -102, -103, -400, -401, -402)
        check("contrasts=$contrasts $label: dense feasible (status=$nsd)", feasd)
        check("contrasts=$contrasts $label: lookup feasible (status=$nsl)", feasl)
        if feasd && feasl
            zdiff = abs(xd[1] - xl[1])
            check("contrasts=$contrasts $label: status matches ($nsd vs $nsl)", nsd == nsl)
            check("contrasts=$contrasts $label: zeta* agrees (|Δ|=$zdiff)", zdiff < 1e-6)
        end
        speedup = sd.time / sl.time
        alloc_ratio = sl.bytes / max(sd.bytes, 1)
        @printf "  n_fg: dense=%d lookup=%d   n_hess: dense=%d lookup=%d   speedup=%.3fx   alloc_ratio(lookup/dense)=%.4f\n" nfgd nfgl nhd nhl speedup alloc_ratio
        push!(results, (contrasts=contrasts, label=label, t_dense=sd.time, t_lookup=sl.time,
            b_dense=sd.bytes, b_lookup=sl.bytes, nfg_dense=nfgd, nfg_lookup=nfgl,
            nhess_dense=nhd, nhess_lookup=nhl, speedup=speedup, alloc_ratio=alloc_ratio,
            status_dense=nsd, status_lookup=nsl))
        flush(stdout)
    end

    # isolated per-FG-callback allocation, lookup state (same methodology as
    # docs/COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md, re-run post-fix)
    st = pcx_lookup.cctx.cmlookup_st
    if st !== nothing
        g = Vector{Float64}(undef, CS.inner_loop_number_variables(pcx_lookup.ctx_cm.obj))
        x0 = CS.inner_loop_initial_values(pcx_lookup.ctx_cm.obj)
        st(x0, g)   # warm-up
        per_call_bytes = @allocated st(x0, g)
        @printf "  isolated per-FG-callback allocation (CMFrechetLookupState, warm): %d bytes\n" per_call_bytes
    end
    flush(stdout)
end

println()
println("="^90)
println("SUMMARY")
println("="^90)
for r in results
    @printf "%-12s %-22s speedup=%.3fx  alloc_ratio=%.4f  n_fg(d/l)=%d/%d  n_hess(d/l)=%d/%d  status(d/l)=%d/%d\n" String(r.contrasts) r.label r.speedup r.alloc_ratio r.nfg_dense r.nfg_lookup r.nhess_dense r.nhess_lookup r.status_dense r.status_lookup
end

println(ALL_PASS[] ? "ALL PASS (correctness/feasibility checks)" : "SOME FAILURES (correctness/feasibility checks)")
ALL_PASS[] || exit(1)

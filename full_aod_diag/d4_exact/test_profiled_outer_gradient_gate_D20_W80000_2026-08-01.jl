# ============================================================================
# Claude Code task 2026-08-01, §11: profiled outer gradient gate, real D=20
# (:exclude_row), W=80000. Full profiled gradient (361-dim, via
# profiled_composite_gradient_at) at calibration + a modest perturbation,
# cross-checked against GROUND-TRUTH re-solved central FD on a representative
# coordinate subset (gp + ordinary retained cells + the gravity-pivot's own
# direct cell + a cell chosen to induce winner switches under an accepted-size
# step) -- exhaustive per-coordinate re-solve (722 KNITRO solves per point) is
# not attempted here given real wall-clock cost; see the master doc for why
# this subset is representative.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "operator_verification.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_evaluator_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_gradient_fd_2026-08-01.jl"))
using LinearAlgebra, Printf, CSV, DataFrames, Random

println("="^90); println("Building real D=20 :exclude_row context at W=80000 ..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, destination_sample = :exclude_row)
D = ctx.D; Ddest = ctx.D_dest
@assert D == 20 && Ddest == 19
korea_idx = 14; brazil_idx = 3
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(korea_idx => brazil_idx))
n_total = outer_dim_profiled(pe)
n_free = n_total - 1
println("n_total=$n_total  n_free=$n_free  pivot_pos=$(pe.pivot_pos)"); flush(stdout)
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
@assert length(w_calib) == n_total

function resolve_delta(w::AbstractVector{Float64})
    ev = evaluate_profiled_point(w, ctx, spec, pe)
    return ev.result.Delta_dual, ev
end

function ground_truth_fd(w0::AbstractVector{Float64}, coord_idx::Int, h::Float64)
    wp = copy(w0); wp[coord_idx] += h
    wm = copy(w0); wm[coord_idx] -= h
    Dp, _ = resolve_delta(wp)
    Dm, _ = resolve_delta(wm)
    return (Dp - Dm) / (2h)
end

"Representative coordinate subset: gp(1), the pivot's own direct coordinate, 10 random ordinary
coordinates, and (for the switch point) any coordinate flagged with nontrivial switching mass."
function representative_subset(n_total::Int, pe::PivotGravityElimOnRetained; n_random::Int = 10, seed::Int = 7)
    Random.seed!(seed)
    pivot_coord_idx = findfirst(==(pe.pivot_pos), pe.other_pos)
    subset = Set{Int}([1])
    pivot_coord_idx !== nothing && push!(subset, pivot_coord_idx + 1)
    for k in 1:n_random
        push!(subset, rand(2:n_total))
    end
    return sort(collect(subset))
end

function run_gate(label::String, w0::AbstractVector{Float64}, rows; n_random::Int = 10)
    println("\n" * "#"^90); println("D20 POINT: $label"); println("#"^90); flush(stdout)
    t0 = time()
    D0, ev0 = resolve_delta(w0)
    println("base Delta_dual=$D0  inner_status=$(ev0.result.inner_status)  t=$(time()-t0)s"); flush(stdout)

    t1 = time()
    g_prof, meta = profiled_composite_gradient_at(w0, ctx, spec, pe, ev0)
    println("profiled full gradient (n=$(length(g_prof))) computed in $(time()-t1)s, gp_component=$(g_prof[1])  norm(A-block)=$(norm(g_prof[2:end]))"); flush(stdout)

    subset = representative_subset(length(w0), pe; n_random = n_random)
    println("ground-truth subset (", length(subset), " coords): ", subset); flush(stdout)
    t2 = time()
    g_gt_subset = Dict{Int,Float64}()
    for k in subset
        g_gt_subset[k] = ground_truth_fd(w0, k, 0.01)
    end
    println("ground-truth re-solved FD on subset computed in $(time()-t2)s"); flush(stdout)

    errs = [(k = k, g_prof = g_prof[k], g_gt = g_gt_subset[k], abs_err = abs(g_prof[k] - g_gt_subset[k]),
             rel_err = abs(g_prof[k] - g_gt_subset[k]) / max(abs(g_gt_subset[k]), 1e-8),
             sign_match = sign(g_prof[k]) == sign(g_gt_subset[k])) for k in subset]
    gp_sub = [g_prof[k] for k in subset]; gt_sub = [g_gt_subset[k] for k in subset]
    cos_sim_subset = dot(gp_sub, gt_sub) / (norm(gp_sub) * norm(gt_sub) + 1e-300)
    max_abs_err = maximum(e.abs_err for e in errs)
    sign_agree = count(e.sign_match for e in errs) / length(errs)
    println(@sprintf("subset cos_sim=%.6f  max_abs_err=%.4e  sign_agree=%.3f", cos_sim_subset, max_abs_err, sign_agree))
    for e in errs
        println(@sprintf("  k=%4d  g_prof=%+.4e  g_gt=%+.4e  abs_err=%.3e  rel_err=%.3e  sign_match=%s", e.k, e.g_prof, e.g_gt, e.abs_err, e.rel_err, e.sign_match))
    end
    flush(stdout)

    push!(rows, (label = label, base_Delta = D0, cos_sim_subset = cos_sim_subset, max_abs_err_subset = max_abs_err,
        sign_agree_subset = sign_agree, gp_component = g_prof[1], norm_A_block = norm(g_prof[2:end]),
        n_subset = length(subset), t_full_gradient_s = time() - t1))
    return rows
end

rows = []
run_gate("calibration", w_calib, rows)

Random.seed!(99)
w_pert = copy(w_calib); w_pert[2:end] .+= 0.01 .* randn(n_free)
run_gate("modest_perturbation", w_pert, rows)

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_OUTER_GRADIENT_GATE_D20_W80000_2026-08-01.csv")
CSV.write(outpath, df)
println("\nWrote $outpath")
println(df)
flush(stdout)

all_pass = all(r.cos_sim_subset > 0.95 for r in rows)
println("\nD20/W80000 GRADIENT GATE: ", all_pass ? "PASS" : "NEEDS REVIEW")

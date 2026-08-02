# ============================================================================
# Claude Code task 2026-08-01, §11: profiled outer gradient gate, D=4.
# Compares profiled_composite_gradient_at (fixed-dual central FD + O(1)
# incremental winner update, the SAME mechanism production's C+ gradient
# uses) against GROUND-TRUTH central finite differences of the actual
# RE-SOLVED Delta_dual (a fresh inner_loop_KNITRO_profiled solve at each
# perturbed point) -- the "slow ground truth" this whole FD-secant mechanism
# is designed to approximate cheaply, per this repo's own established
# discipline (c8_nestedw_gradcheck.jl's role for the production gradient).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "operator_verification.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
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
using LinearAlgebra, Printf, CSV, DataFrames

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
D = ctx.D
println("D=$D  Aod_offset=$(ctx.Aod_offset)"); flush(stdout)

spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
n_free = length(pe.other_pos)
n_total = outer_dim_profiled(pe)
println("n_free=$n_free  n_total=$n_total"); flush(stdout)
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
@assert length(w_calib) == n_total

"Re-solve ground truth: rebuild the profiled operator bundle + fresh inner KNITRO solve at w, return Delta_dual."
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

function run_gate(label::String, w0::AbstractVector{Float64}, h_gt::Float64, rows)
    println("\n" * "#"^90); println("POINT: $label"); println("#"^90); flush(stdout)
    D0, ev0 = resolve_delta(w0)
    println("base Delta_dual=$D0  inner_status=$(ev0.result.inner_status)  verified=$(ev0.result.primal_dual_gap<1e-3)"); flush(stdout)

    g_prof, meta = profiled_composite_gradient_at(w0, ctx, spec, pe, ev0)
    println("profiled gradient computed, gp_component=$(g_prof[1])  norm(A-block)=$(norm(g_prof[2:end]))"); flush(stdout)

    g_gt = zeros(n_total)
    for k in 1:n_total
        g_gt[k] = ground_truth_fd(w0, k, h_gt)
    end
    println("ground-truth FD gradient computed (re-solved), norm=$(norm(g_gt))"); flush(stdout)

    diff = g_prof .- g_gt
    max_abs_err = maximum(abs.(diff))
    max_rel_err = maximum(abs.(diff) ./ max.(abs.(g_gt), 1e-8))
    cos_sim = dot(g_prof, g_gt) / (norm(g_prof) * norm(g_gt) + 1e-300)
    sign_agree = count(sign.(g_prof) .== sign.(g_gt)) / n_total

    # predicted vs realized objective change for an accepted-sized step (a random unit direction, step=0.01)
    dirvec = randn(n_total); dirvec ./= norm(dirvec)
    step = 0.01
    predicted_change = dot(g_prof, dirvec) * step
    D_step, _ = resolve_delta(w0 .+ step .* dirvec)
    realized_change = D_step - D0

    println(@sprintf("max_abs_err=%.4e  max_rel_err=%.4e  cos_sim=%.6f  sign_agree=%.3f", max_abs_err, max_rel_err, cos_sim, sign_agree))
    println(@sprintf("predicted_change=%.4e  realized_change=%.4e", predicted_change, realized_change))
    flush(stdout)

    push!(rows, (label = label, base_Delta = D0, max_abs_err = max_abs_err, max_rel_err = max_rel_err,
        cos_sim = cos_sim, sign_agree = sign_agree, predicted_change = predicted_change, realized_change = realized_change,
        gp_component = g_prof[1], norm_A_block_profiled = norm(g_prof[2:end]), norm_A_block_gt = norm(g_gt[2:end])))

    # per-coordinate detail (gp, ordinary, and worst-agreement coordinates)
    return g_prof, g_gt, rows
end

rows = []
run_gate("calibration", w_calib, 0.01, rows)

Random_seed = 42
using Random
Random.seed!(Random_seed)
w_pert_small = copy(w_calib); w_pert_small[2:end] .+= 0.01 .* randn(n_free)
run_gate("small_perturbation", w_pert_small, 0.01, rows)

w_pert_switch = copy(w_calib); w_pert_switch[2:end] .+= 0.5 .* randn(n_free)
run_gate("many_winner_changes", w_pert_switch, 0.01, rows)

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_OUTER_GRADIENT_GATE_D4_2026-08-01.csv")
CSV.write(outpath, df)
println("\nWrote $outpath")
println(df)
flush(stdout)

all_pass = all(r.cos_sim > 0.99 for r in rows)
println("\nD4 GRADIENT GATE: ", all_pass ? "PASS" : "NEEDS REVIEW")

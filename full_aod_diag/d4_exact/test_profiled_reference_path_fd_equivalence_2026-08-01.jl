# ============================================================================
# Claude Code task 2026-08-01, §12: full-vs-profiled value-function
# equivalence along a REDUCED (profiled) finite-difference path.
#
# At each of {base, base+h*e_k, base-h*e_k} (a few representative k):
#   (i)   solve the profiled inner problem;
#   (ii)  recover the gamma-normalized full A using THAT solve's own verified
#         LFD (recover_gamma_normalized_full_A_from_lfd, already gated
#         2026-08-01 session);
#   (iii) solve the LEGACY FULL inner problem AT THE RECOVERED A;
#   (iv)  compare (Delta_dual, verified LFD) between (i) and (iii).
# This is the SAME recover-then-resolve procedure test_unrestricted_knitro_
# d20_omitrow_smallw_2026-08-01.jl already validated at calibration -- this
# file applies it to FD ENDPOINTS (task's own explicit requirement: "repeat
# this recovery separately at each endpoint"), not just the single base point.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
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
using LinearAlgebra, Printf, CSV, DataFrames

const IS_D20 = "D20" in ARGS || get(ENV, "PROFILED_FD_EQUIV_D20", "0") == "1"

if IS_D20
    include(joinpath(@__DIR__, "context_real_d20.jl"))
    println("Building real D=20 :exclude_row context at W=80000 ..."); flush(stdout)
    ctx = d20_real_setup(W = 80_000, destination_sample = :exclude_row)
    spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(14 => 3))
    tag = "D20_W80000"
    h = 0.01
    probe_coords = [1, 2, pe.pivot_pos + 1, 50, 150]  # gp, first free coord, pivot's own coord, two others
else
    ctx0 = d4_exact_setup()
    ctx = build_unrestricted_operator_ctx(ctx0)
    spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
    tag = "D4"
    h = 0.05
    probe_coords = collect(1:outer_dim_profiled(pe))
end
D = ctx.D
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
n_total = outer_dim_profiled(pe)
println("tag=$tag  n_total=$n_total  probe_coords=$probe_coords"); flush(stdout)

"recover-then-resolve comparison at a single profiled point w -- mirrors test_unrestricted_knitro_d20_omitrow_smallw_2026-08-01.jl's run_point exactly."
const Ddest_ = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

function recover_then_resolve(w::Vector{Float64})
    ev = evaluate_profiled_point(w, ctx, spec, pe)
    p_ok = ev.result.inner_status in (0, -100, -101, -103)
    z_recovered, c_recover, gamma_tilde = recover_gamma_normalized_full_A_from_lfd(ev.theta_full, ctx, ev.st.cf, ev.m_weights)
    θ_full_recovered = copy(ev.theta_full)
    θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest_] .= vec(exp.(z_recovered))

    ctx.obj.use_cached_x = false; ctx.obj.x .= NaN
    K_ref, x_ref, nStatus_ref, nfg_ref, nhess_ref, st_ref = inner_loop_internal_compressed(ctx.obj, θ_full_recovered, ctx)
    ref_ok = nStatus_ref in (0, -100, -101, -103)
    zeta_ref = x_ref[1]; lambda_ref = x_ref[2:end]
    ov_ref = verify_inner_solution_operator_unrestricted!(zeta_ref, lambda_ref, st_ref.cf, ctx.obj, st_ref.cf.W)
    mw_ref, verify_ref = verify_namedtuple_from_operator(ov_ref, ctx.obj, st_ref.cf.W, nStatus_ref)

    mw_diff = maximum(abs.(mw_ref .- ev.m_weights))
    div_diff = abs(verify_ref.Delta_primal - ev.result.Delta_primal)
    winner_match = sum(Float64.(st_ref.cf.winner)) == ev.result.winner_checksum
    return (p_ok = p_ok, ref_ok = ref_ok, Delta_dual_profiled = ev.result.Delta_dual, Delta_dual_ref = verify_ref.Delta_dual,
            Delta_primal_profiled = ev.result.Delta_primal, Delta_primal_ref = verify_ref.Delta_primal,
            mw_diff = mw_diff, div_diff = div_diff, winner_match = winner_match)
end

rows = []
println("\n" * "="^90); println("BASE POINT"); println("="^90); flush(stdout)
r_base = recover_then_resolve(w_calib)
println(@sprintf("base: p_ok=%s ref_ok=%s winner_match=%s Delta_dual(prof)=%.6e Delta_dual(ref@recovered)=%.6e div_diff=%.3e mw_diff=%.3e",
    r_base.p_ok, r_base.ref_ok, r_base.winner_match, r_base.Delta_dual_profiled, r_base.Delta_dual_ref, r_base.div_diff, r_base.mw_diff))
push!(rows, merge((point = "base", coord = 0, direction = "na"), r_base))
flush(stdout)

for k in probe_coords
    for (dirn, sgn) in (("plus", +1.0), ("minus", -1.0))
        wpt = copy(w_calib); wpt[k] += sgn * h
        r = recover_then_resolve(wpt)
        println(@sprintf("k=%d %s: p_ok=%s ref_ok=%s winner_match=%s Delta_dual(prof)=%.6e Delta_dual(ref@recovered)=%.6e div_diff=%.3e mw_diff=%.3e",
            k, dirn, r.p_ok, r.ref_ok, r.winner_match, r.Delta_dual_profiled, r.Delta_dual_ref, r.div_diff, r.mw_diff))
        push!(rows, merge((point = "fd_endpoint", coord = k, direction = dirn), r))
        flush(stdout)
    end
end

# derivative comparison: profiled FD derivative of Delta_dual_profiled vs reference-path FD derivative
# of Delta_dual_ref (the "recovered reference path"), per each probed coordinate.
deriv_rows = []
for k in probe_coords
    rp = only(filter(r -> r.coord == k && r.direction == "plus", rows))
    rm = only(filter(r -> r.coord == k && r.direction == "minus", rows))
    d_profiled = (rp.Delta_dual_profiled - rm.Delta_dual_profiled) / (2h)
    d_reference_path = (rp.Delta_dual_ref - rm.Delta_dual_ref) / (2h)
    push!(deriv_rows, (coord = k, fd_derivative_profiled = d_profiled, fd_derivative_reference_path = d_reference_path,
        abs_diff = abs(d_profiled - d_reference_path), rel_diff = abs(d_profiled - d_reference_path) / max(abs(d_reference_path), 1e-8)))
    println(@sprintf("DERIVATIVE k=%d: profiled_FD=%.6e  reference_path_FD=%.6e  abs_diff=%.3e  rel_diff=%.3e",
        k, d_profiled, d_reference_path, abs(d_profiled - d_reference_path), abs(d_profiled - d_reference_path) / max(abs(d_reference_path), 1e-8)))
end
flush(stdout)

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_REFERENCE_PATH_FD_EQUIVALENCE_2026-08-01$(IS_D20 ? "_D20" : "_D4").csv")
CSV.write(outpath, df)
println("\nWrote $outpath")

df2 = DataFrame(deriv_rows)
outpath2 = joinpath(@__DIR__, "..", "..", "PROFILED_REFERENCE_PATH_FD_DERIVATIVE_COMPARISON_2026-08-01$(IS_D20 ? "_D20" : "_D4").csv")
CSV.write(outpath2, df2)
println("Wrote $outpath2")

all_pass = all(r.p_ok && r.ref_ok && r.winner_match && r.div_diff < 1e-3 for r in rows)
println("\nREFERENCE-PATH FD EQUIVALENCE ($tag): ", all_pass ? "PASS" : "NEEDS REVIEW")

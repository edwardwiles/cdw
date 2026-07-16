# ============================================================================
# check_A_profile_optimality.jl -- decisive-core diagnostic suite for whether
# A_od* (the Frechet benchmark) is genuinely profile-optimal in the
# unrestricted (all-A_od-free) full-A CC robustness program, or whether the
# outer derivative is blind to profitable moves.
#
# Runs (per A_profile_optimality_report.md):
#   Diagnostic 1: analytical vs AD vs finite-difference moment Jacobian at A*.
#   Diagnostic 8-lite: winner-switch fraction vs h (confirms/refutes that the
#     AD Jacobian's mismatch, if any, comes from the winner-boundary term).
#   Diagnostic 4-lite: fully re-solved directional delta* test at A*, several
#     gamma'_focal targets, several directions, comparing FD slope to the
#     envelope-theorem-predicted slope.
#
# Usage:
#   DVAL=5 julia --project=. diagnostics/check_A_profile_optimality.jl
# (KNITRO env must be sourced first -- see .knitro_env.sh)
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "jacobian_checks.jl"))
include(joinpath(@__DIR__, "directional_resolve.jl"))

const DVAL = parse(Int, get(ENV, "DVAL", "5"))
const WVAL = parse(Int, get(ENV, "WVAL", "8000"))
const OUT_DIR = get(ENV, "OUT_DIR", @__DIR__)
isdir(OUT_DIR) || mkpath(OUT_DIR)

println("="^78)
println(">>> A_od profile-optimality diagnostics: D=$DVAL, W=$WVAL")
println("="^78)

t_start = time()
ctx = build_diag_context(D=DVAL, W=WVAL, gravMoment=1)
D = ctx.D
Aod_offset = ctx.Aod_offset
# IMPORTANT: the Frechet benchmark's A_od_theta block is NOT all-ones under the gammanorm gauge
# (build_theta_gammanorm rescales each column d by s_d = gamma0_d^(-sigma/(mu*(sigma-1))), which is
# 1 only if the OLD-gauge gamma0_d happened to be 1). The true benchmark is theta0_up's own A-block.
Aod0 = ctx.θ0_up[Aod_offset+1:Aod_offset+D^2]
θ0 = copy(ctx.θ0_up)

@printf("mu=%.6f sigma=%.6f theta*=1/mu=%.6f beta=theta*/(sigma-1)=%.6f\n", θ0[1], ctx.σ, 1/θ0[1], ctx.β)
@printf("kappa bounds: [%.6f, %.6f]  gammap bounds: [%.6f, %.6f]  gammap_frechet=%.6f\n",
    ctx.bounds.κ_min, ctx.bounds.κ_max, ctx.bounds.γp_lo, ctx.bounds.γp_hi, θ0[3+D])

# ---- sanity checks at A* ----
# (1) production's raw G is a DEVIATION object: E_F[G_od] = 0 at the benchmark (NOT lambda_od --
#     see context.jl's tradeshare_means docstring).
Gmean0 = tradeshare_means(θ0, ctx.U, ctx.γobj, D, ctx.nTotalMoments)
raw_G_err = maximum(abs.(Gmean0))
@printf("Raw-G zero-moment check: max|E_F[G_od]| = %.3e (should be ~ MC error, ~1/sqrt(W))\n", raw_G_err)
# (2) the user's economic-setup g_od (share-recovered) SHOULD equal lambda_od (Frechet identity).
Sharemean0 = tradeshare_share_means(θ0, ctx.U, ctx.γobj, D, ctx.nTotalMoments)
λ_idxout = zeros(D^2)
for d in 1:D, o in 1:D
    λ_idxout[idx_out(o, d, D)] = ctx.λ[o, d]
end
frechet_err = maximum(abs.(Sharemean0 .- λ_idxout))
@printf("Frechet identity check: max|E_F[share_od] - lambda_od| = %.3e (should be ~ MC error, ~1/sqrt(W))\n", frechet_err)

# ============================================================================
# DIAGNOSTIC 1: analytical vs AD vs FD Jacobian
# ============================================================================
println("\n" * "="^78); println(">>> DIAGNOSTIC 1: moment Jacobian at A* (analytical vs AD vs FD)"); println("="^78)

gamma_norm = gamma(θ0[1] * (1 - ctx.σ) + 1)   # matches production's own moment normalizer exactly (see jacobian_checks.jl docstring)
J_analytical = analytical_jacobian_full(ctx.λ, ctx.β, D; gamma_norm=gamma_norm)
J_ad = ad_jacobian_full(ctx.θ0_up, Aod0, ctx.U, ctx.γobj, D, ctx.nTotalMoments, Aod_offset)

hs_jac = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
fd_reports = NamedTuple[]
J_fd_best = nothing
best_h = NaN
for h in hs_jac
    global J_fd_best, best_h
    J_fd = fd_jacobian_full(ctx.θ0_up, Aod0, ctx.U, ctx.γobj, D, ctx.nTotalMoments, Aod_offset, h)
    rep_vs_analytical = jacobian_error_report(J_analytical, J_fd, D; label="FD(h=$h) vs analytical")
    rep_vs_ad = jacobian_error_report(J_ad, J_fd, D; label="FD(h=$h) vs AD")
    push!(fd_reports, (h=h, rep_vs_analytical..., ad_diag_max_abs=rep_vs_ad.diag_max_abs, ad_off_max_abs=rep_vs_ad.off_max_abs))
    @printf("h=%.0e | FD vs ANALYTICAL: diag maxabs=%.3e maxrel=%.3e | offdiag maxabs=%.3e maxrel=%.3e | cross maxabs=%.3e\n",
        h, rep_vs_analytical.diag_max_abs, rep_vs_analytical.diag_max_rel,
        rep_vs_analytical.off_max_abs, rep_vs_analytical.off_max_rel, rep_vs_analytical.cross_max_abs)
    if h == 1e-4
        J_fd_best = J_fd; best_h = h
    end
end
J_fd_best === nothing && (J_fd_best = fd_jacobian_full(ctx.θ0_up, Aod0, ctx.U, ctx.γobj, D, ctx.nTotalMoments, Aod_offset, 1e-4); best_h = 1e-4)

rep_ad_vs_analytical = jacobian_error_report(J_analytical, J_ad, D; label="AD vs analytical")
rep_ad_vs_fd = jacobian_error_report(J_fd_best, J_ad, D; label="AD vs FD(best h)")
@printf("\nAD vs FULL ANALYTICAL: diag maxabs=%.3e maxrel=%.3e | offdiag maxabs=%.3e maxrel=%.3e\n",
    rep_ad_vs_analytical.diag_max_abs, rep_ad_vs_analytical.diag_max_rel,
    rep_ad_vs_analytical.off_max_abs, rep_ad_vs_analytical.off_max_rel)
@printf("AD vs FD(h=%.0e):    diag maxabs=%.3e maxrel=%.3e | offdiag maxabs=%.3e maxrel=%.3e\n",
    best_h, rep_ad_vs_fd.diag_max_abs, rep_ad_vs_fd.diag_max_rel, rep_ad_vs_fd.off_max_abs, rep_ad_vs_fd.off_max_rel)

# ---- the decisive decomposition check: AD vs the SMOOTH-ONLY prediction (winner boundary excluded) ----
J_smooth = smooth_only_jacobian_full(ctx.λ, ctx.β, D; gamma_norm=gamma_norm)
rep_ad_vs_smooth = jacobian_error_report(J_smooth, J_ad, D; label="AD vs smooth-only prediction")
@printf("AD vs SMOOTH-ONLY prediction (lambda_od/(beta*gamma_norm), off-diag=0): diag maxrel=%.3e | offdiag maxabs=%.3e (should be exactly 0)\n",
    rep_ad_vs_smooth.diag_max_rel, rep_ad_vs_smooth.off_max_abs)

if rep_ad_vs_smooth.diag_max_rel < 0.10 && rep_ad_vs_analytical.off_max_rel > 0.99 && rep_ad_vs_analytical.diag_max_rel > 0.3
    println(">>> FINDING: AD Jacobian matches the SMOOTH-ONLY prediction almost exactly (diag) and is")
    println("    EXACTLY ZERO off-diagonal, while the FULL analytical Jacobian (with the winner-boundary")
    println("    Dirac term) disagrees substantially on both blocks. AD systematically drops the")
    println("    winner-boundary term -- confirmed, not just plausible.")
end

# ============================================================================
# DIAGNOSTIC 8-lite: winner-switch fraction vs h
# ============================================================================
println("\n" * "="^78); println(">>> DIAGNOSTIC 8-lite: winner-switch fraction vs h"); println("="^78)

# pick the (j,d) pair with the LARGEST predicted analytical off-diagonal magnitude (most informative)
function find_best_offdiag_pair(J_analytical, D)
    best_mag = -1.0
    d_focus, j_focus, o_other = 1, 1, 2
    for d in 1:D, o in 1:D, j in 1:D
        j == o && continue
        mag = abs(J_analytical[idx_out(o, d, D), idx_in(j, d, D)])
        if mag > best_mag
            best_mag = mag; d_focus = d; j_focus = j; o_other = o
        end
    end
    return d_focus, j_focus, o_other, best_mag
end
d_focus, j_focus, o_other, best_mag = find_best_offdiag_pair(J_analytical, D)
@printf("Focus perturbation: Aod_theta[j=%d,d=%d], off-diagonal row o=%d (analytical value=%.4e)\n",
    j_focus, d_focus, o_other, J_analytical[idx_out(o_other, d_focus, D), idx_in(j_focus, d_focus, D)])

hs_switch = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
switch_rows = winner_switch_report(ctx.θ0_up, Aod0, ctx.U, ctx.γobj, D, ctx.nTotalMoments, Aod_offset,
    j_focus, d_focus, o_other, hs_switch)
an_diag = J_analytical[idx_out(j_focus, d_focus, D), idx_in(j_focus, d_focus, D)]
an_off = J_analytical[idx_out(o_other, d_focus, D), idx_in(j_focus, d_focus, D)]
@printf("%10s %14s %14s %14s | analytical: diag=%.4e off=%.4e\n", "h", "frac_switch", "fd_diag", "fd_offdiag", an_diag, an_off)
for r in switch_rows
    @printf("%10.1e %14.6f %14.6e %14.6e\n", r.h, r.frac_switch, r.fd_diag, r.fd_offdiag)
end

# ============================================================================
# DIAGNOSTIC 4-lite: fully re-solved directional delta* test at A*
# ============================================================================
println("\n" * "="^78); println(">>> DIAGNOSTIC 4-lite: directional re-solve test at A*"); println("="^78)

ctx_nograv = build_diag_context(D=DVAL, W=WVAL, gravMoment=0)
qvec = flatten_jd(ctx.q_tilde)

# build directions: (a) targeted competitor swap at (o_other, j_focus, d_focus); (b) 3 random gaussian; projected feasible.
Random.seed!(20260714)
directions = Tuple{String,Vector{Float64}}[]
v_comp = zeros(D^2)
v_comp[idx_in(j_focus, d_focus, D)] = 1.0
v_comp[idx_in(o_other, d_focus, D)] = -1.0
v_comp = project_gravity_feasible(v_comp, qvec)
if norm(v_comp) > 1e-10
    push!(directions, ("competitor_swap", v_comp ./ norm(v_comp)))
end
for r in 1:3
    v = randn(D^2)
    v = project_gravity_feasible(v, qvec)
    norm(v) > 1e-10 && push!(directions, ("random$r", v ./ norm(v)))
end

γp_star = ctx_nograv.θ0_up[3+D]
γp_targets = sort(unique([
    γp_star,
    γp_star - 0.15 * (γp_star - ctx_nograv.bounds.γp_lo),
    γp_star + 0.15 * (ctx_nograv.bounds.γp_hi - γp_star),
]))

hs_dir = [-3e-2, -1e-2, -3e-3, -1e-3, 1e-3, 3e-3, 1e-2, 3e-2]

all_dir_rows = NamedTuple[]
for γp in γp_targets
    @printf("\n--- gamma'_focal target = %.6f (Frechet=%.6f) ---\n", γp, γp_star)
    rows, g_prod_Aod = directional_resolve_report(ctx, ctx_nograv, γp, directions; hs=hs_dir, δ_budget=1.0)
    if g_prod_Aod !== nothing
        @printf("||g_prod(Aod block)|| = %.4e   ||g_prod_proj_onto_gravity_feasible|| = %.4e\n",
            norm(g_prod_Aod), norm(project_gravity_feasible(g_prod_Aod, qvec)))
    end
    for r in rows
        @printf("%20s h=%+7.4f  delta*=%12.6g  status=%5d  fd_slope=%12.5g  env_slope=%12.5g\n",
            r.direction, r.h, r.deltastar, r.nStatus, r.fd_slope, r.envelope_slope)
        push!(all_dir_rows, r)
    end
end

# any descent found?
descents = [r for r in all_dir_rows if r.h != 0.0 && !isnan(r.deltastar) && !occursin("COLD", r.direction)]
baseline_by_target = Dict(γp => only(r.deltastar for r in all_dir_rows if r.direction == "baseline(h=0)" && r.γp_target == γp) for γp in γp_targets)
any_descent = any(r.deltastar < baseline_by_target[r.γp_target] - 1e-6 for r in descents)
println("\n>>> Any direction/h found that LOWERS delta* below the A* baseline (net of ~1e-6 numerical slack)? ", any_descent)

# ============================================================================
# CSV output
# ============================================================================
csv_path_jac = joinpath(OUT_DIR, "diag1_jacobian_$(DVAL).csv")
open(csv_path_jac, "w") do io
    println(io, "h,diag_max_abs,diag_mean_abs,diag_max_rel,diag_mean_rel,off_max_abs,off_mean_abs,off_max_rel,off_mean_rel,cross_max_abs,ad_diag_max_abs,ad_off_max_abs")
    for r in fd_reports
        println(io, "$(r.h),$(r.diag_max_abs),$(r.diag_mean_abs),$(r.diag_max_rel),$(r.diag_mean_rel),$(r.off_max_abs),$(r.off_mean_abs),$(r.off_max_rel),$(r.off_mean_rel),$(r.cross_max_abs),$(r.ad_diag_max_abs),$(r.ad_off_max_abs)")
    end
end

csv_path_switch = joinpath(OUT_DIR, "diag8_winnerswitch_$(DVAL).csv")
open(csv_path_switch, "w") do io
    println(io, "h,frac_switch,fd_diag,fd_offdiag,analytical_diag,analytical_offdiag")
    for r in switch_rows
        println(io, "$(r.h),$(r.frac_switch),$(r.fd_diag),$(r.fd_offdiag),$(an_diag),$(an_off)")
    end
end

csv_path_dir = joinpath(OUT_DIR, "diag4_directional_$(DVAL).csv")
open(csv_path_dir, "w") do io
    println(io, "direction,h,gammap_target,deltastar,nStatus,fd_slope,envelope_slope,warm")
    for r in all_dir_rows
        println(io, "$(r.direction),$(r.h),$(r.γp_target),$(r.deltastar),$(r.nStatus),$(r.fd_slope),$(r.envelope_slope),$(r.warm)")
    end
end

println("\nCSV written: $csv_path_jac")
println("CSV written: $csv_path_switch")
println("CSV written: $csv_path_dir")
@printf("\nTOTAL WALL TIME: %.1f s\n", time() - t_start)
println("DIAGNOSTICS DONE")

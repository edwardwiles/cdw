# ============================================================================
# Part 3/4 driver (Stage A, D=4): full-(D+2) fixed-dual FD gradient vs
# current pointwise AD vs fully re-solved profile FD, at real sequential
# iterates (Frechet benchmark + an off-benchmark target), several directions.
#
#   DVAL=4 WVAL=8000 H_FD=0.03 julia --project=. sequential_gravity/derivative_diagnostics/run_full_d2_gradient_validation.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using Printf, Random, LinearAlgebra, DelimitedFiles

const H_FD = parse(Float64, get(ENV, "H_FD", "0.03"))
const Acol_offset = 3

function full_ad_grad_log(θ0::Vector{Float64}, frozen_moments::Function, x_star::Vector{Float64}, D::Int)
    obj = build_fixed_dual_bundle(γ, U, length(θ0), D + 2, frozen_moments; find_smallest=true)
    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ0, obj.U, obj)
    obj.H[:, 2] .= 1.0
    obj(x_star, zeros(length(x_star)))   # nonempty g -> refreshes obj.arg1 = dPsi!(arg0) at (θ0,x_star)
    λ_x0 = collect(@view x_star[2:end])
    ad_raw = ForwardDiff.gradient(θθ -> CS._methodB_envelope_scalar(θθ, frozen_moments, γ, U, λ_x0, obj.arg1, D + 2, D + 3), θ0)
    sign_conv = obj.find_smallest ? -1.0 : 1.0
    ad_level = sign_conv .* (-ad_raw ./ 1e10)
    ad_log = ad_level[Acol_offset+1:Acol_offset+D] .* θ0[Acol_offset+1:Acol_offset+D]
    return ad_log
end

function run_case(label::String, θ0::Vector{Float64})
    println("\n" * "="^78); println(">>> CASE: $label"); println("="^78)
    frozen = freeze_gravity_linearization(θ0, seq_gravcol, grad_R_theta)
    @printf("  R_mean=%.4e Rcol=%.4e ok=%s\n", frozen.R, frozen.Rcol, frozen.ok)
    idr = test_full_fixed_dual_identity(θ0, frozen, γ, U, D; find_smallest=true)
    @printf("  identity: delta*=%.8f Q_full=%.8f rel_diff=%.3e nStatus=%d\n", idr.δ_star, idr.Q_fixed, idr.rel_diff, idr.nStatus)
    idr.rel_diff < 1e-6 || error("identity failed for $label, stopping")

    frozen_moments = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, frozen.θ, frozen.Rcol, frozen.gcol, frozen.dRdθ)
    x_star = idr.x_star

    ad_log = full_ad_grad_log(θ0, frozen_moments, x_star, D)
    t0 = time()
    fd_log = fixed_dual_fd_gradient(θ0, γ, U, frozen_moments, D + 2, x_star, H_FD; l=length(θ0), Acol_offset=Acol_offset, find_smallest=true)
    t_fd = time() - t0
    @printf("  fixed-dual FD gradient wall time: %.3fs\n", t_fd)

    @printf("  %4s %12s %14s %14s %14s\n", "o", "Acol0", "AD(log)", "FD_full(log)", "ratio AD/FD")
    for o in 1:D
        @printf("  %4d %12.5f %14.6e %14.6e %14.4f\n", o, θ0[Acol_offset+o], ad_log[o], fd_log[o], ad_log[o] / fd_log[o])
    end

    return (θ0=θ0, frozen=frozen, frozen_moments=frozen_moments, x_star=x_star, ad_log=ad_log, fd_log=fd_log)
end

case_frechet = run_case("Frechet benchmark", copy(θr0))
γp_lo, γp_hi = KBOUNDS.γp_lo, KBOUNDS.γp_hi
θ_off = copy(θr0); θ_off[3] = θr0[3] - 0.15 * (θr0[3] - γp_lo)
case_off = run_case("off-benchmark (gamma'_focal lower)", θ_off)

# ============================================================================
# Directional comparison against fully re-solved profile FD (the expensive
# ground truth) -- coordinate, random, negative-gradient, mixed directions.
# ============================================================================
function directional_row(case, v::Vector{Float64}, h::Real, dirlabel::String)
    D = length(v)
    θ0, frozen_moments, x_star = case.θ0, case.frozen_moments, case.x_star
    ad_slope = dot(case.ad_log, v)
    fd_slope = fixed_dual_fd_directional_derivative(θ0, build_fixed_dual_bundle(γ, U, length(θ0), D + 2, frozen_moments; find_smallest=true), x_star, v, h; Acol_offset=Acol_offset)
    t0 = time()
    profile_slope, δp, δm, statusp, statusm = full_profile_directional_slope(θ0, γ, U, D + 2, frozen_moments, v, h; Acol_offset=Acol_offset)
    t_profile = time() - t0
    rel_fd = abs(fd_slope - profile_slope) / max(abs(profile_slope), 1e-10)
    rel_ad = abs(ad_slope - profile_slope) / max(abs(profile_slope), 1e-10)
    @printf("  [%s] h=%.3g  AD=% .6e  FD_full=% .6e  profile=% .6e  relerr(FD)=%.3e  relerr(AD)=%.3e  statuses=(%d,%d)  t_profile=%.1fs\n",
        dirlabel, h, ad_slope, fd_slope, profile_slope, rel_fd, rel_ad, statusp, statusm, t_profile)
    return (dirlabel=dirlabel, h=h, ad_slope=ad_slope, fd_slope=fd_slope, profile_slope=profile_slope,
            rel_fd=rel_fd, rel_ad=rel_ad, statusp=statusp, statusm=statusm)
end

rows = NamedTuple[]
for case in (case_frechet, case_off)
    println("\n" * "="^78); println(">>> Directional profile-FD comparison for this case"); println("="^78)
    r_focus = argmax(abs.(case.fd_log))
    v_coord = zeros(D); v_coord[r_focus] = 1.0
    push!(rows, directional_row(case, v_coord, H_FD, "coord_o=$r_focus"))

    Random.seed!(20260714)
    v_rand = randn(D); v_rand ./= norm(v_rand)
    push!(rows, directional_row(case, v_rand, H_FD, "random"))

    v_multi = zeros(D); v_multi[1] = 1.0; v_multi[min(3, D)] = -0.7
    v_multi ./= norm(v_multi)
    push!(rows, directional_row(case, v_multi, H_FD, "multi_origin"))

    v_neg = -case.fd_log ./ norm(case.fd_log)
    push!(rows, directional_row(case, v_neg, H_FD, "neg_grad_full"))
end

open(joinpath(@__DIR__, "part4_full_d2_directional_D$(D)_W$(W)_h$(H_FD).csv"), "w") do io
    writedlm(io, ["dirlabel" "h" "ad_slope" "fd_slope" "profile_slope" "rel_fd" "rel_ad" "statusp" "statusm"], ',')
    for r in rows
        writedlm(io, [[r.dirlabel r.h r.ad_slope r.fd_slope r.profile_slope r.rel_fd r.rel_ad r.statusp r.statusm]], ',')
    end
end

println("\n" * "="^78); println(">>> SUMMARY"); println("="^78)
@printf("%-16s %8s %10s %10s\n", "direction", "h", "relerr_FD", "relerr_AD")
for r in rows
    @printf("%-16s %8.3g %10.4f %10.4f\n", r.dirlabel, r.h, r.rel_fd, r.rel_ad)
end
println("\nPART 3/4 gradient validation DONE")

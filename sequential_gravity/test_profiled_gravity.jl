# Phase-1 validation of the profiled-gravity core (spec §20 tests C,D,E,F,G,H,I/J).
# Self-contained and KNITRO-free: synthetic Exp(1) draws + a known u.  Run with
#   julia --project=. sequential_gravity/test_profiled_gravity.jl
#
# The directional-derivative test (I/J) is the mandatory gate: it checks the influence function
# ψ_R against exact re-inversion at a tilted distribution (spec §16). Everything downstream in
# Phase 2 depends on it.  We validate the softmax-consistent model at temperature ρ=RHO (spec §10):
# the inversion, the share Jacobian, and the influence function all use the SAME ρ, so ψ_R is the
# exact derivative of the (smoothed) residual.  ρ→0 recovers the hard-max model.

include(joinpath(@__DIR__, "profiled_gravity.jl"))
using .ProfiledGravity
using LinearAlgebra, Random, Printf
using ForwardDiff

const RESULTS = Tuple{String,Bool,String}[]
report(name, ok, detail = "") = (push!(RESULTS, (name, ok, detail)); @printf("  [%s] %s  %s\n", ok ? "PASS" : "FAIL", name, detail))

# ---- shared synthetic setup ----------------------------------------------------------------
Random.seed!(2024)
const D = 5
const S = 4000
const σ = 2.5
const μ = 1 / (σ - 1) * 0.4
const ref = 1
const RHO = 3e-3           # softmax temperature (spec §10 consistent smoothing)

Uraw = -log.(rand(S, D))                 # Exp(1) draws (as in genExpRands!)
log_x = build_log_x_fromU(Uraw, μ, σ)    # = μ(1-σ) log U

u_data = 0.6 .* randn(D, D)
for d in 1:D; u_data[ref, d] = 0.0; end

# observed shares λ̂[:,d] = smoothed model shares at u_data under UNIFORM weights (fixed data)
p_uniform = fill(1 / S, S)
λ̂ = zeros(D, D)
for d in 1:D
    λ̂[:, d] .= dest_stats(log_x, log.(p_uniform), u_data[:, d]; ρ = RHO).share
end
@assert all(λ̂ .> 1e-6)

# a non-uniform LFD-like weight vector p0 (bounded tilt of uniform)
htilt = randn(S); htilt .-= sum(p_uniform .* htilt)
p0 = p_uniform .* (1 .+ 0.5 .* tanh.(htilt)); p0 ./= sum(p0)
@assert all(p0 .> 0)

logw = 0.3 .* randn(D)
logτ = 0.4 .* randn(D, D)
for o in 1:D; logτ[o, o] = 0.0; end

println("=== Phase-1 profiled-gravity validation (D=$D, S=$S, σ=$σ, μ=$(round(μ,digits=4)), ρ=$RHO) ===")

# ---- Test C: destination inversion round trip ----------------------------------------------
# (i) invert under UNIFORM p (same as data): must recover u_data exactly (up to gauge).
# (ii) invert under p0: recovered shares must match λ̂ tightly.
let
    okall = true; worst_share = 0.0; worst_u = 0.0
    for d in 1:D
        invu = invert_destination(log_x, p_uniform, λ̂[:, d]; ref = ref, ρ = RHO, tol = 1e-11, maxit = 200)
        worst_u = max(worst_u, maximum(abs.(invu.u_full .- u_data[:, d])))
        inv0 = invert_destination(log_x, p0, λ̂[:, d]; ref = ref, ρ = RHO, tol = 1e-11, maxit = 200)
        worst_share = max(worst_share, inv0.max_abs_share_error)
        okall &= invu.converged && inv0.converged
    end
    report("C inversion round trip", okall && worst_share < 1e-10 && worst_u < 1e-6,
           @sprintf("recover u err=%.2e, share err under p0=%.2e", worst_u, worst_share))
end

# ---- Test D: AD gradient of φ = model share --------------------------------------------------
let
    d = 3
    u = invert_destination(log_x, p0, λ̂[:, d]; ref = ref, ρ = RHO, tol = 1e-11).u_full
    fi = free_idx(ref, D)
    g_ad = ForwardDiff.gradient(uf -> ProfiledGravity.potential_free(uf, log_x, log.(p0); ref = ref, ρ = RHO), u[fi])
    st = dest_stats(log_x, log.(p0), u; ρ = RHO)
    err = maximum(abs.(g_ad .- st.share[fi]))
    report("D AD grad φ = share", err < 1e-10, @sprintf("max err=%.2e", err))
end

# ---- Test E: AD Hessian symmetric, = smoothed closed form; hard-max AD = hard closed form ----
let
    d = 4
    u = invert_destination(log_x, p0, λ̂[:, d]; ref = ref, ρ = RHO, tol = 1e-11).u_full
    st = dest_stats(log_x, log.(p0), u; ρ = RHO)
    H_ad  = share_jacobian_ad(log_x, log.(p0), u; ref = ref, ρ = RHO)
    H_sm  = share_jacobian_smoothed(st.rweight, st.W, st.share, RHO; ref = ref)
    sym = maximum(abs.(H_ad .- H_ad'))
    diff_sm = maximum(abs.(H_ad .- H_sm))
    # hard-max cross-check (ρ=0): AD hessian of hard potential = diag(share)-ss'
    st0 = dest_stats(log_x, log.(p0), u; ρ = 0.0)
    H_ad0 = share_jacobian_ad(log_x, log.(p0), u; ref = ref, ρ = 0.0)
    H_cf0 = share_jacobian_closed(st0.share; ref = ref)
    diff_hard = maximum(abs.(H_ad0 .- H_cf0))
    report("E AD Hessian = closed forms (soft & hard)",
           sym < 1e-9 && diff_sm < 1e-7 && diff_hard < 1e-9,
           @sprintf("asym=%.1e |AD-sm|=%.1e |AD0-cf0|=%.1e", sym, diff_sm, diff_hard))
end

# ---- Test F: normalization invariance --------------------------------------------------------
let
    d = 2
    u = invert_destination(log_x, p0, λ̂[:, d]; ref = ref, ρ = RHO, tol = 1e-11).u_full
    st1 = dest_stats(log_x, log.(p0), u; ρ = RHO)
    st2 = dest_stats(log_x, log.(p0), u .+ 3.7; ρ = RHO)
    share_inv = maximum(abs.(st1.share .- st2.share))
    umat = copy(u_data)
    R1 = gravity_residual(umat, logτ, logw, σ).R_sum
    umat2 = copy(umat); umat2[:, d] .+= 2.1
    R2 = gravity_residual(umat2, logτ, logw, σ).R_sum
    report("F normalization invariance (shares & gravity)",
           share_inv < 1e-12 && abs(R1 - R2) < 1e-10,
           @sprintf("Δshare=%.2e, ΔR=%.2e", share_inv, abs(R1 - R2)))
end

# ---- Test G: gravity identity (logA form == U form) -----------------------------------------
let
    gr = gravity_residual(copy(u_data), logτ, logw, σ)
    R_via_u = sum(gr.Qt .* (gr.Qt .+ gr.ut ./ (σ - 1)))
    report("G gravity identity logA==U form", abs(gr.R_sum - R_via_u) < 1e-10,
           @sprintf("|Δ|=%.2e (R=%.4f)", abs(gr.R_sum - R_via_u), gr.R_sum))
end

# ---- Test H: adjoint identity  ⟨c, H⁻¹v⟩ = ⟨H⁻ᵀc, v⟩ ----------------------------------------
let
    d = 5
    u = invert_destination(log_x, p0, λ̂[:, d]; ref = ref, ρ = RHO, tol = 1e-11).u_full
    st = dest_stats(log_x, log.(p0), u; ρ = RHO)
    H = share_jacobian_smoothed(st.rweight, st.W, st.share, RHO; ref = ref)
    c = randn(D - 1); v = randn(D - 1)
    lhs = dot(c, H \ v); rhs = dot(H' \ c, v)
    report("H adjoint identity", abs(lhs - rhs) < 1e-8 * (1 + abs(lhs)), @sprintf("|Δ|=%.2e", abs(lhs - rhs)))
end

# ---- Tests I/J: directional-derivative validation of the influence function -----------------
let
    focal = 2
    omitted = [d for d in 1:D if d != focal]

    function invert_all(p; warm = nothing)
        umat = copy(u_data)
        for d in omitted
            ui = warm === nothing ? nothing : warm[:, d]
            inv = invert_destination(log_x, p, λ̂[:, d]; ref = ref, ρ = RHO, tol = 1e-12, maxit = 300, u_init = ui)
            umat[:, d] .= inv.u_full
        end
        return umat
    end

    umat0 = invert_all(p0)
    infl = influence_function(log_x, p0, umat0, λ̂, omitted, logτ, logw, σ; ref = ref, ρ = RHO, scale = :R_beta)
    @printf("    R_beta(0)=%.6f  R_sum(0)=%.6f  E_F[ψ_R]=%.2e\n", infl.R_beta, infl.R_sum, infl.Eψ)
    for pd in infl.per_dest
        @printf("    dest %d: M_d=%.3e cond(H)=%.2e smin=%.2e adj_resid=%.1e share_err=%.1e\n",
                pd.d, pd.M_d, pd.condH, pd.smin_H, pd.adjoint_resid, pd.max_share_err)
    end
    report("I ψ_R centered (E_F[ψ_R]≈0)", abs(infl.Eψ) < 1e-8, @sprintf("|E|=%.2e", abs(infl.Eψ)))

    ndir = 3
    ok_dd = true
    for dir in 1:ndir
        h = randn(S); h .-= dot(p0, h); h ./= maximum(abs.(h))    # E_{p0}[h]=0, bounded
        predicted = dot(p0, h .* infl.ψ_R)
        best_rel = Inf; best_t = 0.0
        for t in (1e-3, 3e-4, 1e-4)
            for sgn in (1.0, -1.0)
                tt = sgn * t
                pt = p0 .* (1 .+ tt .* h)
                any(pt .<= 0) && continue
                pt ./= sum(pt)
                grt = gravity_residual(invert_all(pt; warm = umat0), logτ, logw, σ)
                numeric = (grt.R_beta - infl.R_beta) / tt
                rel = abs(numeric - predicted) / (abs(predicted) + 1e-12)
                rel < best_rel && (best_rel = rel; best_t = tt)
            end
        end
        @printf("    dir%d: pred=%.5e  best rel_err=%.2e (t=%.0e)\n", dir, predicted, best_rel, best_t)
        ok_dd &= best_rel < 5e-3
    end
    report("J directional-derivative (ψ_R vs exact re-inversion)", ok_dd, "tol 5e-3")
end

# ---- summary --------------------------------------------------------------------------------
println("\n=== SUMMARY ===")
npass = count(r -> r[2], RESULTS)
for (name, ok, _) in RESULTS
    @printf("  %-46s %s\n", name, ok ? "PASS" : "FAIL")
end
@printf("%d/%d passed\n", npass, length(RESULTS))
exit(npass == length(RESULTS) ? 0 : 1)

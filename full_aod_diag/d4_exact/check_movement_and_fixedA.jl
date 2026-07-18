# User question 1: (a) did the verified upper-direction point actually move A_od away from A_od*,
# and (b) is its kappa better than the best achievable by holding A_od==A_od* fixed and only
# optimizing gamma'_focal? Reuses only already-validated machinery (evaluate_fullA, pivot_expand).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))

const COMMIT = "a377fff"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D

# starting point A_od* (from theta0_up, the calibration/build_theta_gammanorm point)
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
z0 = log.(Aod_theta0)
zfree0 = pivot_reduce(z0, pe)
gp0 = ctx.θ0_up[3+D]

# the maxit=40 best-feasible point (more converged than the maxit=15 one)
w_best = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
          0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
          1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
          0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
gp_found = w_best[1]; zfree_found = w_best[2:end]
z_found = pivot_expand(zfree_found, pe)
Aod_theta_found = exp.(z_found)

function structural_Aod_and_AodPow(θ_full)
    D = ctx.D; μ = θ_full[1]
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
    lambda_g = reshape(ctx.γ.P, (D, D))'
    Aod = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ)) .* (lambda_g ./ lambda_g[1,:]')
    AodPow = (Aod ./ ctx.γ.cHat) .^ (-μ)
    return Aod, AodPow
end

θ_full0 = copy(ctx.θ0_up)
θ_full_found = CS.reconstruct_full(vcat(gp_found, vec(Aod_theta_found)), ctx.m)
Aod0, AodPow0 = structural_Aod_and_AodPow(θ_full0)
Aod_f, AodPow_f = structural_Aod_and_AodPow(θ_full_found)
A_od0 = 1.0 ./ AodPow0     # gravity_tariff.jl's convention: structural A_od = 1/AodPow
A_od_f = 1.0 ./ AodPow_f

println("="^78); println("(a) DID A_od ACTUALLY MOVE?"); println("="^78)
rel_change_Aodtheta = abs.(Aod_theta_found .- Aod_theta0) ./ abs.(Aod_theta0)
rel_change_Aod = abs.(A_od_f .- A_od0) ./ abs.(A_od0)
println("Aod_theta (raw outer parameter):")
println("  max relative change = ", maximum(rel_change_Aodtheta), "   RMS relative change = ", sqrt(sum(rel_change_Aodtheta.^2)/length(rel_change_Aodtheta)))
println("  min/max level at start:  ", extrema(Aod_theta0))
println("  min/max level at found:  ", extrema(Aod_theta_found))
println("\nStructural A_od (=1/AodPow, gravity_tariff.jl's economic object):")
println("  max relative change = ", maximum(rel_change_Aod), "   RMS relative change = ", sqrt(sum(rel_change_Aod.^2)/length(rel_change_Aod)))
println("  per-entry relative change matrix (o=row,d=col):")
show(stdout, "text/plain", rel_change_Aod); println()
println("\ngamma'_focal: start=", gp0, "  found=", gp_found, "  (this alone moving is NOT interesting -- A_od moving is the question)")

println("\n" * "="^78); println("(b) FIXED-A COMPARISON: best achievable kappa holding A_od == A_od* fixed"); println("="^78)
function Delta_fixedA(gp::Float64)
    xf = vcat(gp, vec(Aod_theta0))   # A-block held EXACTLY at A_od*, only gamma'_focal varies
    r = evaluate_fullA(xf, ctx; cache = nothing, warm = true)
    return r.Delta_dual
end

println("Scanning Delta(gamma', A_od=A_od*) to check monotonicity before bisecting...")
gp_scan = range(ctx.bounds.γp_lo, gp0, length = 12)
for gp in gp_scan
    println("  gamma'=", round(gp, digits=4), "  Delta=", Delta_fixedA(gp))
end

# bisection for the smallest gamma' (upper direction) with Delta(gamma', A*) <= delta=1
lo, hi = ctx.bounds.γp_lo, gp0   # Delta(gp0)~0 (feasible), Delta(bounds.γp_lo) expected large/infeasible
@assert Delta_fixedA(hi) <= ctx.δ "base point itself infeasible -- scan assumption violated"
if Delta_fixedA(lo) <= ctx.δ
    println("Even the theoretical bound gamma'=", lo, " is feasible holding A fixed -- no bisection needed, fixed-A reaches the bound")
    gp_fixedA_star = lo
else
    for _ in 1:40
        mid = (lo + hi) / 2
        if Delta_fixedA(mid) <= ctx.δ
            global hi = mid
        else
            global lo = mid
        end
    end
    global gp_fixedA_star = hi
end
κ_fixedA = 1 - gp_fixedA_star^(ctx.σ / (ctx.σ - 1))
κ_movedA = 1 - gp_found^(ctx.σ / (ctx.σ - 1))
println("\nFixed-A* extremal gamma'_focal = ", gp_fixedA_star, "  -> kappa_fixedA = ", κ_fixedA)
println("Moved-A  (verified point)  gamma'_focal = ", gp_found, "  -> kappa_movedA = ", κ_movedA)
println("kappa gain from moving A_od = ", κ_movedA - κ_fixedA, "  (", round(100*(κ_movedA-κ_fixedA)/κ_fixedA, digits=2), "% relative)")
verdict = κ_movedA > κ_fixedA + 1e-6 ? "GENUINE improvement from moving A_od (not just a fixed-A-reachable point)" :
          κ_movedA < κ_fixedA - 1e-6 ? "WORSE than fixed-A -- the search did not find a real improvement (or fixed-A bisection found a better point)" :
          "ESSENTIALLY THE SAME as fixed-A -- consistent with the solver staying in the A_od* basin"
println("\nVERDICT: ", verdict)

open(joinpath(OUTDIR, "movement_and_fixedA_check.txt"), "w") do io
    println(io, "max_rel_change_Aod_theta = ", maximum(rel_change_Aodtheta))
    println(io, "max_rel_change_structural_Aod = ", maximum(rel_change_Aod))
    println(io, "gp_fixedA_star = ", gp_fixedA_star, "  kappa_fixedA = ", κ_fixedA)
    println(io, "gp_found = ", gp_found, "  kappa_movedA = ", κ_movedA)
    println(io, "kappa_gain = ", κ_movedA - κ_fixedA)
    println(io, "verdict = ", verdict)
end
println("\nWrote ", joinpath(OUTDIR, "movement_and_fixedA_check.txt"))

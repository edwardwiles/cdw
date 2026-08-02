# Phase 4 gate: generic_psi_2026-08-01.jl's Float64 output vs production CS.Psi!/dPsi!/ddPsi!
# (cc_algo/Psi.jl), scalar-by-scalar, over a grid spanning both branches (u<=1 and u>1) plus the
# u==1 boundary. Also checks ForwardDiff.Dual compatibility (task requirement: "whose Float64
# output is first verified against production" before it is trusted for Dual arguments).
include(joinpath(@__DIR__, "generic_psi_2026-08-01.jl"))
using ForwardDiff, Printf, Random

# CS is the module production's Psi!/dPsi!/ddPsi! live in -- loaded via context.jl in every other
# gate this session touches; load it minimally here without the full D4 context machinery.
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "Psi.jl"))

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

Random.seed!(2026)
grid = vcat([-5.0, -1.0, 0.0, 0.5, 0.999, 1.0, 1.0001, 1.5, 2.0, 5.0, 10.0], 4 .* randn(20))

function compare_to_production(grid)
    a1 = Vector{Float64}(undef, 1); a0 = Vector{Float64}(undef, 1)
    maxdiff_psi = 0.0; maxdiff_dpsi = 0.0; maxdiff_ddpsi = 0.0
    for u in grid
        a0[1] = u
        Psi!(a1, a0);   maxdiff_psi   = max(maxdiff_psi,   abs(a1[1] - Psi_scalar(u)))
        dPsi!(a1, a0);  maxdiff_dpsi  = max(maxdiff_dpsi,  abs(a1[1] - dPsi_scalar(u)))
        ddPsi!(a1, a0); maxdiff_ddpsi = max(maxdiff_ddpsi, abs(a1[1] - ddPsi_scalar(u)))
    end
    return maxdiff_psi, maxdiff_dpsi, maxdiff_ddpsi
end
maxdiff_psi, maxdiff_dpsi, maxdiff_ddpsi = compare_to_production(grid)
check(@sprintf("Psi_scalar matches CS.Psi! over %d-point grid incl. u=1 boundary (max|Δ|=%.3e)", length(grid), maxdiff_psi), maxdiff_psi == 0.0)
check(@sprintf("dPsi_scalar matches CS.dPsi! (max|Δ|=%.3e)", maxdiff_dpsi), maxdiff_dpsi == 0.0)
check(@sprintf("ddPsi_scalar matches CS.ddPsi! (max|Δ|=%.3e)", maxdiff_ddpsi), maxdiff_ddpsi == 0.0)

# ForwardDiff self-consistency: d/du Psi_scalar == dPsi_scalar, d/du dPsi_scalar == ddPsi_scalar,
# via ForwardDiff.derivative (proves Dual-compatibility, not just Float64 correctness).
maxdiff_fd1 = maximum(u -> abs(ForwardDiff.derivative(Psi_scalar, u) - dPsi_scalar(u)), grid)
maxdiff_fd2 = maximum(u -> abs(ForwardDiff.derivative(dPsi_scalar, u) - ddPsi_scalar(u)), grid)
check(@sprintf("ForwardDiff.derivative(Psi_scalar) matches dPsi_scalar (max|Δ|=%.3e)", maxdiff_fd1), maxdiff_fd1 < 1e-10)
check(@sprintf("ForwardDiff.derivative(dPsi_scalar) matches ddPsi_scalar (max|Δ|=%.3e)", maxdiff_fd2), maxdiff_fd2 < 1e-10)

println(ALL_PASS[] ? "\nALL PASS" : "\nSOME FAILED")
exit(ALL_PASS[] ? 0 : 1)

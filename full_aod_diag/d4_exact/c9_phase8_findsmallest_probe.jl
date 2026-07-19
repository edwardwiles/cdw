# Quick probe: which find_smallest setting is dual-feasible at (g_offset, A_natural)
# for the upper/lower branch offsets, before committing to a long profile/polish run.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using Printf

x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

for fs in (true, false)
    ctx = d20_real_setup(W = 80000, find_smallest = fs)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2
    Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
    z0 = log.(Aod_theta_natural)
    zfree0 = pivot_reduce(reshape(z0, D, D), pe)
    gp0 = ctx.θ0_up[3+D]
    for (lab, g) in (("calibration", gp0), ("upper(gp0*1.01)", gp0 * 1.01), ("lower(gp0*0.99)", gp0 * 0.99))
        xf = x_free_from_w(vcat(g, zfree0), pe)
        r = evaluate_fullA(xf, ctx; warm = false)
        @printf("find_smallest=%s  %-20s g=%.6f  inner_status=%-5d  Delta=%s\n", fs, lab, g, r.inner_status, string(r.Delta_dual))
    end
end

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
println("D = ", ctx.D)
println("W (draws) = ", size(ctx.U, 1))
println("ncore (obj0.d) = ", ctx.obj.d)
println("outer_constr_index = ", ctx.obj.outer_constr_index)
println("free_idx length = ", length(ctx.free_idx))

for L in (10, 20, 50)
    aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    println("L=$L  ncm=", aug.ncm, "  d_new=", aug.obj_cm.d, "  inner Hessian dim (=outer_constr_index)=", aug.obj_cm.outer_constr_index)
end

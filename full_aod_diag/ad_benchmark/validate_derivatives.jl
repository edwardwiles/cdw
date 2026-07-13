# §7: correctness tests A-D (E-G partially; see final report for what's covered).
include("setup_context.jl")
include("derivative_core.jl")
include("derivative_methods.jl")

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
points = data["points"]

results = NamedTuple[]

for name in (:A, :B, :C, :D)
    pt = points[name]
    θ = pt.θ
    ctx = (U = pp.U, γobj = pp.γ, λ = pt.λ, arg1 = pt.arg1, d = data["nTotalMoments"], outer_constr_index = data["outer_constr_index"])

    # --- production reference (recomputed exactly, as in quick_validate.jl) ---
    obj = make_ad_obj(pp, so)
    CS.inner_loop_internal(obj, θ)
    obj(pt.x, Float64[], Float64[]; constr = zeros(obj.d - obj.outer_constr_index + 2))
    g = zeros(obj.l); jac = zeros((obj.d - obj.outer_constr_index + 2) * obj.l)
    obj(pt.x, g, θ; jac = jac)
    ∂c_∂θ = reshape(jac, obj.l, obj.d - obj.outer_constr_index + 2)'
    prod_div_grad = ∂c_∂θ[1, :]

    # --- Method A: dense Jacobian + contraction ---
    tA = @elapsed JA = method_A_dense_jacobian(θ, ctx)
    gA = contract_dense_to_div_grad(JA, ctx)
    errA = maximum(abs.(gA .- prod_div_grad)) / maximum(abs.(prod_div_grad))

    # --- Method B: direct scalar ForwardDiff ---
    tB = @elapsed gB = method_B_forward_scalar(θ, ctx)
    errB = maximum(abs.(gB .- prod_div_grad)) / maximum(abs.(prod_div_grad))

    # --- Method C: Enzyme reverse ---
    local gC, errC, tC, enzyme_ok, enzyme_err
    enzyme_ok = true; enzyme_err = ""
    try
        tC = @elapsed gC = method_C_enzyme_reverse(θ, ctx)
        errC = maximum(abs.(gC .- prod_div_grad)) / maximum(abs.(prod_div_grad))
    catch e
        enzyme_ok = false
        enzyme_err = sprint(showerror, e)
        gC = fill(NaN, obj.l); errC = NaN; tC = NaN
    end

    println("=== point $name ===")
    @printf("  Method A (dense+contract):  err=%.3e  time=%.3fs\n", errA, tA)
    @printf("  Method B (scalar FD):       err=%.3e  time=%.3fs\n", errB, tB)
    if enzyme_ok
        @printf("  Method C (Enzyme reverse):  err=%.3e  time=%.3fs\n", errC, tC)
    else
        println("  Method C (Enzyme reverse):  FAILED — ", first(split(enzyme_err, "\n")))
    end

    push!(results, (point=name, errA=errA, tA=tA, errB=errB, tB=tB, enzyme_ok=enzyme_ok, errC=errC, tC=tC, enzyme_err=enzyme_err))
end

@save joinpath(@__DIR__, "validate_results.jld2") results
println("VALIDATE DONE")

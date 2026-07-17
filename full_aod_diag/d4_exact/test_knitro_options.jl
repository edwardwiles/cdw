# Task §16 validation. Run: julia --project=. full_aod_diag/d4_exact/test_knitro_options.jl
# Confirms the eval_fcga=yes -> silent hessopt=4->6(LBFGS) fallback on the REAL economic problem
# (not just a trivial probe), and that eval_fcga=no restores genuine BFGS. maxit=3 to keep this cheap.
include(joinpath(@__DIR__, "context.jl"))

const COMMIT = "bf00b00"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup(outer_loop_opt = joinpath(@__DIR__, "csw_outer_default_maxit3.opt"))
x0 = CS.pack_free(ctx.θ0_up, ctx.m)

function make_div_grad_fn!(obj, m)
    ncon_inner = obj.d - obj.outer_constr_index + 2
    cfg_cache = Ref{Any}(nothing)
    return function (g_free, x_free, θ_full, inner_x)
        obj(inner_x, Float64[], Float64[]; constr = zeros(ncon_inner))
        λ = @view inner_x[2:end]
        ctx2 = (U = obj.U, γobj = obj.γ, λ = λ, arg1 = obj.arg1, d = obj.d, outer_constr_index = obj.outer_constr_index)
        f = x -> envelope_scalar_div_ctx(reconstruct_full(x, m), ctx2)
        if cfg_cache[] === nothing
            cfg_cache[] = ForwardDiff.GradientConfig(f, x_free)
        end
        ForwardDiff.gradient!(g_free, f, x_free, cfg_cache[])
        return g_free
    end
end
function obj_grad_fn!(g_free, x_free, obj)
    fill!(g_free, 0.0); g_free[1] = (-1.0)^obj.find_smallest
end

for (label, optfile, logfile) in [
        ("DEFAULT (eval_fcga=yes, hessopt requested=4)", "csw_outer_default_maxit3.opt", "knitro_default.log"),
        ("FIX (eval_fcga=no, hessopt=4 should take effect)", "csw_outer_fcga_no_maxit3.opt", "knitro_fcga_no.log")]
    println("="^78); println(label); println("="^78)
    ctx_i = d4_exact_setup(outer_loop_opt = joinpath(@__DIR__, optfile))
    div_grad_fn! = make_div_grad_fn!(ctx_i.obj, ctx_i.m)
    ogf! = (g, x) -> obj_grad_fn!(g, x, ctx_i.obj)
    gravity_grad_fn! = (g, x) -> gravity_grad_free!(g, x, ctx_i.D, ctx_i.Aod_free_pos, ctx_i.fixed_vals[1], ctx_i.q_tilde, ctx_i.N_obs)

    logpath = joinpath(OUTDIR, logfile)
    open(logpath, "w") do io
        redirect_stdout(io) do
            r = CS.outer_loop_cached(ctx_i.obj, ctx_i.m, ctx_i.θ_lo, ctx_i.θ_hi, ctx_i.θ0_up;
                obj_grad_fn! = ogf!, div_grad_fn! = div_grad_fn!,
                gravity_grad_fn! = gravity_grad_fn!, has_gravity = true, gravity_value_scale = -1.0 / ctx_i.N_obs,
                use_cache = true, outer_loop_opt = joinpath(@__DIR__, optfile))
            println("RESULT: status=", r.nStatus, " outer_iters=", r.outer_iters, " opt_err=", r.opt_err)
        end
    end
    println("KNITRO log written: ", logpath)
    hessopt_msg = filter(l -> occursin("hessopt", l) || occursin("Changing hessopt", l), readlines(logpath))
    for l in hessopt_msg
        println("  >> ", l)
    end
    println(isempty(hessopt_msg) ? "  (no hessopt override message found in log)" : "  ^^ hessopt-related message(s) found above")
end

println("\nKNITRO OPTION VERIFICATION COMPLETE -- see results/fullA_d4/", "bf00b00", "/knitro_default.log and knitro_fcga_no.log")

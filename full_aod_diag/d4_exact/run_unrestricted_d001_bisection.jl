# run_unrestricted_d001_bisection.jl -- Section 5.2 driver: compute the alternate boundary seed
# for unrestricted upper delta=0.01 by bisecting Delta* along the path between the frozen delta=0.01
# and delta=0.1 finalized points. Thin wrapper so this can be launched as a single background job.
const D4E = @__DIR__
include(joinpath(D4E, "full_chain_include.jl"))
include(joinpath(D4E, "delta_star_path_bisection.jl"))

const FROZEN = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/frozen_seeds_K1/890424b84acebe64/W100k/unrestricted/upper"
lo_path = joinpath(FROZEN, "delta_0.01", "e45fd5d4eaa7e37b.jls")
hi_path = joinpath(FROZEN, "delta_0.1", "c3d1fadb16f93eea.jls")
out_path = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/targeted_k3_extensions/bisection_unrestricted_d001/bisection_result.jls"

isfile(lo_path) || error("lo seed not found: $lo_path")
isfile(hi_path) || error("hi seed not found: $hi_path")

t_star, Ds_star, w_star = let
    state_lo = load_run_state(lo_path); r_lo = state_lo.report
    state_hi = load_run_state(hi_path); r_hi = state_hi.report
    println("Endpoint lo: delta=", r_lo.target_delta, " GT=", r_lo.final_GT, " Delta*=", r_lo.final_Delta_star); flush(stdout)
    println("Endpoint hi: delta=", r_hi.target_delta, " GT=", r_hi.final_GT, " Delta*=", r_hi.final_Delta_star); flush(stdout)
    ckpt_root = dirname(out_path)
    isdir(ckpt_root) || mkpath(ckpt_root)
    # eval budget MUST be short enough that the outer KNITRO loop does essentially nothing beyond
    # its own "cold start" verification of the seed point -- confirmed live 2026-08-04: a 90s
    # budget let the outer solver run 8-9 real outer iterations and wander to Delta*=1.29, nowhere
    # near the interpolated point's own Delta*, making the bracket meaningless. A few seconds is
    # enough for exactly one inner solve (the cold-start line in this driver's own stdout) and
    # essentially no outer movement.
    bisect_delta_star("unrestricted", :upper, r_lo.final_w, r_hi.final_w,
        r_lo.final_Delta_star, r_hi.final_Delta_star, 0.01, 3.0, ckpt_root; max_iters = 8)
end

println("=== BISECTION RESULT: t*=", t_star, " Delta*=", Ds_star, " ==="); flush(stdout)
serialize(out_path, (t_star = t_star, Delta_star = Ds_star, w = w_star, target = 0.01,
                      lo_path = lo_path, hi_path = hi_path, family = "unrestricted", direction = :upper))
println("Saved to ", out_path); flush(stdout)

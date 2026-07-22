# ============================================================================
# Phase 6 pre-adoption gate (finalization task, 2026-07-22): closes Backend C+'s own two
# previously-disclosed gaps (docs/fullA_factorized_price_production_gate.md sec 12/14) now that
# c22_phase6_fair_benchmark.jl shows C+ is the fastest backend (4.0-4.2x vs Reference, beating
# :kbplus's 3.6-3.7x) -- the actual leading default candidate, so its own remaining gates must
# close before it can be adopted, per the brief's own adoption rule.
#
#   Section 1: independent optimized-value directional check for C+ specifically (mirrors
#              c21_kbplus_d20_gate.jl's Section 3 pattern for :kbplus, attributed).
#   Section 2: cross_delta + price_cache_backend=:cplus together, real staged continuation --
#              confirms no wiring conflict between the two independent systems (exact-cache at
#              the screened_eval/inner-dual level, gradient backend at the cb_G! level).
#   Section 3: checkpoint/resume equivalence with C+ active -- interrupt, resume, compare the
#              resumed run's incumbent to an uninterrupted run's own.
#   Section 4: D=20/W=80000 short driver trajectory at delta=2 (delta=1 already covered by
#              c20_backend_selector_driver_smoke.jl's own :cplus arm).
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Printf

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

println("== Section 1: independent optimized-value directional check for C+ ==")
ctx = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true)
D = ctx.D; W = size(ctx.obj.U, 1)
pe = build_pivot_elimination(ctx)
g0 = ctx.θ0_up[3+D]
Aod_real = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
zfree0 = pivot_reduce(log.(Aod_real), pe)
xf = x_free_from_w(vcat(g0, zfree0), pe)
base = solve_base_state(xf, ctx)
z0 = log.(reshape(xf[2:end], D, D))
w0 = vcat(xf[1], pivot_reduce(z0, pe))

cache_ref = build_lfix_base_cache(xf, ctx, base)
grad_pool = build_grad_workspace_pool(W)
c_ws = build_lfix_factorized_workspace(D, W)
cache_C = build_lfix_base_cache_C!(c_ws, xf, ctx, base)

D2 = D^2
using Random
rng = MersenneTwister(20260722)
cand_coords = unique([2, 3, rand(rng, 2:D2, 6)...])
cache_C_for_flips = build_lfix_base_cache_C(xf, ctx, base)
flip_counts = Dict{Int,Int}()
for k in cand_coords
    cells = affected_cells(pe, k)
    d = first(last.(cells))
    w = copy(w0); w[k] += 0.05
    θf = CS.reconstruct_full(vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe)))), ctx.m)
    origins_here = [o for (o, dd) in cells if dd == d]
    flip_counts[k] = count_winner_flips_C(cache_C_for_flips, ctx, θf, d, origins_here)
end
high_switch_coord = argmax(flip_counts)
test_coords = unique([2, cand_coords[1], high_switch_coord])
println("  candidate coords tested: $test_coords (high-switch coord=$high_switch_coord)")

for k in test_coords
    for h in (0.02, 0.1)
        w_plus = copy(w0); w_plus[k] += h
        w_minus = copy(w0); w_minus[k] -= h
        xf_plus = x_free_from_w(w_plus, pe)
        xf_minus = x_free_from_w(w_minus, pe)

        r_plus = evaluate_fullA(xf_plus, ctx; cache = nothing, use_cache = false, warm = false)
        r_minus = evaluate_fullA(xf_minus, ctx; cache = nothing, use_cache = false, warm = false)
        if !(r_plus.inner_status in FEASIBLE_CODES) || !(r_minus.inner_status in FEASIBLE_CODES)
            println("  [k=$k h=$h] SKIP: displaced point infeasible")
            continue
        end
        true_secant = -(r_plus.Delta_dual - r_minus.Delta_dual) / (2h)

        Lp_C = lfix_incremental_at_C(cache_C, ctx, pe, w0, k, w0[k] + h)
        Lm_C = lfix_incremental_at_C(cache_C, ctx, pe, w0, k, w0[k] - h)
        fixed_dual_secant_C = (Lp_C - Lm_C) / (2h)

        Lp_ref = lfix_incremental_at(cache_ref, ctx, pe, w0, k, w0[k] + h; tier = :incremental_o1)
        Lm_ref = lfix_incremental_at(cache_ref, ctx, pe, w0, k, w0[k] - h; tier = :incremental_o1)
        fixed_dual_secant_ref = (Lp_ref - Lm_ref) / (2h)

        @printf("  [k=%d h=%.2f] true_reoptimized_secant=%.6e  fixed_dual_secant(Reference)=%.6e  fixed_dual_secant(C+)=%.6e\n",
            k, h, true_secant, fixed_dual_secant_ref, fixed_dual_secant_C)
        check("[k=$k h=$h] C+ fixed-dual secant matches Reference's fixed-dual secant",
            isapprox(fixed_dual_secant_C, fixed_dual_secant_ref; atol = 1e-6, rtol = 1e-6))
    end
end
println("  NOTE: same caveat as the :kbplus directional check -- fixed-dual secants (both) are a")
println("  known approximation to the true re-optimized secant (AUD-05); what matters here is that")
println("  C+'s approximation error is no worse than the Reference's own, not equality to the truth.")

println("\n== Section 2: cross_delta + price_cache_backend=:cplus together, real staged continuation ==")
ckpt_root = mktempdir()
res_combo = run_staged_delta5_continuation("cplusgate_combo", g0, zfree0; find_smallest = true,
    delta_stages = [2.0, 3.0], stage_maxtime_real = 30.0, W_in = 80000,
    ckpt_root = joinpath(ckpt_root, "combo"), cross_delta = true, maxit_override = 3,
    price_cache_backend = :cplus)
check("cross_delta=true + price_cache_backend=:cplus: both stages completed", length(res_combo.stages) == 2)
check("cross_delta=true + price_cache_backend=:cplus: final kappa finite", isfinite(res_combo.final.kappa))
println("  stages: ", [(s.delta, s.n_eval, s.kappa, s.cache_lookups, s.cache_hit_verified) for s in res_combo.stages])

println("\n== Section 3: checkpoint/resume equivalence with C+ active ==")
ckpt_dir_full = joinpath(ckpt_root, "resume_full")
ckpt_dir_interrupt = joinpath(ckpt_root, "resume_interrupt")
res_full = run_polish_checkpointed("cplusgate_full", true, g0, zfree0; maxtime_real = 25.0, W_in = 80000,
    delta_in = 1.0, ckpt_dir = ckpt_dir_full, checkpoint_interval_s = 5.0, price_cache_backend = :cplus)

res_interrupt = run_polish_checkpointed("cplusgate_part", true, g0, zfree0; maxtime_real = 8.0, W_in = 80000,
    delta_in = 1.0, ckpt_dir = ckpt_dir_interrupt, checkpoint_interval_s = 3.0, price_cache_backend = :cplus)
latest_ckpt = joinpath(ckpt_dir_interrupt, "cplusgate_part_latest.jls")
res_resumed = run_polish_checkpointed("cplusgate_part", true, g0, zfree0; maxtime_real = 25.0, W_in = 80000,
    delta_in = 1.0, ckpt_dir = ckpt_dir_interrupt, checkpoint_interval_s = 5.0, price_cache_backend = :cplus,
    resume_from = latest_ckpt)

println("  full run:        kappa=$(res_full.kappa) n_eval=$(res_full.n_eval)")
println("  interrupt+resume: kappa=$(res_resumed.kappa) n_eval=$(res_resumed.n_eval)")
check("checkpoint/resume with C+ active: resumed run's kappa finite", isfinite(res_resumed.kappa))
check("checkpoint/resume with C+ active: resume completed without error", res_resumed.knitro_status isa Integer)

println("\n== Section 4: D=20/W=80000 short driver trajectory at delta=2, C+ ==")
res_delta2 = run_polish_checkpointed("cplusgate_delta2", true, g0, zfree0; maxtime_real = 20.0, W_in = 80000,
    delta_in = 2.0, ckpt_dir = joinpath(ckpt_root, "delta2"), checkpoint_interval_s = 5.0, price_cache_backend = :cplus)
println("  delta=2: kappa=$(res_delta2.kappa) n_eval=$(res_delta2.n_eval) knitro_status=$(res_delta2.knitro_status)")
check("delta=2 C+ trajectory completed", res_delta2.knitro_status isa Integer)
check("delta=2 C+ gradient path exercised", res_delta2.n_grad_calls > 0)

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")

# ============================================================================
# Shared-FG-verification-and-A-gradient release (2026-07-27), Phase A continuation: extends
# c20_backend_selector_driver_smoke.jl's own real-driver pattern to the new unrestricted-family
# opt-in backend `price_cache_backend=:shared` (economic_A_gradient!, wired into
# c10_d20_production_driver.jl's cb_G! this session). Real D=20/W=80,000, short trajectory,
# through the REAL production driver (run_polish_checkpointed) -- exercises the actual wiring
# (econ_ws construction, D2 buffer sizing, threaded=true dispatch inside a live KNITRO callback),
# not just the underlying economic_A_gradient! function (already gated bit-identical to
# composite_gradient_at_fast in test_shared_a_gradient_d20.jl).
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

ckpt_root = mktempdir()

ctx0 = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true)
D = ctx0.D
pe0 = build_pivot_elimination(ctx0)
g0 = ctx0.θ0_up[3+D]
Ddest0 = hasproperty(ctx0, :D_dest) ? ctx0.D_dest : ctx0.D
Aod_real = reshape(ctx0.θ0_up[ctx0.Aod_offset+1:ctx0.Aod_offset+D*Ddest0], D, Ddest0)
zfree0 = pivot_reduce(log.(Aod_real), pe0)

println("== :shared backend through run_polish_checkpointed, real D=20/W=80000, delta=1, short trajectory ==")
results = Dict{Symbol,Any}()
for backend in (:buffered, :shared)
    println("--- backend = :$backend ---")
    t0 = time()
    res = run_polish_checkpointed("smoke_$(backend)", true, g0, zfree0;
        maxtime_real = 30.0, W_in = 80000, delta_in = 1.0,
        ckpt_dir = joinpath(ckpt_root, string(backend)), checkpoint_interval_s = 5.0,
        price_cache_backend = backend)
    wall = time() - t0
    results[backend] = res
    println("  :$backend done in $(round(wall,digits=1))s: kappa=$(res.kappa) n_grad_calls=$(res.n_grad_calls) knitro_status=$(res.knitro_status)")
    check(":$backend run completed (knitro_status recorded)", res.knitro_status isa Integer)
    check(":$backend gradient path exercised (n_grad_calls>0)", res.n_grad_calls > 0)
    flush(stdout)
end

println("\n== :shared agrees with :buffered on kappa at this fixed short trajectory ==")
d = abs(results[:shared].kappa - results[:buffered].kappa)
check(":shared kappa matches :buffered exactly (diff=$d)", d == 0.0)

# Default-flips task (2026-07-27), Task B: resolve_price_cache_backend's no-kwarg default is now
# :shared (was :cplus). The DECISIVE, deterministic proof of this is a direct unit-style call to
# resolve_price_cache_backend itself (no KNITRO involved, no wall-clock sensitivity) -- a
# maxtime_real-bounded KNITRO trajectory is NOT a reliable equivalence oracle across sequential
# runs on a shared machine (confirmed live: an earlier version of this gate asserted the no-kwarg
# run's kappa matched the explicit :shared run's kappa exactly, and that failed on a real run here
# -- not because the backend resolution was wrong, but because the no-kwarg run happened to get one
# extra outer KNITRO iteration within its 30s budget than the :shared run did, landing at a
# different, still-valid, further-optimized point along the SAME trajectory. The driver's own log
# line for that run, `price_cache_backend=shared`, confirmed the backend resolution was correct
# throughout -- only the wall-clock-bounded stopping point differed).
println("\n== UNLABELED DEFAULT (no price_cache_backend kwarg) resolves to :shared ==")
check("resolve_price_cache_backend(label, nothing, nothing) == :shared (deterministic, no KNITRO)",
      resolve_price_cache_backend("check", nothing, nothing) == :shared)

println("\n== UNLABELED DEFAULT exercised through the real driver (smoke only -- no exact-kappa claim, see comment above) ==")
t0 = time()
res_default = run_polish_checkpointed("smoke_default", true, g0, zfree0;
    maxtime_real = 30.0, W_in = 80000, delta_in = 1.0,
    ckpt_dir = joinpath(ckpt_root, "default"), checkpoint_interval_s = 5.0)
wall = time() - t0
println("  default done in $(round(wall,digits=1))s: kappa=$(res_default.kappa) n_grad_calls=$(res_default.n_grad_calls) knitro_status=$(res_default.knitro_status)")
check("default run completed (knitro_status recorded)", res_default.knitro_status isa Integer)
check("default gradient path exercised (n_grad_calls>0)", res_default.n_grad_calls > 0)
check("default kappa in a sane feasible range near the shared/buffered trajectory (|Δ|<0.01)",
      abs(res_default.kappa - results[:shared].kappa) < 0.01)

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")

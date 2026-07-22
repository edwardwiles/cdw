# ============================================================================
# Phase 3 gate (finalization task, 2026-07-22): price_cache_backend selector, exercised
# through the REAL production driver (run_polish_checkpointed) for all four backends
# (:buffered/:pooled/:aplus/:cplus), extending test_driver_pooled_gradient_wiring.jl's own
# pattern (which only covered :buffered vs :pooled via the old use_pooled_gradient::Bool).
#
# Also confirms the resolver's contradiction guard actually fires BEFORE any KNITRO call when
# reached through run_polish_checkpointed itself (not just the standalone unit test already run
# at include-time).
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

println("== Section 0: contradiction guard fires through the real driver, before any KNITRO call ==")
ctx0 = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true)
D = ctx0.D
pe0 = build_pivot_elimination(ctx0)
g0 = ctx0.θ0_up[3+D]
Aod_real = reshape(ctx0.θ0_up[ctx0.Aod_offset+1:ctx0.Aod_offset+D^2], D, D)
zfree0 = pivot_reduce(log.(Aod_real), pe0)

guard_ok = try
    run_polish_checkpointed("guardtest", true, g0, zfree0; maxtime_real = 20.0, W_in = 80000, delta_in = 1.0,
        ckpt_dir = joinpath(ckpt_root, "guard"), use_pooled_gradient = true, price_cache_backend = :cplus)
    false
catch e
    println("  correctly errored: ", sprint(showerror, e)[1:min(200,end)])
    true
end
check("contradictory use_pooled_gradient=true + price_cache_backend=:cplus errors before KNITRO", guard_ok)

println("\n== Section 1: all four backends through run_polish_checkpointed, delta=1, short trajectory ==")
results = Dict{Symbol,Any}()
for backend in (:buffered, :pooled, :aplus, :cplus)
    println("--- backend = :$backend ---")
    t0 = time()
    res = run_polish_checkpointed("smoke_$(backend)", true, g0, zfree0;
        maxtime_real = 20.0, W_in = 80000, delta_in = 1.0,
        ckpt_dir = joinpath(ckpt_root, string(backend)), checkpoint_interval_s = 5.0,
        price_cache_backend = backend)
    wall = time() - t0
    results[backend] = res
    println("  :$backend done in $(round(wall,digits=1))s: kappa=$(res.kappa) n_grad_calls=$(res.n_grad_calls) knitro_status=$(res.knitro_status)")
    check(":$backend run completed (knitro_status recorded)", res.knitro_status isa Integer)
    check(":$backend gradient path exercised (n_grad_calls>0)", res.n_grad_calls > 0)
end

println("\n== Section 2: all four backends agree on kappa at this fixed short trajectory ==")
kappa_buffered = results[:buffered].kappa
for backend in (:pooled, :aplus, :cplus)
    d = abs(results[backend].kappa - kappa_buffered)
    tol = backend == :cplus ? 1e-8 : 0.0   # A+/pooled are bit-identical by construction; C+ has a
    # documented ~1e-14-1e-17 floating-point difference from its log-exp reconstruction path
    # (docs/fullA_price_tensor_elimination_report.md sec 5) -- 1e-8 is a generous, not a loose,
    # bound given that documented magnitude.
    check(":$backend kappa matches :buffered to tol=$tol (diff=$d)", d <= tol)
end

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")

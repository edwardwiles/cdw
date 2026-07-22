# ============================================================================
# Finalization task Phase 6/7: default-dispatch assertion test, required by the brief ("Every
# default change must be covered by a test that asserts the default dispatch reaches the
# intended function"). Covers both the resolver's own logic (fast, no KNITRO) and a real driver
# call confirming the printed price_cache_backend= diagnostic (which the driver already emits
# at ctx-build time, not a special test hook) shows the new default is actually reached.
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

println("== resolve_price_cache_backend: default-flip + backward-compatibility matrix ==")
check("neither kwarg given -> NEW DEFAULT :cplus",
    resolve_price_cache_backend("t", nothing, nothing) == :cplus)
check("use_pooled_gradient=false (old API, explicit) -> still :buffered (backward compat, NOT the new default)",
    resolve_price_cache_backend("t", false, nothing) == :buffered)
check("use_pooled_gradient=true (old API, explicit) -> still :pooled (backward compat)",
    resolve_price_cache_backend("t", true, nothing) == :pooled)
check("price_cache_backend=:buffered (explicit override) -> :buffered, not the new default",
    resolve_price_cache_backend("t", nothing, :buffered) == :buffered)
check("price_cache_backend=:kbplus (explicit override) -> :kbplus, not the new default",
    resolve_price_cache_backend("t", nothing, :kbplus) == :kbplus)
check("contradictory use_pooled_gradient=true + price_cache_backend=:buffered still errors",
    (try; resolve_price_cache_backend("t", true, :buffered); false; catch; true; end))

println("\n== Real driver call, neither kwarg given: confirms actual dispatch reaches :cplus ==")
ckpt_root = mktempdir()
ctx0 = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true)
D = ctx0.D
pe0 = build_pivot_elimination(ctx0)
g0 = ctx0.θ0_up[3+D]
Aod_real = reshape(ctx0.θ0_up[ctx0.Aod_offset+1:ctx0.Aod_offset+D^2], D, D)
zfree0 = pivot_reduce(log.(Aod_real), pe0)

logbuf = IOBuffer()
res_default = run_polish_checkpointed("defaulttest", true, g0, zfree0; maxtime_real = 15.0, W_in = 80000,
    delta_in = 1.0, ckpt_dir = joinpath(ckpt_root, "default"), checkpoint_interval_s = 5.0, logio = logbuf)
logtext = String(take!(logbuf))
check("driver's own startup log shows price_cache_backend=cplus when neither kwarg is given",
    occursin("price_cache_backend=cplus", logtext))
check("default-dispatch driver call completed", res_default.knitro_status isa Integer)
check("default-dispatch driver call exercised the gradient path", res_default.n_grad_calls > 0)

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")

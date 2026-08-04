# k3_preflight_smoke.jl -- TARGETED_K3_EXTENSIONS_2026-08-04 Section 7 preflight gate. Must PASS
# before any K=3 W=100k wave is launched. Constructs origin-ZC and CM+ZC (cm_meanzc) at
# K_mean=K_pair=3, prints inner/outer dimensions, verifies the free eta_nu layout for all three
# powers, and runs real D20/W=5,000 cold+warm smokes (short maxtime_real, real KNITRO, real data --
# not a mock). Does NOT run D4 (this repo's D4 derivative-test harnesses are hand-built per family
# for the K=1 shapes already in production; extending them to K=3 is out of scope for this
# preflight given the session's time budget -- flagged as a real, disclosed gap in the report, not
# silently skipped).
#
# Usage: CAMPAIGN_W=5000 julia --project=. -t <n> k3_preflight_smoke.jl
# CAMPAIGN_W must be set in the environment BEFORE this file is included (continuation_polish_run_fn*.jl
# read it into a top-level const at include time, per this repo's own "no default" rule) -- setting
# ENV["CAMPAIGN_W"] later in this same script would be a no-op for those already-evaluated consts.
haskey(ENV, "CAMPAIGN_W") || error("k3_preflight_smoke.jl requires CAMPAIGN_W set in the environment before launch (e.g. CAMPAIGN_W=5000), not set internally.")
const D4E = @__DIR__
include(joinpath(D4E, "full_chain_include.jl"))
include(joinpath(D4E, "continuation_polish_run_fn_k3.jl"))
using Statistics

lp(xs...) = (println(xs...); flush(stdout))
const RESULTS = Dict{String,Any}()
const FAILURES = String[]
check(name, cond::Bool) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name); cond)

lp("="^78); lp("K=3 PREFLIGHT: dimension + layout verification (direct construction, no KNITRO)"); lp("="^78)

ctx0 = d20_real_setup(W = 5_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx0.D
g0 = ctx0.θ0_up[ctx0.free_idx][1]
pe0 = build_pivot_elimination(ctx0)
zfree0 = pivot_reduce(reshape(log.(ctx0.θ0_up[ctx0.free_idx][2:end]), ctx0.D, ctx0.D_dest), pe0)
w_a = vcat(g0, zfree0)
lp("Economic block [gp;A_nonpivot] length: ", length(w_a), " (expect 380)")
check("economic block length == 380", length(w_a) == 380)

# --- origin_zc K=3: OriginByPowerLayout(D,3,3) -- verified by direct code read
# (cm_originzc_config.jl:43 power_target_layout default :origin_by_power) -- n_eta = K_mean*D.
layout_oz3 = OriginByPowerLayout(D, 3, 3)
n_eta_oz3 = n_eta(layout_oz3)
lp("origin_zc K_mean=K_pair=3: OriginByPowerLayout n_eta=", n_eta_oz3, " (expect K_mean*D = 3*", D, " = ", 3D, ")")
check("origin_zc K=3 n_eta == 3*D", n_eta_oz3 == 3D)
nu0_oz3 = Vector{Float64}(undef, n_eta_oz3)
for o in 1:D, k in 1:3
    nu0_oz3[target_index(layout_oz3, o, k)] = mean(ctx0.U[:, o] .^ k)
end
check("origin_zc K=3 nu0 all finite", all(isfinite, nu0_oz3))
w_originzc_k3 = vcat(w_a, log.(nu0_oz3))
lp("origin_zc K=3 full outer vector length: ", length(w_originzc_k3), " (expect 380+", 3D, "=", 380 + 3D, ")")
check("origin_zc K=3 outer vector length", length(w_originzc_k3) == 380 + 3D)

# --- cm_meanzc K=3: SharedByPowerLayout(3,3) -- verified by direct code read
# (cm_meanzc_production.jl:104/110 SharedByPowerLayout(aug.K_mean,aug.K_pair)) -- n_eta = K_mean.
layout_mz3 = SharedByPowerLayout(3, 3)
n_eta_mz3 = n_eta(layout_mz3)
lp("cm_meanzc K_mean=K_pair=3: SharedByPowerLayout n_eta=", n_eta_mz3, " (expect K_mean=3)")
check("cm_meanzc K=3 n_eta == 3", n_eta_mz3 == 3)
nu0_mz3 = [mean(mean(ctx0.U[:, o] .^ k) for o in 1:D) for k in 1:3]
check("cm_meanzc K=3 nu0 all finite", all(isfinite, nu0_mz3))
w_meanzc_k3 = vcat(w_a, log.(nu0_mz3))
lp("cm_meanzc K=3 full outer vector length: ", length(w_meanzc_k3), " (expect 383)")
check("cm_meanzc K=3 outer vector length", length(w_meanzc_k3) == 383)

lp("="^78); lp("K=3 PREFLIGHT: D20/W=5,000 COLD smoke (real KNITRO, real data, short budget)"); lp("="^78)
OUTDIR = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/targeted_k3_extensions/k3_preflight"
isdir(OUTDIR) || mkpath(OUTDIR)
ACTIVE_TARGET_DELTA[] = 1.0

cold_oz = try
    production_run_fn_k3("origin_zc", w_originzc_k3, EXPLORE_DIRECT_SR1, 30.0, joinpath(OUTDIR, "originzc_cold"); find_smallest = true)
catch e
    lp("origin_zc K=3 COLD smoke threw: ", sprint(showerror, e))
    nothing
end
check("origin_zc K=3 cold smoke ran without throwing", cold_oz !== nothing)
if cold_oz !== nothing
    check("origin_zc K=3 cold smoke found a feasible point", cold_oz.best !== nothing)
    cold_oz.best !== nothing && lp("  origin_zc K=3 cold: kappa=", cold_oz.kappa, " Delta=", cold_oz.best.Delta, " inner_status_knitro=", cold_oz.knitro_status)
end

cold_mz = try
    production_run_fn_k3("cm_meanzc", w_meanzc_k3, EXPLORE_DIRECT_SR1, 30.0, joinpath(OUTDIR, "meanzc_cold"); find_smallest = true)
catch e
    lp("cm_meanzc K=3 COLD smoke threw: ", sprint(showerror, e))
    nothing
end
check("cm_meanzc K=3 cold smoke ran without throwing", cold_mz !== nothing)
if cold_mz !== nothing
    check("cm_meanzc K=3 cold smoke found a feasible point", cold_mz.best !== nothing)
    cold_mz.best !== nothing && lp("  cm_meanzc K=3 cold: kappa=", cold_mz.kappa, " Delta=", cold_mz.best.Delta, " inner_status_knitro=", cold_mz.knitro_status)
end

lp("="^78); lp("K=3 PREFLIGHT: WARM (checkpoint/resume) smoke"); lp("="^78)
warm_oz = nothing
if cold_oz !== nothing && cold_oz.best !== nothing
    warm_oz = try
        production_run_fn_k3("origin_zc", cold_oz.best.w, EXPLORE_DIRECT_SR1, 15.0, joinpath(OUTDIR, "originzc_warm"); find_smallest = true)
    catch e
        lp("origin_zc K=3 WARM smoke threw: ", sprint(showerror, e))
        nothing
    end
end
check("origin_zc K=3 warm (resumed-from-cold-best) smoke ran", warm_oz !== nothing)

lp("="^78)
lp(isempty(FAILURES) ? "K3_PREFLIGHT = pass" : "K3_PREFLIGHT = fail_$(length(FAILURES))_checks: $(join(FAILURES, "; "))")
lp("="^78)
isempty(FAILURES) || error("k3_preflight_smoke.jl: $(length(FAILURES)) check(s) failed")

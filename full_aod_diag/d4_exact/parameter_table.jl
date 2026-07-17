# ============================================================================
# Task §6: machine-generated free-parameter table + verification tests for the
# exact full-A D=4 formulation. Run: julia --project=. full_aod_diag/d4_exact/parameter_table.jl
#
# Writes results/fullA_d4/<commit>/parameter_map.csv and prints a pass/fail
# summary for every required check (round-trip, per-coordinate perturbation
# effect, fixed-coordinate invariance/inertness, duplicate-mapping and
# unused-coordinate detection).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))

const COMMIT = "642dfe3"   # branch-off commit for this investigation; update if re-run after new commits
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup()
D = ctx.D

println("="^78)
println("D = $D full-A exact formulation: parameter table")
println("="^78)
println("l_full (raw theta length)     = ", ctx.l_full)
println("n_free (FreeParamMap)         = ", CS.n_free(ctx.m), "   (natural candidate: 1 + D^2 = ", 1 + D^2, ")")
println("n_fixed                       = ", length(ctx.fixed_idx))
println("n_A_entries_total (D^2)       = ", D^2)
println("n_A_entries_free              = ", D^2, "  (ALL free -- no A[1,d]=1 pins under gamma_d=1-for-all-d gauge)")
println("gamma_focal_prime separate?   = true (theta[", 3+D, "], K = gamma'_focal DIRECTLY)")
println("free BEFORE gravity elimination = ", CS.n_free(ctx.m))
println("free AFTER gravity elimination  = ", CS.n_free(ctx.m) - 1, "  (gravity is one exact linear equality; NOT yet implemented as elimination -- see docs/fullA_d4_code_audit.md sec 5, and gravity_elimination.jl this same directory)")

# ---------------------------------------------------------------------------
# 1. Machine-generated parameter table
# ---------------------------------------------------------------------------
rows = NamedTuple[]
for i in 1:ctx.l_full
    is_free = i in ctx.free_idx
    if i == 1
        name, loc, meaning, rep = "mu", "-", "Frechet shape parameter", "level"
    elseif i == 2
        name, loc, meaning, rep = "sigma", "-", "elasticity of substitution", "level"
    elseif 3 <= i <= 2 + D
        name, loc, meaning, rep = "gamma_theta[$(i-2)]", "-", "OLD gamma_d slot -- INERT under gammanorm gauge (moments! ignores it)", "level"
    elseif i == 3 + D
        name, loc, meaning, rep = "gamma_focal_prime", "-", "counterfactual real-income normalizer at baseIndex (direct, K==this)", "level"
    else
        lin = i - ctx.Aod_offset
        o = mod1(lin, D); d = div(lin - 1, D) + 1
        name, loc, meaning, rep = "A_od[$o,$d]", "A[$o,$d]", "bilateral efficiency shifter theta (Aod_theta), column-major", "level"
    end
    free_pos = is_free ? findfirst(==(i), ctx.free_idx) : missing
    push!(rows, (outer_index = i, name = name, A_location = loc, economic_meaning = meaning,
                  representation = rep, lower_bound = ctx.θ_lo[i], upper_bound = ctx.θ_hi[i],
                  normalization = (3 <= i <= 2+D) ? "gamma_d==1 (all d)" : (i == 3+D ? "direct gamma'_focal (no sigma/(sigma-1) power)" : "-"),
                  is_free = is_free, free_x_position = free_pos,
                  pack_fn = "CS.pack_free / FreeParamMap", unpack_fn = "CS.reconstruct_full / FreeParamMap"))
end

open(joinpath(OUTDIR, "parameter_map.csv"), "w") do io
    println(io, "outer_index,name,A_location,economic_meaning,representation,lower_bound,upper_bound,normalization,is_free,free_x_position,pack_fn,unpack_fn")
    for r in rows
        println(io, join((r.outer_index, r.name, r.A_location, "\"$(r.economic_meaning)\"", r.representation,
                           r.lower_bound, r.upper_bound, "\"$(r.normalization)\"", r.is_free,
                           something(r.free_x_position, ""), r.pack_fn, r.unpack_fn), ","))
    end
end
println("\nWrote ", joinpath(OUTDIR, "parameter_map.csv"), " (", length(rows), " rows)")

# ---------------------------------------------------------------------------
# 2. Round-trip test: full -> pack_free -> reconstruct_full -> full, EXACT
# ---------------------------------------------------------------------------
println("\n---- TEST: pack -> unpack -> pack round trip ----")
rt_ok = CS.round_trip_check(ctx.θ0_up, ctx.m)
x_free0 = CS.pack_free(ctx.θ0_up, ctx.m)
θ_back = CS.reconstruct_full(x_free0, ctx.m)
println("round_trip_check(theta0_up) = ", rt_ok, "  (", rt_ok ? "PASS" : "FAIL", ")")
println("theta0_up == reconstruct_full(pack_free(theta0_up)): ", θ_back == ctx.θ0_up)
@assert rt_ok && θ_back == ctx.θ0_up "round-trip test FAILED"

# also check a second, perturbed point (not just the calibration point)
rng_pt = ctx.θ0_up .* (1.0 .+ 0.05 .* (2 .* rand(MersenneTwister(12345), ctx.l_full) .- 1))
for i in ctx.fixed_idx  # fixed coords must still literally equal fixed_vals for pack_bounds_free to accept it
    rng_pt[i] = ctx.θ0_up[i]
end
rt_ok2 = CS.round_trip_check(rng_pt, ctx.m)
println("round_trip_check(random 5% perturbation) = ", rt_ok2, "  (", rt_ok2 ? "PASS" : "FAIL", ")")
@assert rt_ok2 "round-trip test FAILED at perturbed point"

# ---------------------------------------------------------------------------
# 3. Duplicate-mapping / unused-coordinate detection
# ---------------------------------------------------------------------------
println("\n---- TEST: duplicate-mapping / gap detection (via FreeParamMap constructor) ----")
dup_caught = false
try
    CS.FreeParamMap(ctx.l_full, vcat(ctx.free_idx, ctx.free_idx[1]), ctx.fixed_idx[2:end], ctx.fixed_vals[2:end])
    global dup_caught = false
catch e
    global dup_caught = e isa ErrorException
end
println("Deliberately-duplicated free_idx correctly rejected: ", dup_caught, "  (", dup_caught ? "PASS" : "FAIL", ")")
@assert dup_caught

gap_caught = false
try
    CS.FreeParamMap(ctx.l_full, ctx.free_idx[2:end], ctx.fixed_idx, ctx.fixed_vals)  # drops one free index -> gap
    global gap_caught = false
catch e
    global gap_caught = e isa ErrorException
end
println("Deliberately-gapped map (dropped coordinate) correctly rejected: ", gap_caught, "  (", gap_caught ? "PASS" : "FAIL", ")")
@assert gap_caught

all_idx_check = sort(vcat(ctx.free_idx, ctx.fixed_idx)) == collect(1:ctx.l_full)
println("Real map: free_idx UNION fixed_idx == 1:l_full exactly (no gap/dup): ", all_idx_check, "  (", all_idx_check ? "PASS" : "FAIL", ")")
@assert all_idx_check

# ---------------------------------------------------------------------------
# 4. Per-free-coordinate perturbation: verify EVERY claimed-free coordinate
#    actually moves the moment system (K and/or G), i.e. is not a silently
#    dead/unused outer coordinate.
# ---------------------------------------------------------------------------
println("\n---- TEST: every free coordinate has a nonzero effect on (K,G) ----")
W = size(ctx.U, 1); d_moments = ctx.nTotalMoments
K0 = zeros(W); G0 = zeros(W, d_moments)
EK_moments_gammanorm_directgp!(K0, G0, ctx.θ0_up, ctx.U, ctx.obj)
meanG0 = vec(sum(G0, dims = 1)) ./ W; meanK0 = sum(K0) / W

h_probe = 1e-4
dead_free = Int[]
for (k, i) in enumerate(ctx.free_idx)
    θp = copy(ctx.θ0_up)
    θp[i] *= (1 + h_probe)
    Kp = zeros(W); Gp = zeros(W, d_moments)
    EK_moments_gammanorm_directgp!(Kp, Gp, θp, ctx.U, ctx.obj)
    meanGp = vec(sum(Gp, dims = 1)) ./ W; meanKp = sum(Kp) / W
    moved = (abs(meanKp - meanK0) > 1e-12) || (maximum(abs.(meanGp .- meanG0)) > 1e-12)
    moved || push!(dead_free, i)
end
println("Free coordinates with ZERO measurable effect on mean(K,G) at h=$h_probe: ", dead_free)
println(isempty(dead_free) ? "PASS (every free coordinate is live)" : "FAIL (found dead/unused free coordinate(s))")
@assert isempty(dead_free)

# ---------------------------------------------------------------------------
# 5. Fixed-coordinate check: distinguish TRULY INERT (gamma_theta slots -- the
#    moment function ignores them entirely, byte-identical output) from
#    HELD-FIXED-BUT-ECONOMICALLY-ACTIVE (mu, sigma -- moments! DOES use them,
#    they are just excluded from the outer search by construction).
# ---------------------------------------------------------------------------
println("\n---- TEST: fixed-coordinate behavior (inert vs held-fixed-but-active) ----")
for i in ctx.fixed_idx
    θp = copy(ctx.θ0_up)
    θp[i] *= 1.01
    Kp = zeros(W); Gp = zeros(W, d_moments)
    EK_moments_gammanorm_directgp!(Kp, Gp, θp, ctx.U, ctx.obj)
    identical = (Kp == K0) && (Gp == G0)
    kind = (3 <= i <= 2 + D) ? "gamma_theta slot (claimed INERT)" : (i in (1,2) ? "mu/sigma (held-fixed-but-active)" : "?")
    println("  theta[$i] ($kind): perturbing outside the map -> (K,G) byte-identical to base: ", identical)
    if 3 <= i <= 2 + D
        @assert identical "gamma_theta slot $i is claimed inert by moments_gammanorm.jl but perturbing it changed (K,G) -- code comment is WRONG"
    else
        @assert !identical "mu/sigma perturbation had NO effect on (K,G) -- unexpected, these should be economically active"
    end
end
println("Fixed-coordinate behavior matches documentation: PASS")

# ---------------------------------------------------------------------------
# 6. Full ForwardDiff.gradient output length sanity check (free-only AD)
# ---------------------------------------------------------------------------
println("\n---- TEST: ForwardDiff over x_free produces exactly n_free partials ----")
f_probe = x -> sum(CS.reconstruct_full(x, ctx.m))
g_probe = ForwardDiff.gradient(f_probe, x_free0)
println("length(ForwardDiff.gradient(...)) = ", length(g_probe), " (expect ", CS.n_free(ctx.m), "): ",
        length(g_probe) == CS.n_free(ctx.m) ? "PASS" : "FAIL")
@assert length(g_probe) == CS.n_free(ctx.m)

println("\n" * "="^78)
println("ALL PARAMETER-TABLE TESTS PASSED")
println("="^78)

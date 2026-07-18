# ============================================================================
# Continuation 8, Section 8 (final D=4 candidate verification), Part 1.
#
# Runs the standing brief's full verification battery on THREE final
# candidates: the upper incumbent, the ORIGINAL registered lower incumbent,
# and the new lower_v2 (produced by c8_finalverify_lower_v2_opt.jl, Part 0).
# Every diagnostic is produced by reusing this investigation's OWN existing
# machinery (never re-derived):
#   - cold dense CC solve, Delta_primal/dual, gravity, moments, bounds:
#     evaluate_fullA (oracle.jl), exactly as candidate_registry.jl's report()
#     and phaseA_*_revalidation.jl's step1 do.
#   - optimized-value secants at several h: h_sweep.jl's h_sweep_one_direction
#     (needs three_way_derivatives.jl for solve_base_state/optimized_Delta,
#     winner_switching.jl for compute_winners/switch_stats).
#   - gravity-tangent A-only directions: perturbations of w[2:end] (z_free)
#     only, g held fixed -- by construction (gravity_elimination.jl's
#     pivot_expand always re-solves the pivot coordinate to keep R(A)=0
#     exact for ANY z_free), every such perturbation IS a tangent direction
#     to the gravity manifold, with zero extra derivation needed.
#   - exact local poll: reproduces phaseA_upper_revalidation.jl step 6 /
#     phaseA_lower_lfixcomposite_fast_revalidation.jl step 4's exact poll
#     methodology (axis directions + 20 random directions, several radii,
#     both signs, improvement defined by each candidate's own find_smallest
#     direction).
#   - external KKT in REDUCED/pivot-eliminated ("scaled") coordinates:
#     stationarity_check.jl::external_stationarity_check, unmodified, at the
#     same h-grid phaseA_upper_revalidation.jl used.
#   - external KKT in UNSCALED (original economic: gamma'_focal + full
#     log(Aod_theta), D^2+1 dims, gravity NOT eliminated) coordinates: no
#     existing script computes this (context_scaled.jl/test_context_scaled.jl
#     turned out, on inspection, to be an unrelated D/W-scaling benchmark
#     tool, not an unscaled-KKT check -- confirmed by reading both files
#     before writing this). What DOES already exist and IS reused here is
#     gravity_elimination.jl's own closed-form gravity gradient
#     (`gravity_linear_coeffs`, `gravity_offset`: g_gravity is exactly affine
#     in z=log(Aod_theta), so grad_R is the CONSTANT vector `c`) -- this
#     script assembles the general KKT formula stationarity_check.jl's own
#     header explicitly flags as not needed in reduced coordinates ("no
#     nu*grad_R term is needed here, unlike the task's general formula"),
#     using that same closed-form c as grad_R.
# ============================================================================
include(joinpath(@__DIR__, "stationarity_check.jl"))   # -> context.jl, winners.jl, oracle.jl, gravity_elimination.jl (transitively), external_stationarity_check
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "winner_switching.jl"))
include(joinpath(@__DIR__, "h_sweep.jl"))
using LinearAlgebra, Random, Printf, Dates

const COMMIT_C8 = strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT_C8, "c8_finalverify_battery")
mkpath(OUTDIR)
println(">>> c8_finalverify_battery.jl  commit=$COMMIT_C8  outdir=$OUTDIR")
flush(stdout)

ctx = d4_exact_setup(find_smallest = true)   # find_smallest does NOT affect Delta_dual/evaluate_fullA (verified: candidate_registry.jl
                                              # evaluates both upper AND lower incumbents through one shared find_smallest=true ctx and
                                              # reproduces each one's own recorded Delta_dual exactly) -- ONE ctx reused for all 3 candidates,
                                              # `find_smallest` passed explicitly per-candidate to direction-dependent checks (KKT eta sign, poll).
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

x_free_from_w(w::AbstractVector) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
function Delta_of_w(w::AbstractVector; warm::Bool = true)
    r = evaluate_fullA(x_free_from_w(w), ctx; cache = nothing, warm = warm)
    return r.Delta_dual, r
end

function write_csv_rows(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && (open(path, "w") do io; println(io, "(no rows)"); end; return)
    cols = keys(rows[1])
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((r[c] for c in cols), ","))
        end
    end
end

# ============================================================================
# The three final candidates. Upper and original-lower w vectors copied
# VERBATIM from candidate_registry.jl (w_lfixcomposite_sr1,
# w_lower_lfixcomposite_fast). lower_v2's w copied verbatim from
# c8_finalverify_lower_v2_opt.jl's printed best_feasible_tracked output
# (results/fullA_d4/0a8e68c/c8_finalverify_lower_v2_opt_20260718_173246/summary.txt).
# ============================================================================
const W_UPPER = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
const W_LOWER_ORIG = [0.9967391744173478, 0.33826763364911505, 0.2756097423805949, 0.3168080759212972, 0.28291467586724917, 1.124214996356822, 1.0586966677239436, 1.0353970705480537, 1.0651065385723435, 0.7972795304932372, 0.7498744164437179, 0.797300832305142, 0.7520482961532734, 1.464240420901755, 1.3955928233005424, 1.4031151818251653]
const W_LOWER_V2 = [0.9973649883022927, 0.4191333995096165, 0.3278704261228879, 0.34822377242086583, 0.3266848028818515, 1.1414170966875377, 1.299758470774316, 1.005176411943463, 1.0769193031619044, 0.8872003122885062, 0.7853545670122154, 0.8376830845864721, 0.7594387919364166, 1.7457822520932191, 1.4007037442409944, 1.5130549509169442]

const CANDIDATES = [
    (label = "upper_lfixcomposite_sr1_60s", w = W_UPPER, find_smallest = true, registered_kappa = 0.17245688540655113),
    (label = "lower_lfixcomposite_fast_sr1_300s_ORIGINAL", w = W_LOWER_ORIG, find_smallest = false, registered_kappa = 0.005428799948779983),
    (label = "lower_v2_c8finalverify", w = W_LOWER_V2, find_smallest = false, registered_kappa = nothing),
]

# ============================================================================
# External KKT residual in UNSCALED (original economic: gamma'_focal +
# FULL log(Aod_theta), D^2+1 dims, gravity NOT eliminated) coordinates.
# General formula per stationarity_check.jl's own header comment:
#   grad_f + eta*grad_Delta + nu*grad_R = 0,  eta >= 0,  eta*(Delta-delta)=0
# grad_R = (0, c) is CONSTANT (gravity_elimination.jl proves g_gravity is
# exactly affine in z=log(Aod_theta): grad_R = gravity_linear_coeffs(ctx)).
# ============================================================================
function external_stationarity_check_unscaled(w0::AbstractVector, ctx; find_smallest::Bool, h::Float64 = 0.01, δ::Float64 = 1.0)
    D = ctx.D; D2 = D^2
    gp0 = w0[1]
    Aod0 = exp.(pivot_expand(w0[2:end], pe))     # D x D, the candidate's actual Aod_theta level
    z0 = vec(log.(Aod0))                          # D^2, UNREDUCED log-A (gravity not eliminated here)
    x0 = vcat(gp0, z0)                            # D^2+1
    nn = D2 + 1
    function Delta_of_x(x)
        gp_ = x[1]; z = x[2:end]
        xf = vcat(gp_, exp.(z))
        r = evaluate_fullA(xf, ctx; cache = nothing, warm = true)
        return r.Delta_dual
    end
    Δ0 = Delta_of_x(x0)
    grad_f = zeros(nn); grad_f[1] = find_smallest ? 1.0 : -1.0
    grad_Delta = zeros(nn)
    n_nonfinite = 0
    for i in 1:nn
        xp = copy(x0); xp[i] += h; xm = copy(x0); xm[i] -= h
        Δp = Delta_of_x(xp); Δm = Delta_of_x(xm)
        if isfinite(Δp) && isfinite(Δm)
            grad_Delta[i] = (Δp - Δm) / (2h)
        else
            n_nonfinite += 1
        end
    end
    c = vec(gravity_linear_coeffs(ctx))
    grad_R = vcat(0.0, c)
    # least-squares solve for (eta, nu): [grad_Delta grad_R] * [eta; nu] = -grad_f
    Amat = hcat(grad_Delta, grad_R)
    sol = Amat \ (-grad_f)
    eta, nu = sol[1], sol[2]
    residual_vec = grad_f .+ eta .* grad_Delta .+ nu .* grad_R
    residual_norm = norm(residual_vec)
    gravity_resid_at_x0 = dot(c, z0) + gravity_offset(ctx)
    return (Delta = Δ0, Delta_minus_delta = Δ0 - δ, eta = eta, eta_nonneg = eta >= -1e-8, nu = nu,
            residual_norm = residual_norm, residual_relative = residual_norm / max(norm(grad_f), 1e-12),
            gravity_resid_at_x0 = gravity_resid_at_x0, n_nonfinite_probes = n_nonfinite)
end

# ============================================================================
# Per-candidate battery
# ============================================================================
const H_GRID_KKT = [0.02, 0.01, 0.005, 0.0025, 0.001]
const POLL_RADII = [0.0001, 0.001, 0.005, 0.02]
const N_RANDOM_POLL_DIRS = 20
const TANGENT_H = [0.01, 0.05]
const TANGENT_COORDS = [2, 3, 4]   # first 3 z_free coordinates of w (w[1] is g, held fixed -- gravity-tangent by construction)

summary_rows = NamedTuple[]
secant_rows = NamedTuple[]
tangent_rows = NamedTuple[]
poll_rows = NamedTuple[]
kkt_scaled_hgrid_rows = NamedTuple[]
moment_resid_rows = NamedTuple[]

for cand in CANDIDATES
    label, w0, find_smallest = cand.label, cand.w, cand.find_smallest
    println("\n" * "="^78); println("CANDIDATE: $label  (find_smallest=$find_smallest)"); println("="^78); flush(stdout)
    xf0 = x_free_from_w(w0)

    # ---- 1. fresh dense COLD CC solve: Delta_primal/dual, gravity, moments ----
    println("-- fresh dense cold CC solve --")
    r_cold = evaluate_fullA(xf0, ctx; cache = nothing, warm = false)
    r_warm = evaluate_fullA(xf0, ctx; cache = nothing, warm = true)   # for comparison
    κ = 1 - r_cold.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
    @printf("  kappa=%.10f  Delta_dual=%.10f  Delta_primal=%.10f  primal_dual_gap=%.3e  Delta-delta=%.4e\n",
        κ, r_cold.Delta_dual, r_cold.Delta_primal, r_cold.primal_dual_gap, r_cold.Delta_minus_delta)
    @printf("  gravity_value=%.3e  gravity_raw=%.3e  max_abs_moment_resid=%.3e  max_abs_moment_kkt_resid=%.3e  mean_m_resid=%.3e  inner_status=%d\n",
        r_cold.gravity_value, r_cold.gravity_raw, r_cold.max_abs_moment_resid, r_cold.max_abs_moment_kkt_resid, r_cold.mean_m_resid, r_cold.inner_status)

    for (j, mr) in enumerate(r_cold.moment_resid)
        push!(moment_resid_rows, (label = label, moment_idx = j, moment_resid = mr))
    end

    # ---- 2. bounds check (original economic bound: gamma'_focal; numerical-safeguard box: z_free) ----
    margin_lo = w0[1] - ctx.bounds.γp_lo
    margin_hi = ctx.bounds.γp_hi - w0[1]
    zfree = w0[2:end]
    maxabs_zfree = maximum(abs.(zfree))
    bounds_ok = margin_lo >= 0 && margin_hi >= 0 && maxabs_zfree < 8.0
    @printf("  bounds: gp margin_lo=%.6f margin_hi=%.6f (genuine economic bound); max|z_free|=%.4f of safeguard [-8,8] (numerical, not economic) -- bounds_ok=%s\n",
        margin_lo, margin_hi, maxabs_zfree, bounds_ok)
    # Bound-respecting "safe" h for the KKT probes below: a fixed h=0.01 central-difference probe on
    # coordinate 1 (gamma'_focal) steps OUTSIDE the theoretical box bound whenever the candidate's own
    # margin to that bound is < 0.01 -- true for BOTH lower-direction candidates here (margin_hi ~
    # 0.0026-0.0033, since both sit close to the g~1 boundary), NOT for the upper candidate (margin_hi
    # ~0.107). An out-of-bounds probe is a bandwidth-selection artifact, not evidence of non-stationarity
    # -- h_safe keeps every probe in-bounds with a safety factor, so the "primary" KKT read used for
    # classification below is always a genuinely computable one. The raw h=0.01 result is ALSO still
    # computed/reported for direct comparability with the upper candidate's own historical h=0.01 checks.
    h_safe = min(0.01, 0.4 * min(margin_lo, margin_hi))
    @printf("  h_safe (bound-respecting KKT bandwidth) = %.6f  (0.01 requested; clipped for margin proximity: %s)\n",
        h_safe, h_safe < 0.01)

    # ---- 3. external KKT checks, BOTH coordinate systems -- MUST run before any of the poll/secant/
    # tangent sections below, which make hundreds of evaluate_fullA calls at UNRELATED, often-distant
    # points and would otherwise leave ctx.obj.arg1 (the shared inner-dual warm state) corrupted for
    # stationarity_check.jl::external_stationarity_check's own internal Delta_of_w (which hardcodes
    # warm=true, unlike this workstream's own c8_gammabranch_core.jl variant -- see that file's header
    # for the exact same hazard, caught there first). Cold-prime immediately before, to reset the warm
    # state to something anchored at w0 regardless of what the PREVIOUS candidate's loop iteration left
    # behind. Caught for real in this script's first run: without this reordering, eta printed as an
    # exact -0.0 (grad_Delta identically the zero vector) for both lower-direction candidates.
    # ============================================================================
    println("-- external KKT, reduced (pivot-eliminated, gravity dropped) coordinates --")
    evaluate_fullA(xf0, ctx; cache = nothing, warm = false)   # cold-prime ctx.obj.arg1 at w0 before any warm=true internal FD probes
    scaled_h01 = external_stationarity_check(w0, ctx, pe; find_smallest = find_smallest, h = 0.01, w_lo = w_lo, w_hi = w_hi)
    scaled_verified_h01 = scaled_h01.eta_nonneg && scaled_h01.residual_relative < 0.05 &&
        abs(scaled_h01.complementary_slackness) < 0.05 && scaled_h01.n_active_bounds == 0 && scaled_h01.n_nonfinite_probes == 0
    @printf("  h=0.01: eta=%.6f eta_nonneg=%s resid_rel=%.4f comp_slack=%.4e n_active_bounds=%d nonfinite=%d -> H_BANDWIDTH_KKT(h=0.01)=%s\n",
        scaled_h01.eta, scaled_h01.eta_nonneg, scaled_h01.residual_relative, scaled_h01.complementary_slackness,
        scaled_h01.n_active_bounds, scaled_h01.n_nonfinite_probes, scaled_verified_h01)

    scaled_hgrid_all_pass = true
    for h in H_GRID_KKT
        evaluate_fullA(xf0, ctx; cache = nothing, warm = false)   # re-prime before EACH h (previous h's internal probes end far from w0)
        res = external_stationarity_check(w0, ctx, pe; find_smallest = find_smallest, h = h, w_lo = w_lo, w_hi = w_hi)
        pass_h = res.eta_nonneg && res.residual_relative < 0.10 && res.n_nonfinite_probes == 0
        scaled_hgrid_all_pass &= pass_h
        push!(kkt_scaled_hgrid_rows, (label = label, h = h, eta = res.eta, eta_nonneg = res.eta_nonneg,
              residual_relative = res.residual_relative, complementary_slackness = res.complementary_slackness,
              n_nonfinite_probes = res.n_nonfinite_probes, pass_h = pass_h))
        @printf("  h=%.5f: eta=%.6f resid_rel=%.4f pass=%s\n", h, res.eta, res.residual_relative, pass_h)
    end

    println("-- external KKT, UNSCALED (gamma'_focal + full log(Aod_theta), gravity multiplier nu explicit) coordinates --")
    evaluate_fullA(xf0, ctx; cache = nothing, warm = false)   # cold-prime again (unscaled function also has an internal warm=true probe loop)
    unscaled_h01 = external_stationarity_check_unscaled(w0, ctx; find_smallest = find_smallest, h = 0.01, δ = ctx.δ)
    unscaled_verified_h01 = unscaled_h01.eta_nonneg && unscaled_h01.residual_relative < 0.05 && unscaled_h01.n_nonfinite_probes == 0
    @printf("  h=0.01: eta=%.6f nu=%.6f eta_nonneg=%s resid_rel=%.4f gravity_resid_at_x0=%.3e nonfinite=%d -> UNSCALED_H_BANDWIDTH_KKT(h=0.01)=%s\n",
        unscaled_h01.eta, unscaled_h01.nu, unscaled_h01.eta_nonneg, unscaled_h01.residual_relative,
        unscaled_h01.gravity_resid_at_x0, unscaled_h01.n_nonfinite_probes, unscaled_verified_h01)

    # ---- primary (bound-respecting) KKT read at h_safe, used for classification below ----
    println("-- KKT at h_safe (bound-respecting, PRIMARY read used for classification) --")
    evaluate_fullA(xf0, ctx; cache = nothing, warm = false)
    scaled_hsafe = external_stationarity_check(w0, ctx, pe; find_smallest = find_smallest, h = h_safe, w_lo = w_lo, w_hi = w_hi)
    scaled_verified_hsafe = scaled_hsafe.eta_nonneg && scaled_hsafe.residual_relative < 0.05 &&
        abs(scaled_hsafe.complementary_slackness) < 0.05 && scaled_hsafe.n_active_bounds == 0 && scaled_hsafe.n_nonfinite_probes == 0
    evaluate_fullA(xf0, ctx; cache = nothing, warm = false)
    unscaled_hsafe = external_stationarity_check_unscaled(w0, ctx; find_smallest = find_smallest, h = h_safe, δ = ctx.δ)
    unscaled_verified_hsafe = unscaled_hsafe.eta_nonneg && unscaled_hsafe.residual_relative < 0.05 && unscaled_hsafe.n_nonfinite_probes == 0
    @printf("  h_safe=%.6f: scaled eta=%.6f resid_rel=%.4f verified=%s | unscaled eta=%.6f nu=%.6f resid_rel=%.4f verified=%s\n",
        h_safe, scaled_hsafe.eta, scaled_hsafe.residual_relative, scaled_verified_hsafe,
        unscaled_hsafe.eta, unscaled_hsafe.nu, unscaled_hsafe.residual_relative, unscaled_verified_hsafe)

    # ---- 4. optimized-value central/one-sided secants at several h (h_sweep.jl, reused verbatim) ----
    println("-- secant sweep (h_sweep.jl::h_sweep_one_direction, seed=4242 matching test_h_sweep.jl) --")
    base = solve_base_state(xf0, ctx)
    rng_secant = MersenneTwister(4242)
    v_secant = randn(rng_secant, length(xf0)); v_secant ./= norm(v_secant)
    secant_result = h_sweep_one_direction(xf0, v_secant, ctx, base)
    for row in secant_result
        push!(secant_rows, (label = label, h = row.h, D_left = row.D_left, D_right = row.D_right, D_central = row.D_central,
                             Q_central = row.Q_central, L_central = row.L_central,
                             n_switches_plus = row.n_switches_plus, n_switches_minus = row.n_switches_minus))
        @printf("  h=%.5f  D_central=%.6f  D_left=%.6f  D_right=%.6f  switches(+/-)=%d/%d\n",
            row.h, row.D_central, row.D_left, row.D_right, row.n_switches_plus, row.n_switches_minus)
    end
    d_central_h02 = secant_result[findfirst(r -> r.h == 0.2, secant_result)].D_central
    d_central_h00625 = secant_result[findfirst(r -> r.h == 0.00625, secant_result)].D_central
    secant_stability = abs(d_central_h02 - d_central_h00625) / max(abs(d_central_h00625), 1e-12)
    @printf("  secant stability (|D_central(h=0.2)-D_central(h=0.00625)| / |D_central(h=0.00625)|) = %.4f\n", secant_stability)

    # ---- 5. gravity-tangent A-only directions (perturb z_free only, g fixed) ----
    println("-- gravity-tangent A-only directions --")
    Δ0, r0 = Delta_of_w(w0)
    for i in TANGENT_COORDS, h in TANGENT_H
        wp = copy(w0); wp[i] += h; wm = copy(w0); wm[i] -= h
        Δp, rp = Delta_of_w(wp); Δm, rm = Delta_of_w(wm)
        push!(tangent_rows, (label = label, zfree_coord = i, h = h,
              Delta_central = (Δp - Δm) / (2h), Delta_right = (Δp - Δ0) / h, Delta_left = (Δ0 - Δm) / h,
              gravity_p = rp.gravity_value, gravity_m = rm.gravity_value, gravity_0 = r0.gravity_value))
        @printf("  coord=%d h=%.3f  Delta_central=%.6f  gravity(p/m/0)=%.2e/%.2e/%.2e (should all be ~0 -- exact by pivot construction)\n",
            i, h, (Δp - Δm) / (2h), rp.gravity_value, rm.gravity_value, r0.gravity_value)
    end

    # ---- 6. exact local poll (phaseA_*_revalidation.jl's exact methodology) ----
    println("-- exact local poll --")
    rng_poll = MersenneTwister(20260718)
    random_dirs = [normalize(randn(rng_poll, D2)) for _ in 1:N_RANDOM_POLL_DIRS]
    poll_dirs = vcat([begin; e = zeros(D2); e[i] = 1.0; e; end for i in 1:D2], random_dirs)
    n_improvements = 0
    n_probes = 0
    for radius in POLL_RADII, (di, d) in enumerate(poll_dirs), sgn in (1.0, -1.0)
        w_try = w0 .+ sgn .* radius .* d
        Δ, r_try = Delta_of_w(w_try)
        feasible = isfinite(Δ) && Δ <= ctx.δ + 1e-6
        improved = feasible && (find_smallest ? w_try[1] < w0[1] - 1e-12 : w_try[1] > w0[1] + 1e-12)
        n_probes += 1
        improved && (n_improvements += 1)
        push!(poll_rows, (label = label, radius = radius, dir_idx = di, sign = sgn, gp = w_try[1], Delta = Δ,
                           feasible = feasible, improved = improved))
    end
    println("  poll: $n_probes probes, $n_improvements exact-feasible improvements found")

    # ---- classification (uses the h_safe, bound-respecting KKT read as primary -- see step 2/3 note:
    # a literal h=0.01 probe on coordinate 1 is out-of-bounds for both lower-direction candidates, an
    # artifact of their proximity to the g~1 economic bound, not evidence of non-stationarity) ----
    exact_feasible = r_cold.inner_status in (0, -100, -101, -103) && r_cold.Delta_minus_delta <= 1e-4 &&
        r_cold.max_abs_moment_kkt_resid < 1e-4 && abs(r_cold.gravity_value) < 1e-6
    robust_local = scaled_verified_hsafe && scaled_hgrid_all_pass && n_improvements == 0 && unscaled_verified_hsafe
    bandwidth_kkt_only = scaled_verified_hsafe && !(scaled_hgrid_all_pass && n_improvements == 0 && unscaled_verified_hsafe)

    classification = if exact_feasible && robust_local
        "exact-feasible incumbent"
    elseif exact_feasible && bandwidth_kkt_only
        "bandwidth-KKT candidate (h_safe=$(round(h_safe,digits=5)) only)"
    elseif exact_feasible && n_improvements == 0
        "robust local candidate"
    elseif exact_feasible
        "robust local candidate (poll found nearby improvement -- see poll csv)"
    else
        "best-feasible stalled point"
    end
    println("  CLASSIFICATION: ", classification)

    push!(summary_rows, (label = label, find_smallest = find_smallest, gamma_focal_prime = r_cold.gamma_focal_prime,
        kappa = κ, registered_kappa = something(cand.registered_kappa, NaN),
        Delta_dual = r_cold.Delta_dual, Delta_primal = r_cold.Delta_primal, primal_dual_gap = r_cold.primal_dual_gap,
        Delta_minus_delta = r_cold.Delta_minus_delta, gravity_value = r_cold.gravity_value, gravity_raw = r_cold.gravity_raw,
        max_abs_moment_resid = r_cold.max_abs_moment_resid, max_abs_moment_kkt_resid = r_cold.max_abs_moment_kkt_resid,
        mean_m_resid = r_cold.mean_m_resid, inner_status_cold = r_cold.inner_status, inner_status_warm = r_warm.inner_status,
        Delta_dual_cold = r_cold.Delta_dual, Delta_dual_warm = r_warm.Delta_dual,
        margin_lo_gp_bound = margin_lo, margin_hi_gp_bound = margin_hi, max_abs_zfree = maxabs_zfree, bounds_ok = bounds_ok,
        secant_stability_h02_vs_h00625 = secant_stability,
        n_poll_probes = n_probes, n_poll_improvements = n_improvements,
        h_safe = h_safe,
        scaled_eta_h01 = scaled_h01.eta, scaled_resid_rel_h01 = scaled_h01.residual_relative,
        scaled_verified_h01 = scaled_verified_h01, scaled_hgrid_all_pass = scaled_hgrid_all_pass,
        unscaled_eta_h01 = unscaled_h01.eta, unscaled_nu_h01 = unscaled_h01.nu,
        unscaled_resid_rel_h01 = unscaled_h01.residual_relative, unscaled_verified_h01 = unscaled_verified_h01,
        unscaled_gravity_resid_at_x0 = unscaled_h01.gravity_resid_at_x0,
        scaled_eta_hsafe = scaled_hsafe.eta, scaled_resid_rel_hsafe = scaled_hsafe.residual_relative,
        scaled_verified_hsafe = scaled_verified_hsafe,
        unscaled_eta_hsafe = unscaled_hsafe.eta, unscaled_nu_hsafe = unscaled_hsafe.nu,
        unscaled_resid_rel_hsafe = unscaled_hsafe.residual_relative, unscaled_verified_hsafe = unscaled_verified_hsafe,
        exact_feasible = exact_feasible, classification = classification))
    flush(stdout)
end

write_csv_rows(joinpath(OUTDIR, "summary.csv"), summary_rows)
write_csv_rows(joinpath(OUTDIR, "secant_sweep.csv"), secant_rows)
write_csv_rows(joinpath(OUTDIR, "gravity_tangent_directions.csv"), tangent_rows)
write_csv_rows(joinpath(OUTDIR, "poll.csv"), poll_rows)
write_csv_rows(joinpath(OUTDIR, "kkt_scaled_hgrid.csv"), kkt_scaled_hgrid_rows)
write_csv_rows(joinpath(OUTDIR, "moment_resid.csv"), moment_resid_rows)

println("\n" * "="^78); println("FINAL SUMMARY"); println("="^78)
for r in summary_rows
    @printf("%-45s  kappa=%.10f  Delta-delta=%+.3e  classification=%s\n", r.label, r.kappa, r.Delta_minus_delta, r.classification)
end
println("\nWrote all battery artifacts to ", OUTDIR)

# 2026-08-05: paired-basis preconditioning pilot for the two-family common-marginals restriction
# (eq.35 CDF + eq.36 truncated z^(sigma-1) power). Tests whether a FIXED, invertible, per-
# coordinate-pair residualization/RMS-scaling transform resolves the D20 real-data non-convergence
# documented in MASTER.md sections 21-24 (nStatus=-400 at every tested W/L, near-collinear CDF/POW
# columns, cond(CM) up to ~8,730 / cond(full G) up to ~85,343, max paired-column correlation
# ~0.9997). See docs/audits/cm-paired-basis-preconditioning-2026-08-05/MASTER.md for the task brief.
#
# Reuses EXISTING generic machinery only: precalc_common_marginals_cdf (raw feature construction,
# unchanged), wrap_moments_with_cm (the dense-splice moments! closure, already 100% generic on
# CM's column content -- see its own docstring), evaluate_fullA (dense Architecture A KNITRO inner
# solve, already generic on obj.d). NO new Hessian/gradient/FG/verifier code is written here -- the
# four "arms" below differ ONLY in which W x 2*(D-1)*L matrix is handed to wrap_moments_with_cm.
#
# Usage: julia test_cm_paired_basis_precond_2026-08-05.jl <scale:d4|d20> <W> <L> <outdir>
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics, DelimitedFiles
lp(xs...) = (println(xs...); flush(stdout))

scale  = length(ARGS) >= 1 ? ARGS[1] : "d4"
Wsel   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : (scale == "d4" ? 0 : 20_000)
L      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
outdir = length(ARGS) >= 4 ? ARGS[4] : "/bbkinghome/edav/repo_scratch/cm-paired-basis-preconditioning-2026-08-05"
mkpath(outdir)

lp("="^100)
lp("SCALE=$scale  W=$Wsel  L=$L  outdir=$outdir")
lp("="^100)

t_ctx0 = time()
if scale == "d4"
    global ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
elseif scale == "d20"
    global ctx = d20_real_setup(W = Wsel, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
else
    error("unknown scale=$scale (expected d4 or d20)")
end
lp("Context built in $(round(time()-t_ctx0,digits=2))s. D=", ctx.D, " sigma=", ctx.σ, " muHat=", ctx.μHat,
   " W(draws)=", size(ctx.U,1))

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
refIndex1 = ctx.γ.refIndex1
probs_ = collect(range(1 / L, (L - 1) / L, length = L))
W_draws = size(ctx.U, 1)

# ---- raw two-family CM feature matrix, EXISTING function, unmodified ----
CM_raw, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L;
    include_truncated_moment = true, σHat = ctx.σ, μHat = ctx.μHat, contrasts = :anchored, probs = probs_)
nO = length(origins)
ncm_cdf = nO * L
@assert size(CM_raw, 2) == 2 * ncm_cdf
c = CM_raw[:, 1:ncm_cdf]
p = CM_raw[:, ncm_cdf+1:end]

# =====================================================================================
# Section 3: paired-coordinate alignment map
# =====================================================================================
coord_csv = joinpath(outdir, "CM_PAIRED_COORDINATE_MAP_$(scale)_W$(Wsel)_L$(L).csv")
open(coord_csv, "w") do io
    println(io, "cdf_col,pow_col,quantile_index_l,quantile_prob,z_cutoff,origin_position,origin_index,ref_index1")
    for l in 1:L, (oi, o) in enumerate(origins)
        cdf_col = (l - 1) * nO + oi
        pow_col = ncm_cdf + cdf_col
        println(io, "$cdf_col,$pow_col,$l,$(probs_[l]),$(z[l]),$oi,$o,$refIndex1")
    end
end
lp("Wrote paired-coordinate map: $coord_csv ($(ncm_cdf) pairs)")

# =====================================================================================
# Section 4: pairwise transform (beta_j, s_c,j, s_r,j) computed ONCE under the raw draws
# (the reference distribution F* = the fixed, un-reweighted W draws already fixed by ctx.U --
# same convention as section 21's own "raw F* Monte Carlo draws" diagnostic). RMS, not centered
# sample std (task explicitly requires RMS and explicitly forbids centering).
# =====================================================================================
β   = zeros(ncm_cdf)
s_c = zeros(ncm_cdf)
s_p = zeros(ncm_cdf)
s_r = zeros(ncm_cdf)
paired_corr = zeros(ncm_cdf)
r = similar(p)
for j in 1:ncm_cdf
    cj = @view c[:, j]
    pj = @view p[:, j]
    denom = dot(cj, cj)
    β[j] = denom > 0 ? dot(cj, pj) / denom : 0.0
    @views r[:, j] .= pj .- β[j] .* cj
    s_c[j] = sqrt(dot(cj, cj) / W_draws)
    s_p[j] = sqrt(dot(pj, pj) / W_draws)
    s_r[j] = sqrt(dot(@view(r[:, j]), @view(r[:, j])) / W_draws)
    paired_corr[j] = (s_c[j] > 0 && s_p[j] > 0) ? dot(cj, pj) / (W_draws * s_c[j] * s_p[j]) : NaN
end

DEGEN_TOL = 1e-8
degenerate = falses(ncm_cdf)
for j in 1:ncm_cdf
    scale_ref = max(s_c[j], s_p[j], 1.0)
    if s_r[j] < DEGEN_TOL * scale_ref
        degenerate[j] = true
    end
end
n_degen = count(degenerate)
lp("Degenerate residual coordinates (s_r < $(DEGEN_TOL)*max(s_c,s_p,1)): $n_degen / $ncm_cdf")
if n_degen > 0
    lp("  degenerate j indices: ", findall(degenerate))
    lp("  STOP-AND-CLASSIFY REQUIRED (task sec 4) -- these are exact/near-exact redundancies, not a floor-and-continue case.")
end

diag_csv = joinpath(outdir, "CM_TRANSFORM_DIAGNOSTICS_$(scale)_W$(Wsel)_L$(L).csv")
open(diag_csv, "w") do io
    println(io, "j,beta_j,s_c_j,s_p_j,s_r_j,paired_corr,residual_to_original_rms_ratio,degenerate")
    for j in 1:ncm_cdf
        ratio = s_p[j] > 0 ? s_r[j] / s_p[j] : NaN
        println(io, "$j,$(β[j]),$(s_c[j]),$(s_p[j]),$(s_r[j]),$(paired_corr[j]),$ratio,$(degenerate[j])")
    end
end
lp("Wrote beta/scale/residual diagnostics: $diag_csv")
lp("Max paired |corr| = ", maximum(abs.(filter(!isnan, paired_corr))))
lp("Min s_r_j = ", minimum(s_r), "  Max s_r_j = ", maximum(s_r))

# =====================================================================================
# Section 5: four matched arms -- build the transformed CM matrix (pure linear algebra on the
# already-computed raw features), splice via wrap_moments_with_cm (existing, generic), solve via
# evaluate_fullA (existing, generic dense Architecture A inner KNITRO solve).
# =====================================================================================
function build_arm_CM(arm::Symbol)
    CMt = similar(CM_raw)
    if arm == :A            # raw
        CMt .= CM_raw
    elseif arm == :B        # diagonal RMS scaling only
        for j in 1:ncm_cdf
            @views CMt[:, j]          .= c[:, j] ./ s_c[j]
            @views CMt[:, ncm_cdf+j]  .= p[:, j] ./ s_p[j]
        end
    elseif arm == :C        # pairwise residualization, no scaling
        CMt[:, 1:ncm_cdf]     .= c
        CMt[:, ncm_cdf+1:end] .= r
    elseif arm == :D        # residualization + RMS scaling
        for j in 1:ncm_cdf
            @views CMt[:, j]          .= c[:, j] ./ s_c[j]
            @views CMt[:, ncm_cdf+j]  .= r[:, j] ./ s_r[j]
        end
    else
        error("unknown arm $arm")
    end
    return CMt
end

# T_j (2x2), maps [c_j;p_j] -> [ctilde_j; gtilde_j] for a given arm; used only for the documented
# dual map lambda_raw = T_j' * lambda_tilde -- NOT used to build the CM matrix (that's done directly
# above; this is the algebraic record required by task sec 6).
function T_block(arm::Symbol, j::Int)
    if arm == :A
        return Matrix{Float64}(I, 2, 2)
    elseif arm == :B
        return [1/s_c[j] 0.0; 0.0 1/s_p[j]]
    elseif arm == :C
        return [1.0 0.0; -β[j] 1.0]
    elseif arm == :D
        return [1/s_c[j] 0.0; -β[j]/s_r[j] 1/s_r[j]]
    end
end
Tinv_block(arm::Symbol, j::Int) = inv(T_block(arm, j))

function build_dense_obj(ctxb, CMt::Matrix{Float64})
    obj0 = ctxb.obj
    ncore = obj0.d
    moments_cm! = wrap_moments_with_cm(obj0.moments!, ncore, CMt)
    d_new = ncore + size(CMt, 2)
    outer_constr_index_new = obj0.outer_constr_index + size(CMt, 2)
    return CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest, γ = obj0.γ,
        (moments!) = moments_cm!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
end

results = Dict{Symbol,NamedTuple}()
arm_CM = Dict{Symbol,Matrix{Float64}}()
arm_obj = Dict{Symbol,Any}()   # section 6 reuses the SAME obj instance each arm solved with --
# evaluate_fullA/inner_loop_internal leave solve-relevant mutable state on `obj` (obj.x, threshold
# caches, ...) that a freshly-rebuilt object never gets primed with; rebuilding from scratch for
# the post-solve recompute produced a bogus Delta_dual/KKT readout (confirmed live: a fresh-object
# recompute gave Delta_dual=-1.0e-5 against the real converged 0.0045, while the correctly-scoped
# dual-map identity check on the SAME data passed at 1e-15 -- i.e. the bug was in this diagnostic's
# own object lifecycle, not in the transform or the solve).

for arm in (:A, :B, :C, :D)
    lp("="^100); lp("ARM $arm"); lp("="^100)
    CMt = build_arm_CM(arm)
    arm_CM[arm] = CMt
    obj_arm = build_dense_obj(ctx, CMt)
    arm_obj[arm] = obj_arm
    ctx_cm = merge(ctx, (obj = obj_arm,))

    # conditioning: CM-only block, and core+CM full G (same method every arm, matches
    # test_cm_moment_rank_2026-08-05.jl's own methodology)
    sv_cm = svdvals(CMt)
    cond_cm = sv_cm[1] / sv_cm[end]
    K0 = zeros(W_draws)
    obj_arm.moments!(K0, CS.select_G_from_H(obj_arm, obj_arm.H), θ_full_calib, obj_arm.U, obj_arm)
    G_full = Matrix(obj_arm.H[:, 3:obj_arm.outer_constr_index+1])
    sv_full = svdvals(G_full)
    cond_full = sv_full[1] / sv_full[end]

    iters0 = try CS.INNER_ITERS_TOTAL[] catch; missing end
    t0 = time()
    r_res = evaluate_fullA(x_free_calib, ctx_cm; use_cache = false, warm = false)
    telapsed = time() - t0
    iters1 = try CS.INNER_ITERS_TOTAL[] catch; missing end
    iters = (iters0 === missing || iters1 === missing) ? missing : iters1 - iters0

    lp("  status=$(r_res.inner_status)  Delta_dual=$(r_res.Delta_dual)  t=$(round(telapsed,digits=3))s  iters=$iters")
    lp("  cond(CM)=$cond_cm  cond(fullG)=$cond_full  sigma_min(CM)=$(sv_cm[end])  sigma_max(CM)=$(sv_cm[1])")
    lp("  m_mean=$(r_res.m_mean)  m_min=$(r_res.m_min)  m_max=$(r_res.m_max)  max_abs_moment_kkt_resid=$(r_res.max_abs_moment_kkt_resid)")

    results[arm] = (status = r_res.inner_status, Delta_dual = r_res.Delta_dual, telapsed = telapsed,
                     iters = iters, cond_cm = cond_cm, cond_full = cond_full,
                     sigma_min_cm = sv_cm[end], sigma_max_cm = sv_cm[1],
                     m_mean = r_res.m_mean, m_min = r_res.m_min, m_max = r_res.m_max,
                     max_abs_moment_kkt_resid = r_res.max_abs_moment_kkt_resid,
                     zeta = r_res.zeta, lambda = r_res.lambda, θ_full = r_res.θ_full,
                     x_free = r_res.x_free)
end

# =====================================================================================
# Section 6: exact mathematical equivalence -- T/T^-1 documentation + raw-moment verification at
# each arm's solved point (using ONLY the same-θ recompute pattern evaluate_fullA already uses
# internally, no new formula).
# =====================================================================================
lp("="^100); lp("Section 6: raw-vs-transformed equivalence"); lp("="^100)
obj_raw = arm_obj[:A]   # Arm A solved with CM_raw unmodified -- same object, already-primed state
equiv_results = Dict{Symbol,NamedTuple}()
for arm in (:B, :C, :D)
    res = results[arm]
    if !(res.status in (0, -100, -101, -102, -103))
        lp("ARM $arm: inner solve did not reach a feasible/converged status ($(res.status)) -- skipping raw-moment equivalence check (no valid dual point to map).")
        continue
    end
    obj_arm = arm_obj[arm]   # REUSE the object this arm actually solved with -- rebuilding a fresh
    # object here previously gave a bogus recompute (see note above arm_obj's definition).
    Karm = zeros(W_draws)
    Garm = zeros(W_draws, obj_arm.d)
    obj_arm.moments!(Karm, Garm, res.θ_full, ctx.U, obj_arm)
    inner_x = vcat(res.zeta, res.lambda)
    ncon = obj_arm.d - obj_arm.outer_constr_index + 2
    cbuf = zeros(ncon)
    obj_arm(inner_x, constr = @view(cbuf[1:ncon]))
    m_weights_arm = copy(obj_arm.arg1)
    Delta_dual_recompute = cbuf[1] / 1e10

    # raw moments E_F[c_j], E_F[p_j] under the SAME recovered reweighting m_weights_arm
    Kraw = zeros(W_draws)
    Graw = zeros(W_draws, obj_raw.d)
    obj_raw.moments!(Kraw, Graw, res.θ_full, ctx.U, obj_raw)
    ncore = ctx.obj.d
    raw_kkt(j) = abs(sum(m_weights_arm .* @view(Graw[:, ncore-1+j])) / W_draws)
    max_cdf_kkt_raw = maximum(raw_kkt(j) for j in 1:ncm_cdf)
    max_pow_kkt_raw = maximum(raw_kkt(j) for j in ncm_cdf+1:2*ncm_cdf)

    # dual-map identity check: lambda_raw_pair = T_j' * lambda_tilde_pair; verify
    # dot(g_raw[s,:],lambda_raw) == dot(g_tilde[s,:],lambda_tilde) at a handful of random draws s.
    # CM columns occupy positions ncore:(ncore+2*ncm_cdf-1) within the FULL lambda vector (core
    # columns 1:ncore-1 come first, gravity is last, per wrap_moments_with_cm's own layout
    # contract) -- res.lambda is the FULL dual vector over every inner column, not just the CM
    # sub-block, so it must be sliced with that same offset before any per-pair T_j is applied.
    full_lambda_tilde = res.lambda
    cm_off = ncore - 1
    lambda_tilde = full_lambda_tilde[cm_off+1 : cm_off+2*ncm_cdf]   # [1:ncm_cdf]=ctilde block, [ncm_cdf+1:2ncm_cdf]=second block
    lambda_raw = similar(lambda_tilde)
    for j in 1:ncm_cdf
        Tj = T_block(arm, j)
        v = Tj' * [lambda_tilde[j], lambda_tilde[ncm_cdf+j]]
        lambda_raw[j] = v[1]
        lambda_raw[ncm_cdf+j] = v[2]
    end
    max_dualmap_diff = 0.0
    test_s = unique(rand(1:W_draws, min(500, W_draws)))
    for s in test_s
        lhs = dot(@view(CM_raw[s, :]), lambda_raw)
        rhs = dot(@view(arm_CM[arm][s, :]), lambda_tilde)
        max_dualmap_diff = max(max_dualmap_diff, abs(lhs - rhs))
    end

    lp("ARM $arm: max_cdf_kkt_raw=$max_cdf_kkt_raw  max_pow_kkt_raw=$max_pow_kkt_raw  " *
       "Delta_dual(recompute)=$Delta_dual_recompute (orig $(res.Delta_dual))  max_dualmap_identity_diff=$max_dualmap_diff")
    equiv_results[arm] = (max_cdf_kkt_raw = max_cdf_kkt_raw, max_pow_kkt_raw = max_pow_kkt_raw,
                           Delta_dual_recompute = Delta_dual_recompute, max_dualmap_diff = max_dualmap_diff)
end

# =====================================================================================
# Final verdict-style summary block (machine-parseable)
# =====================================================================================
lp("="^100); lp("VERDICT_BLOCK_START scale=$scale W=$Wsel L=$L"); lp("="^100)
for arm in (:A, :B, :C, :D)
    res = results[arm]
    eq = get(equiv_results, arm, nothing)
    lp("ARM_$arm status=$(res.status) Delta_dual=$(res.Delta_dual) cond_cm=$(res.cond_cm) cond_full=$(res.cond_full) " *
       "t=$(res.telapsed) iters=$(res.iters) m_min=$(res.m_min) m_max=$(res.m_max) " *
       "moment_kkt=$(res.max_abs_moment_kkt_resid)" *
       (eq === nothing ? "" : " raw_cdf_kkt=$(eq.max_cdf_kkt_raw) raw_pow_kkt=$(eq.max_pow_kkt_raw) dualmap_diff=$(eq.max_dualmap_diff)"))
end
lp("VERDICT_BLOCK_END")
lp("DONE scale=$scale W=$Wsel L=$L")

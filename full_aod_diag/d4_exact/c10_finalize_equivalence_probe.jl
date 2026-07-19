# ============================================================================
# Continuation 10, Section 9 (finalize-architecture): end-to-end equivalence
# probe run through the ACTUAL production driver (c10_d20_production_driver.jl),
# used TWICE (once against the pre-change code at diag/fullA-d4-exact tip
# 97cdd80, once against this branch's post-change code) at the IDENTICAL
# real D=20/W=80,000 point with the IDENTICAL draw seed, to directly compare
# numbers rather than re-deriving the isolated-function check workstream (1)
# already did. Prints Delta_dual, gravity_value, KKT residual, moment-residual
# norm, zeta*, ||lambda*||, and the outer composite gradient norm/first few
# components, all to full Float64 precision, so the two runs' stdout logs can
# be diffed directly.
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf, LinearAlgebra

Random.seed!(20260719)
ctx = d20_real_setup(W = 80000, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0 * 1.01, zfree0)
xf0 = x_free_from_w2(w0)

println("=== EQUIVALENCE PROBE (Continuation 10 Section 9) ===")
println("D=", D, " W=80000 draw_seed=20260719 point=(gp0*1.01, zfree0-at-calibration)")

# --- 1. cold evaluation (mirrors screened_eval, warm=false) ---
t0 = time()
r_cold, meta_cold = evaluate_fullA_screened(xf0, ctx; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = false, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_cold = time() - t0
@printf("[cold] wall=%.3fs inner_status=%d screen_status=%s\n", t_cold, r_cold.inner_status, string(meta_cold.screen_status))
@printf("[cold] Delta_dual=%.15f\n", r_cold.Delta_dual)
@printf("[cold] gravity_value=%.15e\n", r_cold.gravity_value)
@printf("[cold] max_abs_moment_kkt_resid=%.15e\n", r_cold.max_abs_moment_kkt_resid)
@printf("[cold] norm_moment_resid=%.15e\n", norm(r_cold.moment_resid))
@printf("[cold] zeta=%.15f\n", r_cold.zeta)
@printf("[cold] norm_lambda=%.15e\n", norm(r_cold.lambda))
@printf("[cold] winner_hash=%s\n", string(r_cold.winner_hash))

# --- 2. warm re-evaluation at the SAME point (mirrors screened_eval, warm=true) ---
t0 = time()
r_warm, meta_warm = evaluate_fullA_screened(xf0, ctx; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = true, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_warm = time() - t0
@printf("[warm] wall=%.3fs inner_status=%d screen_status=%s\n", t_warm, r_warm.inner_status, string(meta_warm.screen_status))
@printf("[warm] Delta_dual=%.15f\n", r_warm.Delta_dual)
@printf("[warm] gravity_value=%.15e\n", r_warm.gravity_value)
@printf("[warm] max_abs_moment_kkt_resid=%.15e\n", r_warm.max_abs_moment_kkt_resid)
@printf("[warm] norm_moment_resid=%.15e\n", norm(r_warm.moment_resid))

# --- 3. outer composite gradient at the same point (base built fresh compressed) ---
base = compressed_base_state(xf0, ctx)
gfull, gmeta = composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
@printf("[grad] norm(gfull)=%.15e\n", norm(gfull))
@printf("[grad] gfull[1:5]=%s\n", string(gfull[1:5]))
@printf("[grad] tie_fallback=%s\n", string(gmeta.tie_fallback))

# --- 4. direct isolated-function check at this specific real cf (separate from workstream 1's
#      own D=4 check -- this is the D=20/W=80,000 real cf the driver actually builds) ---
θ_full0 = CS.reconstruct_full(xf0, ctx.m)
cf0 = build_compressed_factual(θ_full0, ctx; check_ties = true)
ncolI = cf0.oci - 1
G_old = materialize_dense_factual(cf0)
G_new_buf = zeros(cf0.W, ncolI)
materialize_dense_factual_structured!(G_new_buf, cf0)
@printf("[isolated] max|G_new - G_old| = %.3e\n", maximum(abs.(G_new_buf .- G_old)))

# --- 5. isolated KKT/moment-residual BLAS-vs-loop check at this same real G
#      (functions, not top-level begin-blocks, to avoid Julia's top-level soft-scope
#      surprise inside for-loops that mutate an outer accumulator) ---
function kkt_loop_ref(G, m_weights, nkkt, W)
    acc = 0.0
    @inbounds for j in 1:nkkt
        s = 0.0
        for ω in 1:W
            s += m_weights[ω] * G[ω, j]
        end
        acc = max(acc, abs(s / W))
    end
    return acc
end
function mr_loop_ref(G, d, W)
    mr = zeros(d)
    @inbounds for j in 1:d, ω in 1:W
        mr[j] += G[ω, j]
    end
    mr ./= W
    return mr
end
m_w = rand(MersenneTwister(1), cf0.W)   # arbitrary but FIXED weights, just to compare formulas
nkkt = ncolI
kkt_loop = kkt_loop_ref(G_old, m_w, nkkt, cf0.W)
kkt_blas = kkt_residual_blas(G_old, m_w, nkkt, cf0.W)
@printf("[isolated] kkt: loop=%.15e blas=%.15e |diff|=%.3e\n", kkt_loop, kkt_blas, abs(kkt_loop - kkt_blas))

mr_loop = mr_loop_ref(G_old, ncolI, cf0.W)
mr_blas = moment_resid_blas(G_old, ncolI, cf0.W)
@printf("[isolated] moment_resid: |diff|=%.3e\n", maximum(abs.(mr_loop .- mr_blas)))

println("DONE_EQUIVALENCE_PROBE")

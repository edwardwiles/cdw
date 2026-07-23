# Session prompt Section 3: the exact affine map from the free outer coordinates to the
# BASELINE log-cutoff matrix, q(theta_free) = q0 + Q*theta_free (q_od = log(zhat_od),
# zhat = melitz_baseline_cutoff(A,f,...) -- the SAME cutoff the deterministic feasibility
# system (equilibrium.jl's melitz_deterministic_cutoff_constraints,
# delta_star.jl's melitz_cutoff_constraints_at/melitz_cutoff_constraint_jacobian) already
# uses, just in un-differenced form).
#
# `expand_free_theta` (delta_star.jl) is, end to end, an AFFINE map from `theta_free` to
# `vec(log A)`/`vec(log f)`: the A-gravity pivot (`pivot_expand` with `g0=0`) is exactly
# LINEAR in `A_free`; `f[j,j]`'s autarky-cutoff derivation
# (`derive_fjj_from_autarky_cutoff`) is affine in `(log A[j,j], log gamma_prime_j)` (the
# `A_jj^{1-sigma}` and `1/gamma_prime_j` nonlinearities are both in LEVELS -- in LOGS,
# `log f_jj = const + (sigma-1)*log(A_jj) - log(gamma_prime_j)`, exactly affine); the
# f-gravity pivot (`pivot_expand` with `g0 = c[jj]*log(f_jj)`, itself affine in
# `theta_free`) is therefore also affine in `theta_free`. And `q_od` itself
# (`melitz_cutoff`/`melitz_C` composed) is affine in `(log A_od, log f_od)` (again: the
# `sigma-1` power in LEVELS is exactly a linear coefficient once everything is expressed in
# LOGS). The composition of affine maps is affine, so a GLOBAL `(Q, q0)` exists and is
# EXACT (not a local/tangent linearization) -- this file constructs it two independent
# ways and requires them to agree at machine precision.
#
# Two independent constructions, per the session prompt's own instruction not to hand-code
# a fragile pivot-coefficient chain if it can be avoided:
#
#   1. `affine_cutoff_map_basis` (AUTHORITATIVE): treats `expand_free_theta ->
#      melitz_baseline_cutoff -> log` as a black box and recovers `(Q, q0)` from unit
#      free-coordinate basis probes around a fixed origin -- exact for a genuinely affine
#      function up to floating-point roundoff, and automatically correct even if some
#      future session changes the pivot/derivation internals (Section 3.1's own stated
#      goal).
#   2. `affine_cutoff_map_analytical` (INDEPENDENT CROSS-CHECK): hand-derives the SAME map
#      by chaining the A-gravity-pivot's linear expansion matrix, `f[j,j]`'s closed-form
#      affine dependence on `(log A_jj, g)`, and the f-gravity-pivot's affine expansion --
#      WITHOUT ever calling `expand_free_theta`. A genuinely different code path (matrix
#      algebra on the `GravityPivot` structs directly), not merely the same function
#      differentiated a second way.
#
# Both are then used to build the D domestic-support + D*(D-1) export-selection affine
# INEQUALITY system `C*theta_free + b >= 0` (Section 3.2), row-scaled (Section 3.3), and
# registered as true KNITRO linear constraints (Section 3.3, affine_cutoff_knitro.jl) --
# replacing the generic nonlinear FC-callback evaluation of the SAME restriction.

using LinearAlgebra: norm

"""
    melitz_log_cutoff_vec(theta_free, ctx) -> q (length D^2, column-major od2lin order)

`vec(log.(melitz_baseline_cutoff(expand_free_theta(theta_free, ctx)..., ctx.w, ctx.tau,
ctx.expenditure, ctx.sigma)))` -- the full baseline log-cutoff vector at a free outer
point, in the SAME `od2lin`/`lin2od` column-major linear-index convention used throughout
`delta_star.jl`/`equilibrium.jl`.
"""
function melitz_log_cutoff_vec(theta_free::AbstractVector, ctx)
    A, f, _, _ = expand_free_theta(theta_free, ctx)
    zhat = melitz_baseline_cutoff(A, f, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
    return vec(log.(zhat))
end

"""
    melitz_free_dim(ctx) -> n

`2*ctx.D^2 - 2`, the length of `theta_free` -- read off `ctx.f_free_lin`
(`length == D^2-1`) rather than hand-recomputing `D^2`, so this stays correct even if the
outer layout convention ever changes.
"""
melitz_free_dim(ctx) = 1 + (ctx.D^2 - 1) + (length(ctx.f_free_lin) - 1)

"""
    affine_cutoff_map_basis(ctx; base=zeros(melitz_free_dim(ctx))) -> (Q, q0)

Section 3.1's AUTHORITATIVE construction: `q0 = q(base)`, `Q[:,k] = q(base + e_k) -
q(base)` for every free coordinate `k`. Exact for an affine function (up to floating-point
roundoff -- NOT a finite-difference approximation: no `h` division, a UNIT step, so no
truncation error is even possible for a truly affine map, only accumulated rounding).
`q0` is then recovered as `q(base) - Q*base` so the returned map is anchored at the
TRUE origin (`theta_free = 0`), independent of which `base` was used to probe it --
verified by the "base-independence" test (any two `base` choices must give identical
`(Q, q0)` to machine precision, since both are probing the SAME affine function).
"""
function affine_cutoff_map_basis(ctx; base::AbstractVector=zeros(melitz_free_dim(ctx)))
    n = length(base)
    D2 = ctx.D^2
    q_base = melitz_log_cutoff_vec(base, ctx)
    Q = zeros(Float64, D2, n)
    ei = zeros(Float64, n)
    @inbounds for k in 1:n
        ei[k] = 1.0
        Q[:, k] .= melitz_log_cutoff_vec(base .+ ei, ctx) .- q_base
        ei[k] = 0.0
    end
    q0 = q_base .- Q * base
    return Q, q0
end

"""
    affine_cutoff_map_analytical(ctx) -> (Q, q0)

Section 3.1's INDEPENDENT CROSS-CHECK: hand-derives `(Q, q0)` directly from the
`GravityPivot` structs and the closed-form `derive_fjj_from_autarky_cutoff`/`melitz_C`/
`melitz_cutoff` algebra, WITHOUT calling `expand_free_theta` or `melitz_baseline_cutoff` --
a genuinely separate code path from `affine_cutoff_map_basis`.

Derivation (all in LOGS, so every step below is exactly LINEAR or AFFINE, never a
power-law nonlinearity):

  1. `log A` (length `D^2`) is a LINEAR map of `A_free = theta_free[2:1+nA]`
     (`nA = D^2-1`): the A-gravity pivot has `g0=0` always (every A cell free), so
     `pivot_expand` reduces to `logA_full[other[k]] = A_free[k]`,
     `logA_full[pivot] = -sum_k c[other[k]]*A_free[k] / c[pivot]` -- both linear, no
     additive constant. Call this matrix `M_A` (`D^2 x nA`).

  2. `log A[j,j] = m_Ajj . A_free` where `m_Ajj` is `M_A`'s `jj_lin`-th row (a linear
     FUNCTIONAL of `A_free`, hence of `theta_free`).

  3. From `derive_fjj_from_autarky_cutoff` (`f_jj = expenditure_prime_j *
     (markup*w_prime_j/A_jj)^(1-sigma) / (sigma*w_prime_j*gamma_prime_j)`), taking logs:
     `log f_jj = const_fjj + (sigma-1)*log(A_jj) - log(gamma_prime_j)`
     `= const_fjj + (sigma-1)*(m_Ajj . A_free) - theta_free[1]` -- AFFINE in `theta_free`
     (constant `const_fjj` plus a linear functional), matching the session prompt's own
     Section 3's claim that `log f[j,j]` is affine in `(log A_jj, g)`.

  4. The f-gravity pivot's offset is `g0_f = c_full[jj_lin] * log(f_jj)`, itself AFFINE in
     `theta_free` by step 3. `logf_free_full` (the `D^2-1`-length vector over
     `f_free_lin`) is then affine in `theta_free`: identity onto `f_free_free =
     theta_free[2+nA:end]` at every `other` cell, and, at the f-pivot cell,
     `(-g0_f - sum_k c_f[other[k]]*f_free_free[k]) / c_f[pivot]` -- affine (the `g0_f`
     term contributes the ONLY additive constant in the whole chain, via `const_fjj`).

  5. `q_od = log(markup) + log(w_o) + log(tau_od) - log(A_od) + (log(sigma) + log(w_o) +
     log(f_od) - log(expenditure_d)) / (sigma-1)` (derived directly from `melitz_C`/
     `melitz_cutoff`'s own defining formulas, algebraically identical to the session
     prompt's own Section 3 formula with `markup = sigma/(sigma-1)`) -- affine in
     `(log A_od, log f_od)`, hence, composing with steps 1-4, affine in `theta_free`.

Requires agreement with `affine_cutoff_map_basis` at machine precision (test suite).
"""
function affine_cutoff_map_analytical(ctx)
    D, j = ctx.D, ctx.target_country
    D2 = D^2
    nA = D2 - 1
    n = melitz_free_dim(ctx)
    sigma = ctx.sigma
    markup = melitz_markup(sigma)
    c_full = ctx.c_full
    A_pivot = ctx.A_pivot
    jj_lin = ctx.jj_lin
    f_free_lin = ctx.f_free_lin

    # --- Step 1-2: M_A (D^2 x nA), the exact LINEAR map A_free -> logA_full. ---
    M_A = zeros(Float64, D2, nA)
    @inbounds for (k, i) in enumerate(A_pivot.other)
        M_A[i, k] = 1.0
    end
    cf_A_pivot = c_full[A_pivot.pivot]
    @inbounds for (k, i) in enumerate(A_pivot.other)
        M_A[A_pivot.pivot, k] = -c_full[i] / cf_A_pivot
    end
    m_Ajj = M_A[jj_lin, :]   # length nA

    # --- Step 3: log(f_jj) affine in theta_free: coef_fjj . theta_free + offset_fjj. ---
    w_prime_j = ctx.w_prime
    expenditure_prime_j = ctx.w_prime * ctx.L[j]
    offset_fjj = log(expenditure_prime_j) + (1 - sigma) * (log(markup) + log(w_prime_j)) -
                 log(sigma) - log(w_prime_j)
    coef_fjj = zeros(Float64, n)
    coef_fjj[1] = -1.0
    coef_fjj[2:1+nA] .= (sigma - 1) .* m_Ajj

    # --- Step 4: log(f) over f_free_lin (D^2-1 cells) affine in theta_free. ---
    avoid_f = f_gravity_pivot_avoid_indices(D, f_free_lin, A_pivot.pivot)
    f_pivot = build_gravity_pivot(c_full[f_free_lin], 0.0; avoid=avoid_f)
    c_f = c_full[f_free_lin]
    nDom = length(f_free_lin)   # D^2-1
    coef_logf_dom = zeros(Float64, nDom, n)
    offset_logf_dom = zeros(Float64, nDom)
    @inbounds for (k, i) in enumerate(f_pivot.other)
        coef_logf_dom[i, 1+nA+k] = 1.0
    end
    cf_f_pivot = c_f[f_pivot.pivot]
    @inbounds for (k, i) in enumerate(f_pivot.other)
        coef_logf_dom[f_pivot.pivot, 1+nA+k] -= c_f[i] / cf_f_pivot
    end
    scale_g0 = -c_full[jj_lin] / cf_f_pivot
    coef_logf_dom[f_pivot.pivot, :] .+= scale_g0 .* coef_fjj
    offset_logf_dom[f_pivot.pivot] += scale_g0 * offset_fjj

    # --- assemble full log(f) (D^2 cells, including j,j) affine in theta_free. ---
    coef_logf_full = zeros(Float64, D2, n)
    offset_logf_full = zeros(Float64, D2)
    coef_logf_full[jj_lin, :] .= coef_fjj
    offset_logf_full[jj_lin] = offset_fjj
    @inbounds for (k, i) in enumerate(f_free_lin)
        coef_logf_full[i, :] .= coef_logf_dom[k, :]
        offset_logf_full[i] = offset_logf_dom[k]
    end

    # --- Step 5: q_od affine in (log A_od, log f_od), hence in theta_free. ---
    Q = zeros(Float64, D2, n)
    q0 = zeros(Float64, D2)
    invsm1 = 1 / (sigma - 1)
    @inbounds for lin in 1:D2
        o, d = lin2od(lin, D)
        const_od = log(markup) + log(ctx.w[o]) + log(ctx.tau[o, d]) +
                   invsm1 * (log(sigma) + log(ctx.w[o]) - log(ctx.expenditure[d]))
        coef_logA_full_lin = zeros(Float64, n)
        coef_logA_full_lin[2:1+nA] .= M_A[lin, :]
        Q[lin, :] .= -coef_logA_full_lin .+ invsm1 .* coef_logf_full[lin, :]
        q0[lin] = const_od + invsm1 * offset_logf_full[lin]
    end
    return Q, q0
end

"""
    MelitzAffineCutoffSystem

The complete Section 3.2 deterministic affine inequality system `C*theta_free + b >= 0`,
`C = S*Q`, `b = S*q0 - epsilon_vector`, plus its row-scaling factors (Section 3.3) and a
label for every row (for diagnostics/reporting). Row order: `D` domestic-support rows
(`o=1:D`, in order), then `D*(D-1)` export-selection rows (`(o,d)` with `d != o`, `o`
outer/`d` inner loop order, matching `melitz_deterministic_cutoff_constraints`'s own
`g_export` ordering exactly).
"""
struct MelitzAffineCutoffSystem
    D::Int
    Q::Matrix{Float64}          # D^2 x n, the full affine cutoff map's linear part
    q0::Vector{Float64}         # D^2, the full affine cutoff map's constant part
    C::Matrix{Float64}          # m x n (m = D + D*(D-1)), ROW-SCALED
    b::Vector{Float64}          # m, ROW-SCALED
    C_raw::Matrix{Float64}      # m x n, UNSCALED (C before row scaling)
    b_raw::Vector{Float64}      # m, UNSCALED
    row_scale::Vector{Float64}  # m, scale[row] = max(1, norm(C_raw[row,:], 2))
    row_labels::Vector{Tuple{Symbol,Int,Int}}  # (:domestic or :export, o, d) per row
    epsilon_support::Float64
    epsilon_export::Float64
    n_domestic::Int
    n_export::Int
end

"""
    build_melitz_affine_cutoff_system(ctx; epsilon_support=0.0, epsilon_export=0.0,
        method=:basis, base=zeros(melitz_free_dim(ctx))) -> MelitzAffineCutoffSystem

Section 3.2/3.3: builds the complete row-scaled affine system from `(Q, q0)`
(`affine_cutoff_map_basis` by default -- the AUTHORITATIVE construction; pass
`method=:analytical` to build from the independent hand-derived map instead, e.g. for the
equivalence test). `epsilon_support`/`epsilon_export` default to `0.0` (the ACTIVE economic
convention, `zhat[o,o]>=1`/export-selection with NO margin) -- a positive value is a
genuine ADDITIONAL numerical safety margin, not the governing economic restriction; do not
set either away from `0.0` in a result presented as satisfying the paper's own feasibility
definition.
"""
function build_melitz_affine_cutoff_system(ctx; epsilon_support::Real=0.0,
                                            epsilon_export::Real=0.0,
                                            method::Symbol=:basis,
                                            base::AbstractVector=zeros(melitz_free_dim(ctx)))
    D = ctx.D
    Q, q0 = method == :basis ? affine_cutoff_map_basis(ctx; base=base) :
            method == :analytical ? affine_cutoff_map_analytical(ctx) :
            error("method must be :basis or :analytical, got $method")
    n = size(Q, 2)

    n_domestic = D
    n_export = D * (D - 1)
    m = n_domestic + n_export
    C_raw = zeros(Float64, m, n)
    b_raw = zeros(Float64, m)
    labels = Vector{Tuple{Symbol,Int,Int}}(undef, m)

    row = 0
    @inbounds for o in 1:D
        row += 1
        oo = od2lin(o, o, D)
        C_raw[row, :] .= Q[oo, :]
        b_raw[row] = q0[oo] - epsilon_support
        labels[row] = (:domestic, o, o)
    end
    @inbounds for o in 1:D, d in 1:D
        d == o && continue
        row += 1
        oo = od2lin(o, o, D)
        od = od2lin(o, d, D)
        C_raw[row, :] .= Q[od, :] .- Q[oo, :]
        b_raw[row] = q0[od] - q0[oo] - epsilon_export
        labels[row] = (:export, o, d)
    end

    row_scale = [max(1.0, norm(view(C_raw, r, :), 2)) for r in 1:m]
    C = C_raw ./ row_scale
    b = b_raw ./ row_scale

    return MelitzAffineCutoffSystem(D, Q, q0, C, b, C_raw, b_raw, row_scale, labels,
        Float64(epsilon_support), Float64(epsilon_export), n_domestic, n_export)
end

"""
    affine_cutoff_slacks(sys::MelitzAffineCutoffSystem, theta_free) -> (g_domestic, g_export)

`C*theta_free + b`, split back into the domestic/export blocks -- SAME shapes/ordering as
`melitz_cutoff_constraints_at`'s own `(g_domestic, g_export)`, for direct equivalence
testing. Uses `sys.C`/`sys.b` (row-scaled) by default; row scaling only rescales each row's
own units, so the FEASIBILITY SIGN (`>=0` or `<0`) is invariant to it (`row_scale[r] > 0`
always) even though the raw numeric slack differs from `melitz_cutoff_constraints_at`'s
unscaled output by exactly `row_scale[r]`.
"""
function affine_cutoff_slacks(sys::MelitzAffineCutoffSystem, theta_free::AbstractVector)
    g = sys.C * theta_free .+ sys.b
    return g[1:sys.n_domestic], g[sys.n_domestic+1:end]
end

"""
    affine_cutoff_slacks_unscaled(sys, theta_free) -> (g_domestic, g_export)

Same as `affine_cutoff_slacks` but using `C_raw`/`b_raw` -- directly comparable, cell for
cell, to `melitz_cutoff_constraints_at(theta_free, ctx)` (both unscaled).
"""
function affine_cutoff_slacks_unscaled(sys::MelitzAffineCutoffSystem, theta_free::AbstractVector)
    g = sys.C_raw * theta_free .+ sys.b_raw
    return g[1:sys.n_domestic], g[sys.n_domestic+1:end]
end

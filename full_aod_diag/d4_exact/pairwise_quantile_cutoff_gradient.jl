# ================================================================================================
# Fixed-dual boundary-crossing outer gradient for the pairwise-quantile-independence restriction's
# cutoff coordinates (draft eq. 32, Section 7 of the implementation plan).
#
# Core idea (task's own): with (zeta,lambda) held FIXED at the converged inner-dual solution,
# moving one cutoff q_{o,r} only changes R_w for draws whose bin membership for origin `o` crosses
# that boundary -- everything else in R_w (the economic part, every other origin's bin, every other
# cutoff) is untouched. So the fixed-dual objective change Delta_f = f(q_new) - f(q_old) can be
# computed from ONLY the crossed draws, in O(D) work per crossed draw (task's own ΔR_w formula),
# never by re-evaluating Psi over all W draws and never by re-solving the inner problem.
#
# Requires pairwise_quantile_bin_context.jl (PairwiseQuantileOperator, PairwiseQuantileBinState)
# and pairwise_quantile_cutoff_transform.jl (cutoff_jacobian_block!) to already be included.
# ================================================================================================

"Psi(x) evaluated at a single point, matching cc_algo/Psi.jl's Psi! piecewise formula exactly --
used for O(1)-per-crossed-draw objective evaluation (no full-length temp array needed)."
function psi_scalar(x::Float64)
    return x <= 1.0 ? (exp(x) - 1.0) : (0.5 * exp(1) * (x^2 + 1.0) - 1.0)
end

"""
    crossed_draw_range(op, o, q_old, q_new) -> UnitRange{Int}

Using the presorted column `op.sorted_z[:,o]`, returns the (possibly empty) index range into
`sorted_z[:,o]`/`sorted_idx[:,o]` covering exactly the draws whose value lies in the half-open
interval `(min(q_old,q_new), max(q_old,q_new)]` -- i.e. exactly the draws whose bin membership for
origin `o` changes when ONLY the cutoff at this position moves from `q_old` to `q_new` (every other
cutoff for origin `o` held fixed). `O(log W)` (two `searchsortedlast` calls), never an `O(W)` scan.
"""
function crossed_draw_range(op::PairwiseQuantileOperator, o::Int, q_old::Float64, q_new::Float64)
    col = @view op.sorted_z[:, o]
    lo = min(q_old, q_new); hi = max(q_old, q_new)
    first_idx = searchsortedlast(col, lo) + 1
    last_idx = searchsortedlast(col, hi)
    return first_idx:last_idx
end

"""
    bandwidth_target(op, o, q_old, direction, min_crossed, neighbor_bound) -> Float64

Deterministic, data-driven secant bandwidth (task Section 7): the smallest `|Δq|` step in the
given `direction` (`:up` or `:down`) that crosses AT LEAST `min_crossed` draws of origin `o`,
found directly from the presorted column via index offset (`O(1)` beyond the initial `O(log W)`
search -- no scanning/iteration needed), clamped so `q_new` never reaches or crosses
`neighbor_bound` (the adjacent cutoff `q_{o,r-1}` or `q_{o,r+1}`, respecting neighbor order).
Returns `q_old` unchanged if no draws are available in that direction (e.g. `direction=:up` but
every remaining draw is already below `q_old`) or if the neighbor bound leaves no room to move.
"""
function bandwidth_target(op::PairwiseQuantileOperator, o::Int, q_old::Float64, direction::Symbol,
                           min_crossed::Int, neighbor_bound::Float64)
    min_crossed >= 1 || error("bandwidth_target: min_crossed must be >= 1, got $min_crossed")
    col = @view op.sorted_z[:, o]
    W = op.W
    if direction === :up
        start = searchsortedlast(col, q_old) + 1
        start > W && return q_old
        target_idx = min(start + min_crossed - 1, W)
        q_candidate = col[target_idx]
        q_new = min(q_candidate, prevfloat(neighbor_bound))
        return max(q_new, q_old)
    elseif direction === :down
        stop = searchsortedlast(col, q_old)
        stop < 1 && return q_old
        target_idx = max(stop - min_crossed + 1, 1)
        q_candidate = col[target_idx]
        q_new = max(q_candidate, nextfloat(neighbor_bound))
        return min(q_new, q_old)
    else
        error("bandwidth_target: direction must be :up or :down, got $direction")
    end
end

"""
    fixed_dual_delta_f(op, state, o, r, q_new, lambda_M, lambda_P, r_current) -> (delta_f, k_crossed)

Task's exact `ΔR_w` formula, applied to every draw in `crossed_draw_range(op,o,q_old,q_new)`
(`q_old = state.Q[r,o]`), then the fixed-dual objective change
`Δf = (1/W) * Σ_{w crossed} [Ψ(R_w + ΔR_w) - Ψ(R_w)]` via `psi_scalar` (task: "evaluate the
fixed-dual objective change using the current Ψ machinery"). `r_current[w]` is the caller-supplied,
already-converged `R_w` at the CURRENT cutoffs/dual (never recomputed from scratch here). Cost:
`O(k_crossed * D)`, matching the task's stated per-crossed-draw cost -- `state`/`op` are read-only,
NOT mutated (this function never calls `refresh_pairwise_quantile_bins!`).
"""
function fixed_dual_delta_f(op::PairwiseQuantileOperator, state::PairwiseQuantileBinState, o::Int, r::Int,
        q_new::Float64, lambda_M::AbstractMatrix{Float64}, lambda_P::AbstractArray{Float64,3},
        r_current::AbstractVector{Float64})
    D = op.D
    nlast = UInt8(op.L - 1)   # last ACTIVE bin index; bin L (the omitted implicit-zero bin) is > nlast
    q_old = state.Q[r, o]
    rng = crossed_draw_range(op, o, q_old, q_new)
    isempty(rng) && return (0.0, 0)

    bin = state.bin
    pairs = op.pairs
    # b_old for every crossed draw is the SAME single value (r or r+1, depending on direction) --
    # but recomputed directly via searchsortedfirst against the OLD/NEW cutoff columns rather than
    # hand-derived from direction, matching refresh_pairwise_quantile_bins!'s own convention exactly
    # (robust against any off-by-one, at negligible extra cost since L-1 cutoffs is O(1) to search).
    Qold_col = @view state.Q[:, o]
    Qnew = copy(Qold_col); Qnew[r] = q_new

    delta_f = 0.0
    k_crossed = 0
    @inbounds for k in rng
        w = op.sorted_idx[k, o]
        b_old = bin[w, o]
        b_new = UInt8(searchsortedfirst(Qnew, op.sorted_z[k, o]))
        b_old == b_new && continue   # can happen at a shared boundary point; no actual change
        k_crossed += 1

        dR = 0.0
        b_old <= nlast && (dR -= lambda_M[o, b_old])
        b_new <= nlast && (dR += lambda_M[o, b_new])
        for p in 1:D
            p == o && continue
            bp = bin[w, p]
            bp > nlast && continue
            pidx = pair_index(op, o, p)
            (op1, op2) = pairs[pidx]
            if op1 == o
                b_old <= nlast && (dR -= lambda_P[b_old, bp, pidx])
                b_new <= nlast && (dR += lambda_P[b_new, bp, pidx])
            else
                b_old <= nlast && (dR -= lambda_P[bp, b_old, pidx])
                b_new <= nlast && (dR += lambda_P[bp, b_new, pidx])
            end
        end

        Rw_old = r_current[w]
        delta_f += psi_scalar(Rw_old + dR) - psi_scalar(Rw_old)
    end
    return (delta_f / op.W, k_crossed)
end

"""
    cutoff_secant_gradient!(grad_raw, op, state, lambda_M, lambda_P, r_current, layout, raw;
                             min_crossed) -> grad_raw

Fills `grad_raw` (length `n_raw(layout)`) with the fixed-dual boundary-crossing secant estimate of
`∂f/∂raw_k` for EVERY raw outer coordinate, one origin/level at a time. For each `(o,r)`,
`r=1,...,n_cutoffs(layout)=layout.L-1`:
  1. `bandwidth_target` picks an up-step and a down-step (clamped to respect `q_{o,r-1}<q_{o,r}<
     q_{o,r+1}`, using `-Inf`/`+Inf` as the implicit bound at `r=1`/`r=n_cutoffs(layout)`).
  2. `fixed_dual_delta_f` gives `Δf_up`/`Δf_down` from ONLY the crossed draws.
  3. Central secant `(Δf_up - Δf_down)/(q_up - q_down)` when BOTH directions achieved at least
     `min_crossed` crossed draws; one-sided `Δf/(q_new-q_old)` otherwise (whichever direction
     succeeded) -- task's "central when possible" instruction.
  4. The physical-cutoff derivative is mapped back to the `n_cutoffs(layout)` raw coordinates of
     origin `o` via `cutoff_jacobian_block!` (chain rule: `∂f/∂raw_k = Σ_r (∂f/∂q_r)*(∂q_r/∂raw_k)`),
     accumulated into `grad_raw[raw_index(layout,o,k)]` across all levels `r` of that origin (each
     level's Jacobian ROW contributes to every raw coordinate `k<=r`, per `cutoff_jacobian_block!`'s
     own lower-triangular structure).

`min_crossed` has NO default (task's own no-silent-defaults convention) -- callers must choose it
based on the campaign's `W` (e.g. a few hundred draws at `W=100,000`).
"""
function cutoff_secant_gradient!(grad_raw::AbstractVector{Float64}, op::PairwiseQuantileOperator,
        state::PairwiseQuantileBinState, lambda_M::AbstractMatrix{Float64}, lambda_P::AbstractArray{Float64,3},
        r_current::AbstractVector{Float64}, layout::PairwiseQuantileCutoffLayout, raw::AbstractVector{Float64};
        min_crossed::Int)
    D = op.D
    nc = n_cutoffs(layout)
    length(grad_raw) == n_raw(layout) || error("cutoff_secant_gradient!: length(grad_raw) mismatch")
    fill!(grad_raw, 0.0)
    J = zeros(nc, nc)
    dfdq = zeros(nc)

    @inbounds for o in 1:D
        base = raw_index(layout, o, 1)
        rawo = @view raw[base:base+nc-1]
        Qcol = @view state.Q[:, o]
        cutoff_jacobian_block!(J, rawo, Qcol)

        for r in 1:nc
            q_old = Qcol[r]
            lo_bound = r == 1 ? -Inf : Qcol[r-1]
            hi_bound = r == nc ? Inf : Qcol[r+1]

            q_up = bandwidth_target(op, o, q_old, :up, min_crossed, hi_bound)
            q_down = bandwidth_target(op, o, q_old, :down, min_crossed, lo_bound)
            (df_up, k_up) = q_up > q_old ? fixed_dual_delta_f(op, state, o, r, q_up, lambda_M, lambda_P, r_current) : (0.0, 0)
            (df_down, k_down) = q_down < q_old ? fixed_dual_delta_f(op, state, o, r, q_down, lambda_M, lambda_P, r_current) : (0.0, 0)

            if k_up >= min_crossed && k_down >= min_crossed
                dfdq[r] = (df_up - df_down) / (q_up - q_down)
            elseif k_up >= min_crossed
                dfdq[r] = df_up / (q_up - q_old)
            elseif k_down >= min_crossed
                dfdq[r] = df_down / (q_down - q_old)
            elseif k_up > 0
                dfdq[r] = df_up / (q_up - q_old)
            elseif k_down > 0
                dfdq[r] = df_down / (q_down - q_old)
            else
                dfdq[r] = 0.0   # no draws available in either direction (degenerate: q_r has no room to move)
            end
        end

        for k in 1:nc
            gi = raw_index(layout, o, k)
            acc = 0.0
            for r in 1:nc
                acc += dfdq[r] * J[r, k]
            end
            grad_raw[gi] = acc
        end
    end
    return grad_raw
end

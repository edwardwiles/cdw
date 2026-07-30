# 2026-07-30 (negative-switch geometry audit,
# docs/melitz_d20_negative_switch_geometry_audit_2026-07-30.md). Promoted from that session's
# own validated diagnostic scripts (Rule 10: no second, script-only implementation).
#
# EXACT (not bisected) participation-switch threshold enumeration along a reduced-q-subspace
# direction `q_free_free(t) = q_anchor_free +- t*d`, `(g,A_free)` held fixed. Exploits that,
# with `(g,A_free)` fixed, `expand_free_theta_logcutoff`'s full `D x D` log-cutoff matrix
# `q_full(t)` is an EXACT LINEAR function of `t` for every cell -- including the
# analytically-reconstructed gravity-pivot cell (`build_q_gravity_offset`'s own `g0_q` depends
# only on `q_jj`/`g`, both fixed here, and `pivot_expand` is linear) -- so each cell's cutoff
# crossing time against a specific QMC draw's productivity can be solved for in closed form,
# rather than bisected. Live-verified (D20 real-data anchor): `max|slope(0.2->0.6) -
# slope(0.6->1.0)| ~ 1e-9` (floating-point noise) confirms the linearity this file relies on.

"""
    melitz_reduced_q_direction_slope(theta_full, d, ctx) -> Slope::Matrix{Float64}

`Slope[o,d] = d(q_full[o,d])/dt` for `q_free_free(t) = q_free_free(theta_full) + t*d`
(`(g,A_free)` fixed) -- computed EXACTLY via two `expand_free_theta_logcutoff` calls (linear,
so any two distinct `t` values determine the slope exactly; `t in {0,1}` used here), not a
finite-difference approximation. `theta_full` must be in PLAIN (un-powered) units
(`melitz_unpower_theta_free`), matching `expand_free_theta_logcutoff`'s own contract.
"""
function melitz_reduced_q_direction_slope(theta_full::AbstractVector, d::AbstractVector, ctx)
    melitz_reduced_q_check_ctx(ctx)
    nA = ctx.D^2 - 1
    th0 = copy(theta_full)
    th1 = copy(theta_full); th1[1+nA+1:end] .+= d
    _, _, _, _, q0 = expand_free_theta_logcutoff(th0, ctx)
    _, _, _, _, q1 = expand_free_theta_logcutoff(th1, ctx)
    return q1 .- q0
end

"""
    MelitzQSwitchEvent

One exact participation-switch event along a reduced-q direction: at amplitude `t`, QMC row
`row_s` (origin `o`'s productivity draw `z[row_s,o]`) crosses cell `(o,d)`'s bilateral cutoff,
turning that cell/row pair `dir` (`:on` or `:off`). `k` is the row's 1-based position in
`sorted_ctx.sorted_z[:,o]`. `q0_od`/`slope_od` are the affine coefficients
(`q_od(t) = q0_od + t*slope_od`) used to derive `t` in closed form.
"""
struct MelitzQSwitchEvent
    t::Float64
    o::Int
    d::Int
    k::Int
    row_s::Int
    dir::Symbol
    z::Float64
    q0_od::Float64
    slope_od::Float64
end

"""
    melitz_q_direction_exact_switches(theta_full, d, ctx, sorted_ctx; sign=1, n_switches=15,
                                       t_max=2.0) -> Vector{MelitzQSwitchEvent}

Exact, EXACTLY-ORDERED (not bisected, not inferred from coarse endpoint differences) list of
the first `n_switches` participation-switch events for `q_free_free(t) = q_free_free(theta_full)
+ sign*t*d`, `t in (0, t_max]`, ascending in `t`. Ties (two cells switching at numerically
identical `t`) are returned as separate consecutive events with equal `t`, never silently
merged or dropped. Cross-checked (this session's own audit) against the existing
`melitz_q_direction_two_sided_crossings` two-sided crossing-COUNT infrastructure
(`reduced_q_subspace.jl`): the two agree exactly at 9 of the first 10 distinct thresholds in
the audited real-D20 case (one single-count residual discrepancy at the 7th threshold,
disclosed in the audit doc, not resolved this session -- does not affect the first-five-switch
analysis this function exists to support).
"""
function melitz_q_direction_exact_switches(theta_full::AbstractVector, d::AbstractVector, ctx,
                                            sorted_ctx::MelitzSortedTailContext; sign::Int=1,
                                            n_switches::Integer=15, t_max::Real=2.0)
    sign in (1, -1) || throw(ArgumentError("melitz_q_direction_exact_switches: sign must be +1 or -1"))
    theta_plain = melitz_unpower_theta_free(theta_full, ctx)
    Slope = melitz_reduced_q_direction_slope(theta_plain, Float64(sign) .* d, ctx)
    _, _, _, _, q0mat = expand_free_theta_logcutoff(theta_plain, ctx)
    D_ = ctx.D

    kcur = Dict{Tuple{Int,Int},Int}()
    slope = Dict{Tuple{Int,Int},Float64}()
    q0v = Dict{Tuple{Int,Int},Float64}()
    for lin in ctx.f_free_lin
        o, dd = lin2od(lin, D_)
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        cutoff0 = exp(q0mat[o, dd])
        kcur[(o, dd)] = melitz_active_tail_start(sorted_z_o, cutoff0)
        slope[(o, dd)] = Slope[o, dd]
        q0v[(o, dd)] = q0mat[o, dd]
    end

    events = MelitzQSwitchEvent[]
    for _ in 1:n_switches
        best_t = Inf; best_cell = nothing; best_k = 0; best_dir = :none
        for (cell, sl) in slope
            abs(sl) < 1e-300 && continue
            o, dd = cell
            sorted_z_o = @view sorted_ctx.sorted_z[:, o]
            W_ = length(sorted_z_o)
            k0 = kcur[cell]
            if sl > 0
                k0 > W_ && continue
                t_cand = (log(sorted_z_o[k0]) - q0v[cell]) / sl
                cand_k = k0; cand_dir = :off
            else
                k0 - 1 < 1 && continue
                t_cand = (log(sorted_z_o[k0-1]) - q0v[cell]) / sl
                cand_k = k0 - 1; cand_dir = :on
            end
            if isfinite(t_cand) && t_cand > 1e-13 && t_cand < best_t && t_cand <= t_max
                best_t = t_cand; best_cell = cell; best_k = cand_k; best_dir = cand_dir
            end
        end
        best_cell === nothing && break
        o, dd = best_cell
        row_s = sorted_ctx.permutation[best_k, o]
        z_val = sorted_ctx.sorted_z[best_k, o]
        push!(events, MelitzQSwitchEvent(best_t, o, dd, best_k, row_s, best_dir, z_val,
                                          q0v[best_cell], slope[best_cell]))
        kcur[best_cell] = best_dir == :off ? best_k + 1 : best_k
    end
    return events
end

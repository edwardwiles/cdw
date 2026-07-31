# One-off correction: the Phase-1 resume driver's incumbent-selection criterion
# (update_most_extreme! in melitz_overnight_phase1_resume_2026-07-31.jl, copied from the
# original production driver) only gated on classification==:FiniteSolved && within_budget, NOT
# lfd_ok. On the three lower-direction chains, bracket subdivision pushed the "most extreme"
# checkpointed point to GT values where the PRIMAL solve is genuinely FiniteSolved (bit-exact on
# fresh cold re-solve) but LFD/dual recovery deterministically fails (confirmed failing again on
# an independent fresh-process re-check, not a caching artifact) -- a real near-cliff dual
# degeneracy, not a search bug. The governing overnight prompt requires "fully verify the inner
# LFD" at every candidate, so an lfd_ok=false point cannot stand as the reported incumbent.
#
# This script re-points each affected chain's most_extreme_idx at the most extreme point in its
# own history satisfying classification==:FiniteSolved && within_budget && lfd_ok==true. It does
# NOT delete or alter any evaluated point -- purely a bookkeeping correction of which already-
# evaluated point counts as "the incumbent".
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Serialization

mutable struct ChainPoint
    idx::Int
    phase::Symbol
    g::Float64
    GT::Float64
    classification::Symbol
    Delta::Float64
    within_budget::Bool
    best_start::Symbol
    trigger_reason::Symbol
    A_free::Vector{Float64}
    theta_free::Vector{Float64}
    f_full::Matrix{Float64}
    q_full::Matrix{Float64}
    dual_x::Vector{Float64}
    lfd_weights::Vector{Float64}
    lfd_ok::Bool
    lfd_Delta::Float64
    nStatus::Int
    unique_A_points::Int
    unique_inner_solves::Int
    n_fc_calls::Int
    n_ga_calls::Int
    wall_s::Float64
    timestamp::String
end
mutable struct ChainState
    anchor_label::String
    direction::Symbol
    fingerprint::NamedTuple
    points::Vector{ChainPoint}
    phase::Symbol
    step_gt::Float64
    bracket::Union{Nothing,NTuple{4,Float64}}
    n_evaluated::Int
    n_polish::Int
    n_accepted::Int
    most_extreme_idx::Int
    n_nonimproving::Int
    elapsed_wall_s::Float64
    status::Symbol
    cur_A_free::Vector{Float64}
    cur_q::Matrix{Float64}
    cur_p_star::Vector{Float64}
    cur_g::Float64
    cur_GT::Float64
    prev_sys::Any
end

const PRODDIR = joinpath(REPO2, "docs", "key_results", "production_delta0p5_2026-07-31")

for (anchor, direction, sign) in [
    ("current_calibration", "lower", -1),
    ("reduced_q_pre_switch", "lower", -1),
    ("reduced_q_post_switch", "lower", -1),
]
    chainid = "$(anchor)_$(direction)"
    ckpt = joinpath(PRODDIR, "checkpoints", "$(chainid).jls")
    state = deserialize(ckpt)
    old_idx = state.most_extreme_idx
    old_best = state.points[old_idx]

    best_idx = 0
    best_GT = NaN
    for (i, pt) in enumerate(state.points)
        pt.classification == :FiniteSolved || continue
        pt.within_budget || continue
        pt.lfd_ok || continue
        if best_idx == 0 || sign * (pt.GT - state.points[1].GT) > sign * (best_GT - state.points[1].GT)
            best_idx = i
            best_GT = pt.GT
        end
    end
    @assert best_idx != 0 "[$chainid] no FiniteSolved && within_budget && lfd_ok point found at all"
    new_best = state.points[best_idx]
    println("[$chainid] OLD incumbent: idx=$old_idx GT=$(old_best.GT) Delta=$(old_best.Delta) lfd_ok=$(old_best.lfd_ok)")
    println("[$chainid] NEW incumbent (lfd-gated): idx=$best_idx GT=$(new_best.GT) Delta=$(new_best.Delta) lfd_ok=$(new_best.lfd_ok)")
    state.most_extreme_idx = best_idx

    # Also re-tighten state.bracket's feasible edge to the lfd-gated incumbent, and its infeasible
    # edge to the nearest (tightest) known infeasible/over-budget point beyond it, so a future
    # resume of this chain continues narrowing the TRUE bracket instead of re-deriving the stale
    # pre-fix bracket or (worse) treating the degenerate point as still feasible.
    infeas_idx = 0
    infeas_gap = Inf
    for (i, pt) in enumerate(state.points)
        is_infeasible = !(pt.classification == :FiniteSolved && pt.within_budget && pt.lfd_ok)
        is_infeasible || continue
        sign * (pt.GT - new_best.GT) > 0 || continue   # must be beyond (more extreme than) new_best
        gap = abs(pt.GT - new_best.GT)
        if gap < infeas_gap
            infeas_gap = gap
            infeas_idx = i
        end
    end
    @assert infeas_idx != 0 "[$chainid] no infeasible point found beyond the lfd-gated incumbent"
    infeas_pt = state.points[infeas_idx]
    println("[$chainid] tightest known infeasible edge beyond new incumbent: idx=$infeas_idx GT=$(infeas_pt.GT) Delta=$(infeas_pt.Delta) cls=$(infeas_pt.classification)")
    state.bracket = (new_best.GT, new_best.Delta, infeas_pt.GT, infeas_pt.Delta)

    tmp = ckpt * ".tmp"
    serialize(tmp, state)
    mv(tmp, ckpt; force=true)
    println("[$chainid] checkpoint updated (most_extreme_idx + bracket).\n")
end
println("DONE LFD GATE FIX")

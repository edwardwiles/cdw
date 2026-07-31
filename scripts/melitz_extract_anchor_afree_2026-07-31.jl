# Extract just the A_free vector (a plain Vector{Float64}) from an ORIGINAL production
# checkpoint, in an isolated process where ONLY the production struct names are defined. This
# avoids the Julia Serialization type-name collision that occurs when a script also defines a
# DIFFERENT struct sharing the literal name "ChainPoint"/"ChainState" (deserialize resolves types
# by name only, not structural layout, so whichever definition is loaded last wins for every
# deserialize call of that name -- confirmed live, 2026-07-31, in
# melitz_overnight_final_verify_2026-07-31.jl).
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

length(ARGS) >= 2 || error("usage: julia melitz_extract_anchor_afree_2026-07-31.jl <ckpt_path> <out_path>")
ckpt_path, out_path = ARGS[1], ARGS[2]
state = deserialize(ckpt_path)
A_free = state.points[state.most_extreme_idx].A_free
serialize(out_path, A_free)
println("Extracted A_free (length ", length(A_free), ") from ", ckpt_path, " -> ", out_path)

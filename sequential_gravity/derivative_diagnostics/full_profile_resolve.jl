# ============================================================================
# Part 3: fully re-solved profile finite differences (validation-only,
# expensive -- re-solves the exact hard inner CC problem via KNITRO at each
# perturbed A). No fixed-x approximation, no envelope theorem, no boundary
# formula -- this is the ground truth Part 1/2 are checked against.
# ============================================================================

"""
    full_profile_directional_slope(θbase, γobj, U, d, moments_fn, v, h; Acol_offset=3)

δ*(A⊙e^{hv},γ') and δ*(A⊙e^{-hv},γ'), each via a FRESH, cold-started (no
warm start -- use_cached_x defaults false, matching `recover_lfd`'s own
convention) `inner_loop` KNITRO solve of the exact hard `d`-moment CC
problem using `moments_fn`. Generic in `d`/`moments_fn` so the SAME driver
works for the reduced (D+1)-moment diagnostic problem (moments_fn =
EK_moments_focal_norm_directgp!, d=D+1) and the actual production
(D+2)-moment problem (moments_fn = a frozen-gravity moments function from
full_fixed_dual_criterion.jl, d=D+2) -- per Part 4's requirement that the
"fully re-solved profile slope" comparison run on the REAL sequential
objective, with the SAME frozen gravity linearization held fixed across both
re-solves (achieved simply by using the same `moments_fn` closure, which
freezes lastθ/lastRcol/gcol/dRdθ internally, for both the + and - solves).
Returns (slope, δp, δm, statusp, statusm).
"""
function full_profile_directional_slope(θbase::Vector{Float64}, γobj, U::Matrix{Float64}, d::Int,
        moments_fn::Function, v::Vector{Float64}, h::Real; Acol_offset::Int=3)
    D = length(v)
    θp = copy(θbase); θm = copy(θbase)
    @views θp[Acol_offset+1:Acol_offset+D] .*= exp.(h .* v)
    @views θm[Acol_offset+1:Acol_offset+D] .*= exp.(-h .* v)
    objp = build_fixed_dual_bundle(γobj, U, length(θbase), d, moments_fn)
    objm = build_fixed_dual_bundle(γobj, U, length(θbase), d, moments_fn)
    δp, _, statusp = inner_loop(objp, θp)
    δm, _, statusm = inner_loop(objm, θm)
    slope = (δp - δm) / (2h)
    return slope, δp, δm, statusp, statusm
end

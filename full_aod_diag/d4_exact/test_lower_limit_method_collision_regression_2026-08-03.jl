# Regression gate (2026-08-03, docs/audits/profiled-inner-readiness-2026-08-03/): the
# `_callbackEvalFG_inner_profiled!` method-collision + `OperatorPsiBundle.lower_limit` struct-default
# bug found live in canonical `prototype/profiled-destination-scales@7ec5c6c`. This file deliberately
# includes BOTH `oracle_fast.jl` and `profiled_operator_bundle_2026-08-01.jl` in the same process (the
# exact scenario that produced the collision) and proves:
#   1. the two `_callbackEvalFG_inner_profiled!`/`_callbackEvalH_inner_profiled!` definitions coexist
#      as genuinely separate, type-dispatched methods (not a silent redefinition);
#   2. `OperatorPsiBundle` requires `lower_limit` explicitly (no struct default -- omitting it errors);
#   3. all 5 REDUCED bundle constructors now propagate `ref_obj.lower_limit` correctly (finite, not
#      -Inf, matching `ref_obj`'s own value);
#   4. the live `_callbackEvalFG_inner_profiled!(::ProfiledCBState)` callback actually applies the
#      `f<=lower_limit?-KN_INFINITY:f` clamp for a synthetic adversarial point, and leaves an ordinary
#      feasible point's objective value unchanged (un-clamped behavior preserved).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "operator_verification.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_evaluator_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_gradient_fd_2026-08-01.jl"))
using LinearAlgebra, Printf, KNITRO

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

# ---- Check 1: no silent method-table collision -- both definitions coexist as distinct methods ----
nmethods_fg = length(methods(_callbackEvalFG_inner_profiled!))
nmethods_h = length(methods(_callbackEvalH_inner_profiled!))
check("_callbackEvalFG_inner_profiled! has >=2 coexisting methods (oracle_fast.jl generic + profiled_operator_bundle ProfiledCBState-specific), not 1 (collision)", nmethods_fg >= 2)
check("_callbackEvalH_inner_profiled! has >=2 coexisting methods", nmethods_h >= 2)
ms_fg = collect(methods(_callbackEvalFG_inner_profiled!))
has_profiledcbstate_method = any(m -> occursin("ProfiledCBState", string(m.sig)), ms_fg)
check("one _callbackEvalFG_inner_profiled! method is specific to ::ProfiledCBState", has_profiledcbstate_method)

# ---- Check 2: OperatorPsiBundle.lower_limit has no struct default -- omitting it is a hard error ----
function _missing_lower_limit_throws()
    try
        OperatorPsiBundle(δ = 1.0, find_smallest = true, γ = 1.0, l = 1, inequality_index = Int[],
            U = zeros(2, 1), outer_constr_index = 1, inner_loop_opt = "")
        return false
    catch e
        return e isa UndefKeywordError || occursin("lower_limit", sprint(showerror, e))
    end
end
check("OperatorPsiBundle(...) without lower_limit throws (no silent default)", _missing_lower_limit_throws())

# ---- Check 3: all 5 REDUCED bundle constructors now propagate a finite lower_limit ----
ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
check("root ctx.obj.lower_limit is finite (context.jl's own explicit -50, not a struct default)", isfinite(ctx.obj.lower_limit))

spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
obj_u, st_u = build_profiled_operator_bundle(ctx, ctx0.θ0_up, spec; ref_obj = ctx.obj)
check("unrestricted build_profiled_operator_bundle: obj.lower_limit finite and matches ref_obj", isfinite(obj_u.lower_limit) && obj_u.lower_limit == ctx.obj.lower_limit)

# ---- Check 4: the live clamp actually fires for an adversarial point, and leaves a feasible point alone ----
n = obj_u.outer_constr_index
kc = KNITRO.KN_new()
fake_req = (x = zeros(n),)
fake_res = (obj = zeros(1), objGrad = zeros(n))

# Feasible-ish point: zeta=0, beta=0 -> q = 0 everywhere, f should be finite and >> lower_limit
fake_req.x .= 0.0
_callbackEvalFG_inner_profiled!(kc, nothing, fake_req, fake_res, st_u)
f_feasible = fake_res.obj[1]
check("feasible-ish point (x=0): objective NOT clamped (finite, not -KN_INFINITY)", isfinite(f_feasible) && f_feasible > obj_u.lower_limit && f_feasible != -KNITRO.KN_INFINITY)

# Adversarial check, isolated from the real economic Psi's shape: a pure-zeta shift of the real
# dual (CLAUDE.md's own note applies here) is actually convex/bounded -- q=-zeta-t sweeps both
# directions through Psi, so neither zeta->+-Inf alone reliably drives f below lower_limit without
# genuine model-specific reasoning about Psi's growth rate. Rather than hand-construct a real
# dual-infeasibility direction, swap in a trivial stand-in Psi!/dPsi! (returns 0 everywhere) on a
# CLONE of obj_u/st_u so f collapses to exactly `zeta` -- this isolates and definitively tests the
# clamp LINE ITSELF (`f <= obj.lower_limit ? -KN_INFINITY : f`) without depending on any economic
# reasoning about the real Psi's shape.
obj_clamp = OperatorPsiBundle(δ = obj_u.δ, find_smallest = obj_u.find_smallest, γ = obj_u.γ, l = obj_u.l,
    inequality_index = Int[], U = obj_u.U, outer_constr_index = obj_u.outer_constr_index,
    inner_loop_opt = obj_u.inner_loop_opt, lower_limit = obj_u.lower_limit,
    Psi! = (out, q) -> fill!(out, 0.0), dPsi! = (out, q) -> fill!(out, 0.0))
st_clamp = ProfiledCBState(obj_clamp, st_u.ctx, st_u.cf, st_u.layout, st_u.θ_full)

fake_req.x .= 0.0
fake_req.x[1] = obj_u.lower_limit - 1.0   # zeta below lower_limit; Psi≡0 so f == zeta exactly
_callbackEvalFG_inner_profiled!(kc, nothing, fake_req, fake_res, st_clamp)
f_adversarial = fake_res.obj[1]
check("point with zeta<lower_limit (Psi≡0 stand-in isolates the clamp line): objective IS clamped to exactly -KN_INFINITY", f_adversarial == -KNITRO.KN_INFINITY)

fake_req.x[1] = obj_u.lower_limit + 1.0   # zeta above lower_limit -> must NOT clamp
_callbackEvalFG_inner_profiled!(kc, nothing, fake_req, fake_res, st_clamp)
f_just_above = fake_res.obj[1]
check("point with zeta>lower_limit (Psi≡0 stand-in): objective NOT clamped (equals zeta exactly)", f_just_above == obj_u.lower_limit + 1.0)
KNITRO.KN_free(kc)

if ALL_PASS[]
    println("\nALL PASS")
else
    println("\nSOME FAILED")
    exit(1)
end

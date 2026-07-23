# Aggregator include for the full-D Melitz Christensen-Connault module, matching this
# repo's own per-directory convention (cc_algo/include_cc_algo.jl,
# moments/include_moments.jl, setup/include_setup.jl, ...). doubleDiff (misc/doubleDiff.jl)
# must be included separately first.
#
# `delta_star.jl` is now included BEFORE `fake_data.jl`/`fstar_solver.jl` -- both need its
# `melitz_outer_layout`/`build_gravity_pivots`/`lin2od`/gravity-pivot machinery to
# construct a gravity-feasible fixture. This is safe WITHOUT `cc_algo`'s
# `CounterfactualSensitivity` module loaded: `delta_star.jl`'s KNITRO-touching functions
# (`build_melitz_psi_bundle`, `melitz_recover_lfd`, `run_melitz_inner_delta`,
# `melitz_moments_adapter!`) reference `PsiObjectiveBundleDelta`/`inner_loop`/`dPsi!` only
# inside function bodies (resolved at CALL time, not parse/include time) -- only calling
# them (not merely defining them) requires `cc_algo/include_cc_algo.jl` +
# `using .CounterfactualSensitivity` to have been loaded first.

include("types.jl")
include("pareto.jl")
include("firm_quantities.jl")
include("equilibrium.jl")
include("moments.jl")
include("delta_star.jl")
include("fake_data.jl")
include("fstar_solver.jl")
include("fstar_direct.jl")
include("gradient_lab.jl")
include("outer_solve.jl")

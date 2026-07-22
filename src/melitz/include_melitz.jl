# Aggregator include for the full-D Melitz Christensen-Connault module, matching this
# repo's own per-directory convention (cc_algo/include_cc_algo.jl,
# moments/include_moments.jl, setup/include_setup.jl, ...). doubleDiff (misc/doubleDiff.jl)
# and, for delta_star.jl, cc_algo/include_cc_algo.jl (+ `using .CounterfactualSensitivity`)
# must be included/loaded separately first -- see docs/melitz_delta_star.md Section 13 or
# scripts/run_melitz_delta_star_fake.jl for the exact order.

include("types.jl")
include("pareto.jl")
include("firm_quantities.jl")
include("equilibrium.jl")
include("fake_data.jl")
include("moments.jl")
include("fstar_solver.jl")
include("delta_star.jl") # requires cc_algo's CounterfactualSensitivity module to be `using`'d first

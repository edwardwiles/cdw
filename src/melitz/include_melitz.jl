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

include("profiling.jl")
include("knitro_compat.jl")
include("backend_config.jl")
include("run_diagnostics.jl")
include("types.jl")
include("bounded_cache.jl")
include("inner_solve_policy.jl")
include("pareto.jl")
include("firm_quantities.jl")
include("equilibrium.jl")
include("moments.jl")
include("sorted_tail.jl")
include("sorted_dual_argument.jl")
include("moment_operator.jl")
include("delta_star.jl")
include("affine_cutoff.jl")
include("log_cutoff_param.jl")
include("technology_coordinate.jl")
include("outer_parameterization_config.jl")
include("fake_data.jl")
include("pareto_calibration.jl")
include("fstar_solver.jl")
include("fstar_direct.jl")
include("gradient_lab.jl")
include("outer_solve.jl")
include("inner_screening.jl")
include("inner_session.jl")
include("origin_block_screen.jl")
include("localized_gradient.jl")
include("argument_localized_gradient.jl")
include("direct_gradient.jl")
include("sorted_crossing_gradient.jl")
include("touched_row_gradient.jl")
include("cc_bundle.jl")
include("exact_a_gradient.jl")
include("exact_q_smooth_gradient.jl")
include("q_bandwidth_policy.jl")
include("aq_experimental_backend.jl")
include("finite_delta_outer.jl")
include("reduced_q_subspace.jl")
include("reduced_q_switch_geometry.jl")
include("reduced_q_controller.jl")
include("typed_eval_counters.jl")
include("matched_effort_controller.jl")
include("reduced_q_threaded_direction.jl")
include("nuisance_profile.jl")
include("predictor_corrector.jl")
include("lfd_preserving_state.jl")

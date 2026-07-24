module CounterfactualSensitivity

using KNITRO
using LinearAlgebra
using ForwardDiff
using Parameters
using Calculus
using Distributions
using Dates
using DelimitedFiles
using JLD2

export outer_loop,
       inner_loop,
       local_sensitivity,
       KLObjectiveBundle,
       PsiObjectiveBundle,
       KLObjectiveBundleConditional,
       PsiObjectiveBundleConditional,
       KLObjectiveBundleExplicit,
       KLObjectiveBundleImplicit,
       KLObjectiveBundleDelta,
       PsiObjectiveBundleExplicit,
       PsiObjectiveBundleImplicit,
       PsiObjectiveBundleDelta,
       master_cc_algo,
       dPsi!,
       FreeParamMap,
       n_free,
       pack_free,
       reconstruct_full,
       reconstruct_full!,
       pack_bounds_free,
       round_trip_check,
       OuterEvalCache,
       ensure_inner!,
       ensure_grad!,
       summarize,
       write_trace_csv,
       outer_loop_cached,
       reset_jac_h_counters!,
       jac_h_counters_snapshot,
       CertifiedDivergenceLowerBound,
       ThresholdAbortState,
       threshold_abort_result,
       resolve_threshold_for_delta,
       threshold_permits_reject

include("knitro_compat.jl")   # restore KNITRO.jl 0.13/0.14 convenience wrappers on v1.2.1
include("Psi.jl")
include("threshold_early_abort.jl")   # Part C: must precede PsiObjectiveBundle.jl (adds a field to Implicit)
include("ObjectiveBundle.jl")
include("jac_h_instrumentation.jl")   # additive counters/timers + no-jac_h default helpers, must precede PsiObjectiveBundle.jl
include("KLObjectiveBundle.jl")
include("PsiObjectiveBundle.jl")
include("inner_loop_functions.jl")
include("outer_loop_functions.jl")
include("free_param_map.jl")
include("outer_eval_cache.jl")
include("outer_loop_cached.jl")
include("local_sensitivity.jl")
include("ccOuter.jl")
include("ccInner.jl")
include("master_cc_algo.jl")

end
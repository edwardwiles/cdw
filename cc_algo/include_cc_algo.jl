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
       dPsi!

include("knitro_compat.jl")   # restore KNITRO.jl 0.13/0.14 convenience wrappers on v1.2.1
include("Psi.jl")
include("ObjectiveBundle.jl")
include("KLObjectiveBundle.jl")
include("PsiObjectiveBundle.jl")
include("inner_loop_functions.jl")
include("outer_loop_functions.jl")
include("local_sensitivity.jl")
include("ccOuter.jl")
include("ccInner.jl")
include("master_cc_algo.jl")

end
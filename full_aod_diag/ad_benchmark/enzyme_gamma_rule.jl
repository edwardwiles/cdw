# Custom EnzymeRules.@easy_rule for SpecialFunctions.gamma, bypassing Enzyme's built-in
# low-level "known_ops" path (EnzymeSpecialFunctionsExt.jl's known_ops[_logabsgamma] =
# (:logabsgamma, 1, (:digamma, ...))), whose JIT-time symbol resolution for `digamma` is
# broken in this environment (Enzyme.jl issue #2890, confirmed open, exact match). This
# rule is a plain Julia function (gamma(x)*digamma(x), the standard closed-form derivative,
# both cross-checked against finite differences) — it needs no low-level symbol at all, so
# it sidesteps the broken path entirely rather than fixing it. Verified working in isolation
# (sandbox environment, exact match to analytical + finite-difference derivative) BEFORE
# being loaded here. Include this file (once) before any Enzyme.autodiff call that
# differentiates through SpecialFunctions.gamma.
using Enzyme, SpecialFunctions
using Enzyme.EnzymeRules

EnzymeRules.@easy_rule(
    SpecialFunctions.gamma(x::AbstractFloat),
    @setup(dx = Ω * SpecialFunctions.digamma(x)),
    (dx,),
)

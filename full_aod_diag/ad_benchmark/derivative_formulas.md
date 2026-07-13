# Derivative formulas (§3-4)

## Canonical moment map

```
moment_map(θ; U, γobj) -> H,   H ∈ R^{N×(d+2)}
H[:,1] = K(θ)      (the objective; under this session's variant, K ≡ θ[3+D] = γ'_focal DIRECTLY)
H[:,2] = 1         (constant column, structural — never depends on θ)
H[:,3:end] = G(θ)  (d = 18 moments: 16 trade shares, 1 counterfactual price index, 1 gravity)
```

Implemented in `derivative_core.jl::moment_map!` as a thin wrapper around
`full_aod_diag/moments_gammanorm.jl::EK_moments_gammanorm_directgp!` — byte-identical primal
values to production (verified, see `correctness_results.csv`).

## Envelope scalar (the object whose gradient is actually needed)

For the **divergence-budget constraint** (`constr[1]` in `PsiObjectiveBundleImplicit`), with
`λ` = the current inner-solve dual multipliers (`x[2:end]`, fixed) and `arg1` = the current
per-draw Ψ-derivative weight (fixed — both are exactly the quantities the envelope theorem holds
constant while differentiating w.r.t. θ; production performs NO further differentiation through
them for this particular constraint, confirmed by inspection: `PsiObjectiveBundle.jl`'s `ift!`
correction touches only `∂c_∂θ[2:end,:]`, never row 1):

```
envelope_scalar_div(θ) = (1e10/N) · Σ_{draw=1}^{N} arg1[draw] · Σ_{j=1}^{outer_constr_index-1} λ[j]·G[draw,j](θ)
```

`outer_constr_index - 1 = 17` (the inner-matched moments; the 18th, gravity, is excluded — it's
the *other* outer constraint). Implemented in `derivative_core.jl::envelope_scalar_div_ctx`.

**This is not a guess** — `∇_θ envelope_scalar_div(θ) = ∂c_∂θ[1,:]` was verified against
production's actual value (computed via the real `calculate_jac_θ!` + contraction path, i.e.
Method A exactly as KNITRO's callback invokes it) at all 4 benchmark points, relative error
~1e-15 (`quick_validate.jl`, `validate_derivatives.jl`).

### Which inner variables are fixed under the envelope theorem?

`ζ` (implicit in `arg0`/`arg1` via `dPsi!`) and `λ` — the CURRENT inner-solve solution `x`, not
differentiated. This is valid for `constr[1]` because, unlike the gravity constraint, it has no
`ift!`-based total-derivative correction in the source — i.e. production itself already treats
it as a pure envelope scalar; this audit reproduces, not reinterprets, that structure.

### Gravity constraint (`constr[2]`) — explicitly out of scope for the AD-backend comparison

Its gradient is `envelope term (as above, own G-column) + ift!-based correction through ∂x*/∂θ`.
The `ift!` correction was **not** re-derived as a scalar/reverse-mode object in this audit — see
`call_graph_audit.md` §5 and the Non-goals note in the final report. The spec's own instruction
("do not replace an exact analytic gravity derivative with AD merely for uniformity") is honored
by leaving it untouched.

## θ layout (this session's γ_d≡1 + direct-γ' variant, D=4, l=23)

```
θ[1]      = μ
θ[2]      = σ (bounds-pinned, NOT free in the outer search, but a genuine argument of the moment map)
θ[3:6]    = γ_θ[1..4]   -- INERT under this variant (EK_moments_gammanorm_directgp! ignores them; γ≡1 hardcoded)
θ[7]      = γ'_focal DIRECT (K = θ[7] exactly; gradient of K w.r.t. θ = unit vector e_7)
θ[8:23]   = A_od, column-major: idx = 7 + o + (d-1)*4,  o,d ∈ 1:4  (ALL free, no A[1,d]=1 pins)
```

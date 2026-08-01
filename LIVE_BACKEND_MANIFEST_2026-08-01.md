# Live backend manifest — profiled_destination_scales bundle (2026-08-01)

Confirmed by direct construction and inspection during the D4 KNITRO gates (this session):

```
bundle_type                         = OperatorPsiBundle{...}  (Main.OperatorPsiBundle, operator_psi_bundle.jl)
economic_parameterization           = profiled_destination_scales  (new, additive; NOT the production default)
production_default_parameterization = full_gamma_normalized_reference  (UNCHANGED)
factual_moment_count (D4)           = 12  (= D*Ddest - Ddest = 16 - 4)
factual_moment_count (D20 exclude_row) = 361  (= D*Ddest - Ddest = 380 - 19)
france_ratio_count                  = 1
anchor_moment_count_present         = 0  (no beta/dual slot ever allocated for an anchor cell --
                                           full_factual_to_reduced[anchor_j] == 0 for every anchor,
                                           and no reduced index maps back to one; verified by
                                           test_profiled_economic_moment_layout_2026-08-01.jl)
total_reduced_economic_moments (D4) = 13   (outer_constr_index = 14 = 1 + 13)
total_reduced_economic_moments (D20)= 362  (outer_constr_index = 363 = 1 + 362)
dense_construction_count            = 0   (static_bundle_guard_2026-07-30.sh: 0 violations, run against
                                            the full session's added files; grep confirms no
                                            PsiObjectiveBundleImplicit/H/H_copy/G/K/moments!/
                                            select_G_from_H reference in any non-test file added
                                            this session)
inner_solve_driver                  = inner_loop_KNITRO_profiled (profiled_operator_bundle_2026-08-01.jl),
                                       a faithful reduced-dimension sibling of the production
                                       inner_loop_KNITRO_compressed (compressed_live.jl) -- same
                                       KN_add_vars/KN_set_var_lobnds_all/KN_set_var_primal_init_values_all/
                                       KN_add_eval_callback/KN_load_param_file/KN_set_cb_hess/KN_solve
                                       sequence, same option file (ek_inner.opt), only the two
                                       callbacks and the dual-vector dimension differ.
FG_callback                          = _callbackEvalFG_inner_profiled! -> reduced_homogeneous_dual_contraction /
                                        reduced_homogeneous_transpose_contraction! (task §6/§7 kernels)
Hessian_callback                     = _callbackEvalH_inner_profiled! -> reduced_homogeneous_winner_pair_hessian!
                                        (task §8 kernel), built from ReducedHomogeneousWinnerPairHessCtx
verification                         = verify_inner_solution_reduced_profiled! (reduced_operator_verification_2026-08-01.jl),
                                        feeds the SAME verify_namedtuple_from_operator (operator_verification.jl)
                                        the production reference path uses -- same fields, same formulas.
```

No `obj.H`, `obj.moments!`, `obj.H_copy`, `obj.K`, `obj.ones` field access anywhere in the profiled
path (structurally impossible: `OperatorPsiBundle` has no such fields). `select_G_from_H` is never
called on this bundle type.

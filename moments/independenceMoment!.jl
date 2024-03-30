function independenceMoment!(Ū, G, PMM, D, W, μ_σ, ν, ηk, momentOrder, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, sameMarginalsMoment)
    # assures the marginals are identical and independent
    # old implementation not used
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (3 + D^2) + sameMarginalsMoment * (2 + momentOrder) * D^2

    νk = zeros(D, D, momentOrder + 2)
    for o = 1:D
        for d = 1:D
            for k = 1:momentOrder
                νk[o, d, k] = ν[o, d]^k
            end
            νk[o, d, momentOrder+1] = ν[o, d]^μ_σ
            νk[o, d, momentOrder+2] = log(ν[o, d])
        end
    end

    refIndex = 1
    refIndex1 = refIndex + (refIndex - 1) * D
    for ω = 1:W

        # E[U(ref,ref)^k/k!] = ηk
        G[ω, end-offset-k] = Ū[ω, refIndex1, 1] / νk[refIndex, refIndex, 1] - ηk[1]
        for k = 2:momentOrder
            G[ω, end-offset-k] = (Ū[ω, refIndex1, k] / νk[refIndex, refIndex, k] - Ū[ω, refIndex1, k-1] / νk[refIndex, refIndex, k-1]) - (ηk[k] - ηk[k-1])
        end

        idx_corss_moment = 0
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for c = 1:D
                    if c < o
                        c1 = c + (d - 1) * D # uncomment to to U_{od} rather than U_o
                        idx_corss_moment += 1
                        G[ω, end-offset-momentOrder-idx_corss_moment] = (Ū[ω, o1, 1] / νk[o, d, 1]) * (Ū[ω, c1, 1] / νk[c, d, 1]) - (ηk[1]) * (ηk[1])
                        for m = 2:momentOrder

                            #idx_corss_moment = (d-1)*(D*(D-1)*momentOrder*momentOrder)/2 + (o-1)*(o-1)*momentOrder*momentOrder+ (c-1)*momentOrder*momentOrder + (k-1)*momentOrder + m

                            # E[Uod Ucd^m/m!] = ηodk*ηcdm
                            idx_corss_moment += 1
                            G[ω, end-offset-momentOrder-idx_corss_moment] = (Ū[ω, o1, 1] / νk[o, d, 1]) * (Ū[ω, c1, m] / νk[c, d, m] - Ū[ω, c1, m-1] / νk[c, d, m-1]) - (ηk[1]) * (ηk[m] - ηk[m-1])
                        end
                    end
                end


                for k = 2:momentOrder
                    for c = 1:D
                        if c > o
                            c1 = c + (d - 1) * D # uncomment to to U_{od} rather than U_o
                            idx_corss_moment += 1
                            G[ω, end-offset-momentOrder-idx_corss_moment] = (Ū[ω, o1, k] / νk[o, d, k] - Ū[ω, o1, k-1] / νk[o, d, k-1]) * (Ū[ω, c1, 1] / νk[c, d, 1]) - (ηk[k] - ηk[k-1]) * (ηk[1])
                            for m = 2:momentOrder
                                #idx_corss_moment = d*(D*momentOrder) + (d-1)*(D*D*momentOrder*momentOrder) + (o-1)*D*momentOrder*momentOrder+ (c-1)*momentOrder*momentOrder + (k-1)*momentOrder + m
                                # E[Uod^k/k! Ucd^m/m!] = ηodk*ηcdm
                                idx_corss_moment += 1
                                G[ω, end-offset-momentOrder-idx_corss_moment] = (Ū[ω, o1, k] / νk[o, d, k] - Ū[ω, o1, k-1] / νk[o, d, k-1]) * (Ū[ω, c1, m] / νk[c, d, m] - Ū[ω, c1, m-1] / νk[c, d, m-1]) - (ηk[k] - ηk[k-1]) * (ηk[m] - ηk[m-1])
                            end
                        end
                    end
                end
            end
        end
    end
end
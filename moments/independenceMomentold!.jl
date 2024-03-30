function independenceMomentold!(Ū, G, PMM, D, W, μ_σ, ν, ηk, momentOrder, gravMoment, localGravityMoment, GravityMomentFirstApproach, localGravityCrossMoment, sameMarginalsMoment)
    # assures the marginals are identical and independent,
    # old implementation not used
    offset = gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (3 + D^2) + sameMarginalsMoment * (2 + momentOrder) * D^2
    for ω = 1:W
        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for k = 1:momentOrder
                    idx_moment = (d - 1) * (D * momentOrder + D * D * momentOrder * momentOrder) + (o - 1) * momentOrder + k
                    G[ω, end-offset-idx_moment] = Ū[ω, o1]^k / factorial(k) - ηk[k] # E[Uod^k/k!] = ηodk
                    for c = 1:D
                        if c > o
                            c1 = c + (d - 1) * D # uncomment to to U_{od} rather than U_o
                            for m = 1:momentOrder
                                idx_corss_moment = d * (D * momentOrder) + (d - 1) * (D * D * momentOrder * momentOrder) + (o - 1) * D * momentOrder * momentOrder + (c - 1) * momentOrder * momentOrder + (k - 1) * momentOrder + m
                                # E[Uod^k/k! Ucd^m/m!] = ηodk*ηcdm
                                G[ω, end-offset-idx_corss_moment] = (Ū[ω, o1]^k) * (Ū[ω, c1]^m) / (factorial(k) * factorial(m)) - ηk[k] * ηk[m]
                            end
                        end
                    end
                end
            end
        end
    end
end
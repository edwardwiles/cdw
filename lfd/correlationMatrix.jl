   function correlationMatrix(i, up_down)

        U_Means = zeros(D^2)
        U_Vars = zeros(D^2)
        corrMatrix = zeros(D^2, D^2)
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) *StrataWeight[ω]/ W
        end


        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                for ω = 1:W

                    U_Means[o1] += Ū[ω, o1, 1] * (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) *StrataWeight[ω]/ (W * nornamization_factor)
                    U_Vars[o1] += Ū[ω, o1, 1] * Ū[ω, o1, 1] * (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) *StrataWeight[ω]/ (W * nornamization_factor)
                end
            end
        end

        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o 
                U_Vars[o1] = U_Vars[o1] - U_Means[o1] * U_Means[o1]

            end
        end

        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o               
                for c = 1:D
                    for f = 1:D
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        if o1 > c1
                            for ω = 1:W
                                corrMatrix[o1, c1] += (Ū[ω, o1, 1]) .* (Ū[ω, c1, 1]) * (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) *StrataWeight[ω]/ (W * nornamization_factor) - U_Means[o1] * U_Means[c1]*StrataWeight[ω]/ W
                            end
                        end
                    end
                end
            end
        end

        for d = 1:D
            for o = 1:D
                o1 = o + (d - 1) * D # uncomment to to U_{od} rather than U_o               
                for c = 1:D
                    for f = 1:D
                        c1 = c + (f - 1) * D # uncomment to to U_{od} rather than U_o
                        if o1 > c1
                            corrMatrix[o1, c1] = corrMatrix[o1, c1] / sqrt(U_Vars[o1] * U_Vars[c1])
                        end
                    end
                end
            end
        end
        return corrMatrix
    end
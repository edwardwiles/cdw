    function JointPricesCDF(x, d, i, up_down)
        o1 = 1 + (d - 1) * D
        o2 = D + (d - 1) * D
        jcdf = zeros(length(x), length(x))
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i]))*StrataWeight[ω] / W
            x_ω = Ū[ω, o1:o2, 1]
            price_baseIndex = x_ω[d] / λData[d, d]
            price_rw = minimum(x_ω[Not(d)] ./ λData[Not(d), d])
            for i1 = 1:length(x)
                for i2 = 1:length(x)
                    jcdf[i1, i2] += (price_baseIndex < x[i1] && price_rw < x[i2]) ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i]))*StrataWeight[ω] / W : 0
                end
            end
        end
        return jcdf / nornamization_factor
    end
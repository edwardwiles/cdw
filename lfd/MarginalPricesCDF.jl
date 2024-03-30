    function MarginalPricesCDF(x, d, i, up_down)
        o1 = 1 + (d - 1) * D
        o2 = D + (d - 1) * D
        pcdf = zeros(length(x), 3)
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) *StrataWeight[ω]/ W
            for j = 1:length(x)
                x_ω = Ū[ω, o1:o2, 1]
                price_baseIndex = x_ω[d] / λData[d, d]
                price_rw = minimum(x_ω[Not(d)] ./ λData[Not(d), d])
                pcdf[j, 1] += price_baseIndex < x[j] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) *StrataWeight[ω]/ W : 0
                pcdf[j, 2] += price_rw < x[j] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) *StrataWeight[ω]/ W : 0
                pcdf[j, 3] += price_baseIndex / price_rw < x[j] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i]))*StrataWeight[ω] / W : 0
            end
        end
        return pcdf / nornamization_factor
    end
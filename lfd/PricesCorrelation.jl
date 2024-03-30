   function PricesCorrelation(d, i)
        o1 = 1 + (d - 1) * D
        o2 = D + (d - 1) * D
        pmean = zeros(3, 2)
        pcorr = zeros(3)
        for ω = 1:W
            x_ω = Ū[ω, o1:o2, 1]
            price_baseIndex = x_ω[d] / λData[d, d]
            price_rw = minimum(x_ω[Not(d)] ./ λData[Not(d), d])

            pmean[1, 1] += price_baseIndex *StrataWeight[ω]/ W
            pmean[2, 1] += LFD_upper[ω, i] * price_baseIndex *StrataWeight[ω]/ W
            pmean[3, 1] += LFD_lower[ω, i] * price_baseIndex *StrataWeight[ω]/ W

            pmean[1, 2] += price_rw *StrataWeight[ω]/ W
            pmean[2, 2] += LFD_upper[ω, i] * price_rw *StrataWeight[ω]/ W
            pmean[3, 2] += LFD_lower[ω, i] * price_rw *StrataWeight[ω]/ W

            pcorr[1] += price_rw * price_baseIndex *StrataWeight[ω]/ W
            pcorr[2] += LFD_upper[ω, i] * price_rw * price_baseIndex *StrataWeight[ω]/ W
            pcorr[3] += LFD_lower[ω, i] * price_rw * price_baseIndex *StrataWeight[ω]/ W
        end

        pcorr[1] -= pmean[1, 1] * pmean[1, 2]
        pcorr[2] -= pmean[2, 1] * pmean[2, 2]
        pcorr[3] -= pmean[3, 1] * pmean[3, 2]

        return (pcorr, pmean)
    end
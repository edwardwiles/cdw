function computeGamma(c, tau, w, theta, sigma, L)
    # computes gamma and gamma prime, see theory note 
    Phi = sum(c .* (tau .* w) .^ (-theta), dims=1) # added parentesis just to be sure
    priceIndex = (Phi .^ (-(1 - sigma) / theta)) .* gamma((1 + theta - sigma) / theta)
    gammaHat = (priceIndex' ./ (w .* L)) .^ (1 / sigma)
    return gammaHat
end
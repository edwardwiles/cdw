function createTradeCosts!(tradeCosts, scale, D)
    # Construct trade costs based on distance if using fake data 
    # generate random x and y coords for each country, then compute distances, then scale to trade costs 
    params = rand(D, 2) .* 5
    coordMat = zeros(D * D, 5)
    coordMat[:, 1] = repeat(params[:, 1], inner=D)
    coordMat[:, 2] = repeat(params[:, 2], inner=D)
    coordMat[:, 3] = repeat(params[:, 1], outer=D)
    coordMat[:, 4] = repeat(params[:, 2], outer=D)
    coordMat[:, 5] = exp.(0.1 * sqrt.(((coordMat[:, 1] - coordMat[:, 3]) .^ 2 + (coordMat[:, 2] - coordMat[:, 4]) .^ 2)))
    tradeCosts[:, :] = reshape(coordMat[:, 5], (D, D)) .^ scale # convert distance to trade costs (scale is a param)
    nothing
end
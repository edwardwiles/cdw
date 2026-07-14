function runLFD(lfd_output, cc_output, prep_output, prestep_output, setup_output, params)
	# this function takes the LFD RN derivative and creates PDF/ CDF and correlation graphs
	# U realizations are not saved, becuase they are potentially large files, so we re-generate them here assuming we are using the same seed etc.
	@unpack W, baseIndex, OuterScaling, counterType, independenceMoment, sameMarginalsMoment, GravityMomentFirstApproach = params
	@unpack data = setup_output
	@unpack θ_initial, γ, δ_grid, file_name = prep_output
	@unpack Θ_upper, κ_upper, Θ_lower, κ_lower = cc_output
	@unpack LFD_upper, LFD_lower, δ_LFD_upper, δ_LFD_lower = lfd_output

	D = length(γ.L)
	δ_grid_size = length(δ_grid)


	U = γ.Ū # unscaled U
	cHat = γ.cHat
	λData = data.λData
	wHat = γ.wHat
	τ = γ.τ
	# Aod
	Aod = ones(3, δ_grid_size, D, D) # up/initial/low, δ, o, d

	if OuterScaling == 1 # Aod model
		counterType_θ_offset = 0
		if counterType != 1
			counterType_θ_offset = 2 * (D - 1) # D-1 wagesPrime and D-1 gamma_primes
		end

		Aod_offset = counterType_θ_offset + 3 + D
		if independenceMoment == 1
			Aod_offset += 1
		end
		for i ∈ 1:length(δ_grid)
			Aod[1, i, :, :] = reshape(Θ_lower[Aod_offset+1:Aod_offset+D^2, i], (D, D))
			Aod[2, i, :, :] = reshape(θ_initial[Aod_offset+1:Aod_offset+D^2], (D, D))
			Aod[3, i, :, :] = reshape(Θ_upper[Aod_offset+1:Aod_offset+D^2, i], (D, D))
		end
	end


	for i ∈ 1:length(δ_grid)
		@. Aod[1, i, :, :] = Aod[1, i, :, :] * (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / Θ_lower[1])) .* (λData ./ λData[1, :]')
		@. Aod[2, i, :, :] = Aod[2, i, :, :] * (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / θ_initial[1])) .* (λData ./ λData[1, :]')
		@. Aod[3, i, :, :] = Aod[3, i, :, :] * (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / Θ_upper[1])) .* (λData ./ λData[1, :]')
	end

	SamplingWeights = γ.SamplingWeights

	@show Dates.format(now(), "HH:MM") # print time


	u = range(0, 2, length = 100)


	#x, domestic/rw/ratio, δ, bound=lower/initial/upper

	marg_cdf = MarginalPricesCDF(u, baseIndex, Aod, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, λData, D, θ_initial[1], Θ_upper[1], Θ_lower[1], wHat, τ)
	#marg_pdf = MarginalPricesPDF(u, baseIndex, Aod, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, λData)


	p1 = plot(δ_grid, κ_upper, lw = 4, color = colors[1], label = "Upper Bound")
	plot!(p1, δ_grid, κ_lower, lw = 4, color = colors[1], label = "Lower Bound")
	xlabel!(p1, "δ", xlabelfontsize = 14)
	ylabel!(p1, "Counterfactual", ylabelfontsize = 14)
	savefig(p1, string("counterfactuals_", file_name, "_.png"))


	p2 = plot(δ_grid, δ_LFD_upper, lw = 4, color = colors[1], label = "Upper Bound")
	plot!(p2, δ_grid, δ_LFD_lower, lw = 4, color = colors[1], label = "Lower Bound")
	xlabel!(p2, "δ Budget", xlabelfontsize = 14)
	ylabel!(p2, "δ at LFD", ylabelfontsize = 14)
	savefig(p2, string("counterfactual_deltas_", file_name, "_.png"))



	for i ∈ 1:length(δ_grid)

		savefig(plot(xlabel = "Price", ylabel = "CDF", u, [marg_cdf[:, 1, i, 2] marg_cdf[:, 1, i, 3] marg_cdf[:, 1, i, 1]], label = ["P" "P+" "P-"], title = string("δ=", δ_grid[i])), string("marginalCDF_", "delta", δ_grid[i], "_", file_name, "_.png"))
		savefig(
			plot(xlabel = "Price", ylabel = "CDF", u, [marg_cdf[:, 2, i, 2] marg_cdf[:, 2, i, 3] marg_cdf[:, 2, i, 1]], label = ["PRW" "PRW+" "PRW-"], title = string("δ=", δ_grid[i])),
			string("marginalCDFRW", "delta", δ_grid[i], "_", file_name, "_.png"),
		)
		savefig(
			plot(xlabel = "Price Ratio", ylabel = "CDF", u, [marg_cdf[:, 3, i, 2] marg_cdf[:, 3, i, 3] marg_cdf[:, 3, i, 1]], label = ["P/PRW" "P+/PRW+" "P-/PRW-"], title = string("δ=", δ_grid[i])),
			string("marginalCDFPratio", "delta", δ_grid[i], "_", file_name, "_.png"),
		)
		#=
		savefig(plot(xlabel="Price", ylabel="PDF", u[1:end-1], [marg_pdf[:,i, 2, 1] marg_pdf[:,i, 3, 1] marg_pdf[:,i, 1, 1]], label=["P" "P+" "P-"], title=string("δ=", δ_grid[i])), string("marginalPDF", "delta", δ_grid[i], "_", file_name, "_.png"))
		savefig(plot(xlabel="Price", ylabel="PDF", u[1:end-1], [marg_pdf[:,i, 2, 2] marg_pdf[:,i, 3, 2] marg_pdf[:,i, 1, 2]], label=["PRW" "PRW+" "PRW-"], title=string("δ=", δ_grid[i])), string("marginalPDFRW", "delta", δ_grid[i], "_", file_name, "_.png"))
		savefig(plot(xlabel="Price Ratio", ylabel="PDF", u[1:end-1], [marg_pdf[:,i, 2, 3] marg_pdf[:,i, 3, 3] marg_pdf[:,i, 1, 3]], label=["P/PRW" "P+/PRW+" "P-/PRW-"], title=string("δ=", δ_grid[i])), string("marginalPDFPratio", "delta", δ_grid[i], "_", file_name, "_.png"))
		=#
	end

	# x, δ, lower/initial/upper
	y11 = UCDF(u, 1, 1, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, D)
	ybb = UCDF(u, baseIndex, baseIndex, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, D)

	yb = zeros(4, length(u), δ_grid_size, 3) # o=1:4, x, δ, lower/initial/upper
	for i ∈ 1:4
		yb[i, :, :, :] = UCDF(u, i, baseIndex, U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, D)
	end


	for i ∈ 1:δ_grid_size
		savefig(
			plot(
				xlabel = "Unscaled U",
				ylabel = "CDF",
				u,
				[ybb[:, i, 2] yb[1, :, i, 1] yb[2, :, i, 1] yb[3, :, i, 1] yb[4, :, i, 1] y11[:, i, 1]],
				label = ["U_base,base" "U1base-" "U2base-" "U3base-" "U4base-" "U11-"],
				title = string("Marginals for lower bound δ = ", δ_grid[i], "base = ", baseIndex),
			),
			string("lower_marginals_delta_", δ_grid[i], "_", file_name, "_.png"),
		)
		savefig(
			plot(
				xlabel = "Unscaled U",
				ylabel = "CDF",
				u,
				[ybb[:, i, 2] yb[1, :, i, 3] yb[2, :, i, 3] yb[3, :, i, 3] yb[4, :, i, 3] y11[:, i, 3]],
				label = ["U_base,base" "U1base+" "U2base+" "U3base+" "U4base+" "U11+"],
				title = string("Marginals for upper bound δ = ", δ_grid[i], "base = ", baseIndex),
			),
			string("upper_marginals_delta_", δ_grid[i], "_", file_name, "_.png"),
		)
		savefig(
			plot(xlabel = "Unscaled U", ylabel = "CDF", u, [ybb[:, i, 1] ybb[:, i, 2] ybb[:, i, 3]], label = ["U_base,base-" "U_base,base" "U_base,base+"], title = string("Marginals for base country δ = ", δ_grid[i], "base = ", baseIndex)),
			string("marginals_delta_", δ_grid[i], "_", file_name, "_.png"),
		)
	end


	# D^2, D^2, lower/Initial/upper, δ
	M = correlationMatrix(U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, D)

	for i ∈ 1:δ_grid_size

		savefig(heatmap(M[:, :, 1, i], fc = cgrad([:white, :dodgerblue4])),
			string("lower_bound_correlation_delta_", δ_grid[i], "_", file_name, ".png"))
		writedlm(string("correlation_matrix_lower_bound_delta_", δ_grid[i], "_", file_name, ".csv"), M[:, :, 1, i], ',')


		savefig(heatmap(M[:, :, 2, i], fc = cgrad([:white, :dodgerblue4])),
			string("central_correlation_delta_", δ_grid[i], "_", file_name, ".png"))
		writedlm(string("correlation_matrix_central_delta_", δ_grid[i], "_", file_name, ".csv"), M[:, :, 2, i], ',')

		savefig(heatmap(M[:, :, 3, i], fc = cgrad([:white, :dodgerblue4])),
			string("upper_bound_correlation_delta_", δ_grid[i], "_", file_name, ".png"))
		writedlm(string("correlation_matrix_upper_bound_delta_", δ_grid[i], "_", file_name, ".csv"), M[:, :, 3, i], ',')
	end


	@show Dates.format(now(), "HH:MM") # print time

end

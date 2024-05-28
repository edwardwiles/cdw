function master_cc_algo(prep_output, params)

	@unpack δ_grid, file_name, θ_initial = prep_output

	Θ_upper, κ_upper, Θ_lower, κ_lower = ccOuter(prep_output, params) # run the outer loop 
	writedlm(file_name, [δ_grid κ_lower κ_upper], ',')

	LFD_upper, LFD_lower = LFD(Θ_upper, Θ_lower, prep_output, params)

	# save the cc data 
	save_object(string("cc_input_", file_name, ".jld2"),
		(prep_output = prep_output,
			params = params,
			Θ_upper = Θ_upper,
			κ_upper = κ_upper,
			Θ_lower = Θ_lower,
			κ_lower = κ_lower,
			LFD_upper = LFD_upper,
			LFD_lower = LFD_lower,
		))

	if params.runLFD == 1
		runLFD(Θ_upper, Θ_lower, LFD_upper, LFD_lower, prep_output, params)
	end

	if params.runLFDCounterFactual == 1
		LFDCounterFactual(Θ_upper, Θ_lower, LFD_upper, LFD_lower, prestep_output, prep_output, params)
	end

end

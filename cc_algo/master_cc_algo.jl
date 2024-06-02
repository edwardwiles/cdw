function master_cc_algo(prep_output, params)

	@unpack δ_grid, file_name = prep_output

	Θ_upper, κ_upper, Θ_lower, κ_lower = ccOuter(prep_output, params) # run the outer loop 
	writedlm(file_name, [δ_grid κ_lower κ_upper], ',')

	cc_output = (
		Θ_upper = Θ_upper,
		Θ_lower = Θ_lower,
		κ_upper = κ_upper,
		κ_lower = κ_lower,
	)

	return cc_output
end

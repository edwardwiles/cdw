function master_lfd(params, setup_output, prestep_output, prep_output, cc_output)

	lfd_output = LFD(cc_output, prep_output, params)

	# save the cc data 
	save_object(string("cc_input_", file_name, ".jld2"),
		(params = params,
			setup_output = setup_output,
			prestep_output = prestep_output,
			prep_output = prep_output,
			cc_output = cc_output,
			lfd_output = lfd_output,
		))

	master_post_cc_lfd(params, params, setup_output, prestep_output, prep_output, cc_output, lfd_output)

end

function master_post_cc_lfd(postCCParams, params, setup_output, prestep_output, prep_output, cc_output, lfd_output)

	if postCCParams.runLFD == 1
		runLFD(lfd_output, cc_output, prep_output, prestep_output, setup_output, params)
	end

	if postCCParams.runLFDCounterFactual == 1
		LFDCounterFactual(lfd_output, cc_output, prestep_output, prep_output, params)
	end

end
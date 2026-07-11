# Supplemental functions for hybrid divergence objective

# Evaluate convex conjugate of phi in-place
function Psi!(arg1, arg0)
	@inbounds for i in 1:length(arg0)
		if arg0[i] <= 1.0
			arg1[i] = exp(arg0[i])
		else
			arg1[i] = arg0[i]^2 + 1.0
			arg1[i] *= 0.5*exp(1)
		end
	end
	arg1 .-= 1.0
end

# Evaluate its derivative in-place
function dPsi!(arg1, arg0)
	@inbounds for i in 1:length(arg0)
		if arg0[i] <= 1.0
			arg1[i] = exp(arg0[i])
		else
			arg1[i] = exp(1)*arg0[i]
		end
	end
end

# Evaluate its second derivative in-place
function ddPsi!(arg1, arg0)
	@inbounds for i in 1:length(arg0)
		if arg0[i] <= 1.0
			arg1[i] = exp(arg0[i])
		else
			arg1[i] = exp(1)
		end
	end
end
# Compute local sensitivity measure
function local_sensitivity(obj, θ)

	obj.moments!(@view(obj.H[:, 1]), select_G_from_H(obj, obj.H), θ, obj.U, obj)
	calculate_jac_θ!(obj, θ)

	J = mean(obj.jac_h[:, 1, :], dims = 1)[:]
	D = local_sensitivity_jacobian(obj, θ)
	G = select_G_from_H(obj, obj.H)
	K = obj.H[:, 1]
	V = cov(G, G)

	A = V \ D
	B = A / (D' * A)

	i = K - G * B * J - (G / V) * cov(G, K) + G * B * A' * cov(G,K) .- mean(K)
	s = 2 * mean(i .^ 2)

	return s

end

function local_sensitivity_jacobian(obj::KLObjectiveBundle, θ)

	D = zeros(obj.d, obj.l)
	for i in 1:obj.d
		D[i, :] = mean(obj.jac_h[:, i + 1, :], dims = 1)
	end

	return D

end

function local_sensitivity_jacobian(obj::PsiObjectiveBundle, θ)

	D = zeros(obj.d, obj.l)
	for i in 1:obj.d
		D[i, :] = mean(obj.jac_h[:, i + 2, :], dims = 1)
	end

	return D

end
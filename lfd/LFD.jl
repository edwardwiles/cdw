function LFD(θ_initial, U, γ, gravMoment, localGravityMoment, localGravityCrossMoment, GravityMomentFirstApproach, sameMarginalsMoment, NoScalingforSameMartingale, independenceMoment, momentOrder, momentOrderForBaseIndex, useIndependentCFDs, IndMomentOrder, counterType)
    # function that generate the LFD for a particular θ, by running the innerloop
    D = length(γ.L)
    W = size(U, 1)
   # numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (1 + (1-sameMarginalsMoment)*D^2) + sameMarginalsMoment * ((2 * momentOrder) * D^2 + momentOrderForBaseIndex) +  independenceMoment * (D * (D^2 - floor(Int, D * (1 + D) / 2) + 1)+ (D-1)* 2 * momentOrder)
   numMoments = D^2 + 2 * D + gravMoment + localGravityMoment * (D - 1) * D + localGravityCrossMoment * D * (D - 1) * (D - 2) + GravityMomentFirstApproach * (1 + (1-sameMarginalsMoment)*D^2) + sameMarginalsMoment * ((2 * momentOrder + 1) * D^2 + momentOrderForBaseIndex) + independenceMoment * (1+ D * (D^2 - floor(Int, D * (1 + D) / 2))+ (1-useIndependentCFDs)*(D-1)* 2 * momentOrder + useIndependentCFDs*((IndMomentOrder-1) + (IndMomentOrder-1)^D))

    if counterType != 1
        numMoments += (D - 1)
    end

    G = zeros(W, numMoments)
    K = zeros(W, 1)

    obj = PsiObjectiveBundleDelta(
        #δ = 1,
        #find_smallest = true,
        γ=γ,
        (moments!)=moments!,
        #moments_jacobian! = rust_moments_jacobian!,
        d=numMoments,
        outer_constr_index=numMoments + 1 - GravityMomentFirstApproach,
        inequality_index=Int64[],
        l=size(θ_initial, 1),
        U=U,
        N=100,
        outer_loop_opt="ek_outer_loop_options.opt",
        inner_loop_opt="ek_inner_loop_options.opt",
        lower_limit=-50)

    val, x, nStatus = inner_loop(obj, θ_initial)

    moments!(K, G, θ_initial, U, obj)

    arg0 = zeros(W, 1)
    LFD = zeros(W, 1)

    @show G[1, :]
    @show numMoments
    @show x


    meanLDF = 0
    for ω = 1:W
        arg0[ω] = -x[1] - dot(G[ω, 1:numMoments-GravityMomentFirstApproach], x[2:length(x)])
    end

    dPsi!(LFD, arg0)

    # calculate mean and the variance of the moments
    MomentsMean = zeros(numMoments)
    MomentsVar = zeros(numMoments)
    MomentsMean = mean(LFD .* G, dims=1)
    MomentsVar = mean(LFD .* G .* G, dims=1) .- (MomentsMean .* MomentsMean)
    MomentsTest = zeros(numMoments)
    @show MomentsMean
    @show MomentsVar

    for i = 1:numMoments
        MomentsTest[i] = abs(MomentsMean[i])<2*sqrt(MomentsVar[i]/W) ? 1 : 0
    end

    @show MomentsTest
    cInd = 0

    if counterType != 1
        cInd = D^2 + D - 1
        dInd = D^2 + 2 * D - 1
    else
        cInd = D^2
        dInd = D^2 + D
    end

    baseIndex = γ.baseIndex

   @show MomentsTest[cInd+baseIndex]
   @show MomentsTest[dInd+baseIndex]

   # check the confidence interval of the counterfactual bounds
   paramγ = θ_initial[3:3+D-1]
   paramγ_prime = copy(paramγ)
   if counterType != 1
    paramγ_prime = θ_initial[3+D:3+2*D-1]
   else
    paramγ_prime[baseIndex] = θ_initial[3+D] # we are not interested in the other gamma_primes
   end

   σ = θ_initial[2]

   counterVal = (paramγ[baseIndex] / paramγ_prime[baseIndex])^(σ / (σ - 1)) - 1

   Varγ = MomentsVar[cInd+baseIndex]/W
   Varγ_prime = MomentsVar[dInd+baseIndex]/W
  
   #approximation for the variance of the estimator K as the ratio of two estimators
    varκ = (σ / (σ - 1))*(counterVal^2)*(Varγ*paramγ[baseIndex]^(-2) + Varγ_prime*paramγ_prime[baseIndex]^(-2))
    σκ = sqrt(varκ)
   @show MomentsVar
   @show MomentsTest
   @show counterVal
   @show σκ

   return LFD
    
    #=return (LFD = LFD,
    MomentsMean = MomentsMean,
    MomentsVar = MomentsVar,
    MomentsTest = MomentsTest,
    counterVal_up = counterVal+2*σκ,
    counterVal = counterVal,
    counterVal_down = counterVal-2*σκ)
    =#
end
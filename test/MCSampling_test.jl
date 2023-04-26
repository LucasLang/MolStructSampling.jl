using Test
using Statistics

using MolStructSampling.MCSampling
using MolStructSampling.ECWaveFunction

"""
Note that the mean of the exact probability distribution lies around 0.8.
It is not impossible (but extremely unlikely) that the mean of the samples falls outside the specified range.
"""
function MCrun_test()
    # this test wavefunction is simply exp(-r^2), i.e. density = exp(-2r^2)
    n = 1
    masses = [1.0, 1.0]
    charges = [1.0, -1.0]
    M = 1
    C = [1.0]
    L_flattened = [[1.0]]
    B_flattened = [[0.0]]
    Y = ECWaveFunction.YoungOperator([1.0], [ECWaveFunction.PseudoParticlePermutation([])])
    param = WaveFuncParam(n, masses, charges, M, C, L_flattened, B_flattened, Y)
    param_processed = ECWaveFunction.WaveFuncParamProcessed(param)
    prob_dens(r) = ECWaveFunction.calc_probability_density(r, param_processed)
    r_start = [1.0, 0.0, 0.0]
    widths = [0.3]
    nsteps = 1000000
    samples, rho_values, accepted_rejected = MCrun(prob_dens, n, nsteps, r_start, widths)
    r_values = [sqrt(samples[:,i]'*samples[:,i]) for i in 1:nsteps]
    r_mean = mean(r_values)
    return (r_mean>0.75) && (r_mean<0.85)
end

function accepted_rejected_test()
    # this test wavefunction is simply exp(-r^2), i.e. density = exp(-2r^2)
    n = 1
    masses = [1.0, 1.0]
    charges = [1.0, -1.0]
    M = 1
    C = [1.0]
    L_flattened = [[1.0]]
    B_flattened = [[0.0]]
    Y = ECWaveFunction.YoungOperator([1.0], [ECWaveFunction.PseudoParticlePermutation([])])
    param = WaveFuncParam(n, masses, charges, M, C, L_flattened, B_flattened, Y)
    param_processed = ECWaveFunction.WaveFuncParamProcessed(param)
    prob_dens(r) = ECWaveFunction.calc_probability_density(r, param_processed)
    r_start = [1.0, 0.0, 0.0]
    widths = [0.3]
    nsteps = 10000
    samples, rho_values, accepted_rejected = MCrun(prob_dens, n, nsteps, r_start, widths)
    return sum(accepted_rejected) == nsteps
end

@testset "MCSampling.jl" begin
    @test MCrun_test()
    @test accepted_rejected_test()
end

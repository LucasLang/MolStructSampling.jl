using Test
using Statistics

include("../src/MCSampling.jl")
using Main.MCSampling
include("../src/ECWaveFunction.jl")
using Main.ECWaveFunction

"""
Note that the mean of the exact probability distribution lies around 0.8.
It is not impossible (but extremely unlikely) that the mean of the samples falls outside the specified range.
"""
function MCrun_test()
    # this test wavefunction is simply exp(-r^2), i.e. density = exp(-2r^2)
    n = 1
    M = 1
    C = [1.0]
    L_flattened = [[1.0]]
    B_flattened = [[0.0]]
    Y = ECWaveFunction.YoungOperator([1.0], [ECWaveFunction.PseudoParticlePermutation([])])
    param = WaveFuncParam(n, M, C, L_flattened, B_flattened, Y)
    param_processed = ECWaveFunction.WaveFuncParamProcessed(param)
    prob_dens(r) = ECWaveFunction.calc_probability_density(r, param_processed)
    r_start = [1.0, 0.0, 0.0]
    widths = [0.3]
    nsteps = 1000000
    samples = MCrun(prob_dens, n, nsteps, r_start, widths)
    r_values = [sqrt(samples[:,i]'*samples[:,i]) for i in 1:nsteps]
    r_mean = mean(r_values)
    return (r_mean>0.75) && (r_mean<0.85)
end

@testset "Module MCSampling" begin
    @test MCrun_test()
end
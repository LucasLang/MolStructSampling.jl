using MolStructSampling
using Test

@testset verbose=true "MolStructSampling.jl" begin
    include("ECWaveFunction_test.jl")
    include("MCSampling_test.jl")
    include("Analysis_test.jl")
end

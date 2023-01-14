using Test
include("../src/ECWaveFunction.jl")
using Main.ECWaveFunction

function test_flattened_to_lower()
    n = 3
    flattened = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ref = [1.0  0.0  0.0; 2.0  4.0  0.0; 3.0  5.0  6.0]
    return flattened_to_lower(n, flattened) ≈ ref
end

function test_flattened_to_symmetric()
    n = 3
    flattened = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ref = [1.0  2.0  3.0; 2.0  4.0  5.0; 3.0  5.0  6.0]
    return flattened_to_symmetric(n, flattened) ≈ ref
end


@testset "Module ECWaveFunction" begin
    @test test_flattened_to_lower()
    @test test_flattened_to_symmetric()
end


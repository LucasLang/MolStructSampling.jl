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

function test_Transposition_wrongorder()
    t = ECWaveFunction.Transposition((2,1))
end

function test_get_indices()
    indices = (3, 5)
    t = ECWaveFunction.Transposition(indices)
    return (ECWaveFunction.get_indices(t) == indices)
end

function test_transposition_matrix_1()
    t = ECWaveFunction.Transposition((1,4))
    n = 4
    ref = [1.0  0.0  -1.0  0.0;
           0.0  1.0  -1.0  0.0;
           0.0  0.0  -1.0  0.0;
           0.0  0.0  -1.0  1.0]
    return ECWaveFunction.transposition_matrix_pseudo(n, t) ≈ ref
end

function test_transposition_matrix_2()
    t = ECWaveFunction.Transposition((2,4))
    n = 4
    ref = [0.0  0.0  1.0  0.0;
           0.0  1.0  0.0  0.0;
           1.0  0.0  0.0  0.0;
           0.0  0.0  0.0  1.0]
    return ECWaveFunction.transposition_matrix_pseudo(n, t) ≈ ref
end



@testset "Module ECWaveFunction" begin
    @test test_flattened_to_lower()
    @test test_flattened_to_symmetric()
    @test_throws ErrorException test_Transposition_wrongorder()
    @test test_get_indices()
    @test test_transposition_matrix_1()
    @test test_transposition_matrix_2()
end


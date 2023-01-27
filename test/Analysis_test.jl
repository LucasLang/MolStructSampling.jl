using Test

include("../src/Analysis.jl")
using Main.Analysis

function partial_means_test()
    vec = [1, 4, 2, 11, 21, 3, 3, 2]
    ref = [1.0, 2.5, 2.3333333333333335, 4.5, 7.8, 7.0, 6.428571428571429, 5.875]
    return calc_partial_means(vec) ≈ ref
end

@testset "Module Analysis" begin
    @test partial_means_test()
end
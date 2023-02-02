using Test

include("../src/Analysis.jl")
using Main.Analysis

function partial_means_test()
    vec = [1, 4, 2, 11, 21, 3, 3, 2]
    ref = [1.0, 2.5, 2.3333333333333335, 4.5, 7.8, 7.0, 6.428571428571429, 5.875]
    return calc_partial_means(vec) ≈ ref
end

function test_coordinates_single2multiple_vectors()
    r = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ,9.0]
    r_separate_ref = [[0.0, 0.0, 0.0],
                      [1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]]
    return Analysis.coordinates_single2multiple_vectors(r) ≈ r_separate_ref
end

function test_calc_nuclear_COM()
    r_separate_nuc = [[1.0,2,3], [4,5,6], [7,8,9]]
    masses = [1, 2, 1]
    ref_COM = [4.0, 5.0, 6.0]
    return Analysis.calc_nuclear_COM(r_separate_nuc, masses) ≈ ref_COM
end

function test_shift2neworig()
    r_separate = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    neworig = [1.0, 1.0, 1.0]
    r_separate_neworig_ref = [[-1.0, -1.0, -1.0], [0.0, 1.0, 2.0]]
    return Analysis.shift2neworig(r_separate, neworig) ≈ r_separate_neworig_ref
end

function test_determine_basis_inplane_3particle()
    r_separate = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, -4.0, 0.0]]
    e1_ref = (1/sqrt(5))*[1,2,0]
    e2_ref = (1/sqrt(5))*[2,-1,0]
    e3_ref = [0,0,-1.0]
    e = Analysis.determine_basis_inplane_3particle(r_separate)

    c1 = e[:,1] ≈ e1_ref
    c2 = e[:,1] ≈ e1_ref
    c3 = e[:,1] ≈ e1_ref
    return c1 && c2 && c3
end

function test_transform_newbasis()
    r = [[1,2,3], [4,5,6]]
    B = [0 0 1; 1 0 0; 0 1 0]   # this basis change is just a cyclic permutation of the original basis vectors
    r_newbasis_ref = [[2,3,1], [5,6,4]]
    return Analysis.transform_newbasis(r, B) ≈ r_newbasis_ref
end

function test_calc_centroid()
    P = [1 2 3; 4 5 6; 7 8 9; 10 11 12]
    centroid_ref = [5.5, 6.5, 7.5]
    return calc_centroid(P) ≈ centroid_ref
end


@testset "Module Analysis" begin
    @test partial_means_test()
    @test test_coordinates_single2multiple_vectors()
    @test test_calc_nuclear_COM()
    @test test_shift2neworig()
    @test test_determine_basis_inplane_3particle()
    @test test_transform_newbasis()
    @test test_calc_centroid()
end
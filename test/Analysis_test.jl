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

function test_optimal_rotation()
    R = [-0.21483723836839663 0.8872306883463706 0.40824829046386246; -0.5205873894647373 0.2496439529882974 -0.8164965809277261; -0.8263375405610782 -0.3879427823697744 0.40824829046386324]
    P = [1 2 3; 4 5 6; 10 -1 5; 33 22 11; 6 5 4]
    Q = P*R'   # this applies the rotation matrix R to all points P_i, i.e., Q_i = RP_i.
    R_opt = optimal_rotation(P,Q)
    return R_opt ≈ R'     # R_opt should undo the rotation R, i.e., be its inverse = transpose
end

function test_R_COMframe()
    masses = [1,2,3,4,5]
    r_pseudoparticle = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0]
    Nnuc = 3
    R_nuc_ref = [[-4/3, -2/3, 0], [1-4/3, 2-2/3, 3], [-1-4/3, -2-2/3, -3]]
    return R_nuc_ref ≈ R_COMframe(r_pseudoparticle, masses, Nnuc)
end

function test_Ropt2()
    R1 = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
    R2 = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]
    R1_matrix = vecofvec_to_matrix(R1)
    R2_matrix = vecofvec_to_matrix(R2)
    Uopt = optimal_rotation(R1_matrix, R2_matrix)
    Uopt_ref = [0.0 0.0 -1.0;
                0.0 1.0 0.0;
                1.0 0.0 0.0]
    return Uopt ≈ Uopt_ref
end

function test_minRMSD()
    R1 = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
    R2 = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]
    return minRMSD(R1, R2) ≈ sqrt(1.25/3)
end

function test_minRMSD2()
    rot1 = [-0.8386577433025008 -0.35688886660368085 -0.41144079342366763; -0.08045115535302273 -0.6659529361392574 0.7416429723592515; -0.5386843242578357 0.6550855087252673 0.5297944649173267]
    rot2 = [-0.10382531050662758 0.38218953092040714 -0.9182327958383111; 0.1280379522784344 -0.9104043268758157 -0.39340849556426455; -0.9863197187902855 -0.15841440610194413 0.04558824700315456]
    R1 = [1 2 3; 5 1 2; -7 -7 -7; 3 2 1; 7 1 -5]
    R2 = [5 1 0; -11 12 1; 7 3 -4; 1 1 2; 7 7 7]
    R1_rot = R1*rot1'
    R2_rot = R2*rot2'
    dist1 = minRMSD(R1, R2)
    dist2 = minRMSD(R1_rot, R2_rot)
    return dist1 ≈ dist2
end


@testset "Module Analysis" begin
    @test partial_means_test()
    @test test_coordinates_single2multiple_vectors()
    @test test_calc_nuclear_COM()
    @test test_shift2neworig()
    @test test_determine_basis_inplane_3particle()
    @test test_transform_newbasis()
    @test test_calc_centroid()
    @test test_optimal_rotation()
    @test test_R_COMframe()
    @test test_Ropt2()
    @test test_minRMSD()
    @test test_minRMSD2()
end
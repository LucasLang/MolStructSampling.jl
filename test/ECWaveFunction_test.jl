using Test
using LinearAlgebra

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

function test_permutation_matrix()
    transpositions = [(1, 3), (4, 5)]
    p = ECWaveFunction.PseudoParticlePermutation(transpositions)
    n = 4
    ref = [1.0  -1.0  0.0  0.0
           0.0  -1.0  0.0  0.0
           0.0  -1.0  0.0  1.0
           0.0  -1.0  1.0  0.0]
    return ECWaveFunction.permutation_matrix_pseudo(n, p) ≈ ref
end

function test_WaveFuncParamProcessed()
    n = 3
    masses = [1.0, 1.0, 1.0, 1.0]
    charges = [1.0, 1.0, -1.0, -1.0]
    M = 2
    C = [10.0, 11.0]
    L_flattened = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    B_flattened = [[13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                   [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]]
    transpositions_tuples = [[], [(1, 3), (3, 4)]]
    transpositions = [[ECWaveFunction.Transposition(tuple) for tuple in p] for p in transpositions_tuples]
    Y = ECWaveFunction.YoungOperator([1.0, -1.0], [ECWaveFunction.PseudoParticlePermutation(p) for p in transpositions])
    param = WaveFuncParam(n, masses, charges, M, C, L_flattened, B_flattened, Y)
    param_processed = ECWaveFunction.WaveFuncParamProcessed(param)

    ref_A21 = [49.0   0.0   0.0   56.0    0.0    0.0   63.0    0.0    0.0;
                0.0  49.0   0.0    0.0   56.0    0.0    0.0   63.0    0.0;
                0.0   0.0  49.0    0.0    0.0   56.0    0.0    0.0   63.0;
               56.0   0.0   0.0  164.0    0.0    0.0  182.0    0.0    0.0;
                0.0  56.0   0.0    0.0  164.0    0.0    0.0  182.0    0.0;
                0.0   0.0  56.0    0.0    0.0  164.0    0.0    0.0  182.0;
               63.0   0.0   0.0  182.0    0.0    0.0  346.0    0.0    0.0;
                0.0  63.0   0.0    0.0  182.0    0.0    0.0  346.0    0.0;
                0.0   0.0  63.0    0.0    0.0  182.0    0.0    0.0  346.0]
    ref_A22 = [  49.0     0.0     0.0    63.0     0.0     0.0  -168.0    -0.0    -0.0;
                  0.0    49.0     0.0     0.0    63.0     0.0    -0.0  -168.0    -0.0;
                  0.0     0.0    49.0     0.0     0.0    63.0    -0.0    -0.0  -168.0;
                 63.0     0.0     0.0   346.0     0.0     0.0  -591.0    -0.0    -0.0;
                  0.0    63.0     0.0     0.0   346.0     0.0    -0.0  -591.0    -0.0;
                  0.0     0.0    63.0     0.0     0.0   346.0    -0.0    -0.0  -591.0;
               -168.0    -0.0    -0.0  -591.0    -0.0    -0.0  1161.0     0.0     0.0;
                 -0.0  -168.0    -0.0    -0.0  -591.0    -0.0     0.0  1161.0     0.0;
                 -0.0    -0.0  -168.0    -0.0    -0.0  -591.0     0.0     0.0  1161.0]
    ref_B12 = [ 13.0    0.0    0.0   15.0    0.0    0.0  -42.0   -0.0   -0.0;
                 0.0   13.0    0.0    0.0   15.0    0.0   -0.0  -42.0   -0.0;
                 0.0    0.0   13.0    0.0    0.0   15.0   -0.0   -0.0  -42.0;
                15.0    0.0    0.0   18.0    0.0    0.0  -50.0   -0.0   -0.0;
                 0.0   15.0    0.0    0.0   18.0    0.0   -0.0  -50.0   -0.0;
                 0.0    0.0   15.0    0.0    0.0   18.0   -0.0   -0.0  -50.0;
               -42.0   -0.0   -0.0  -50.0   -0.0   -0.0  139.0    0.0    0.0;
                -0.0  -42.0   -0.0   -0.0  -50.0   -0.0    0.0  139.0    0.0;
                -0.0   -0.0  -42.0   -0.0   -0.0  -50.0    0.0    0.0  139.0]
    c1 = param_processed.C ≈ param.C
    c2 = param_processed.p_coeffs ≈ param.Y.coeffs
    c3 = param_processed.A[2, 1] ≈ ref_A21
    c4 = param_processed.A[2, 2] ≈ ref_A22
    c5 = param_processed.B[1, 2] ≈ ref_B12
    return c1 && c2 && c3 && c4 && c5
end

function test_identity_permutation()
    p = ECWaveFunction.PseudoParticlePermutation([])
    n = 4
    ref = Matrix{Float64}(I, n, n)
    return ECWaveFunction.permutation_matrix_pseudo(n, p) ≈ ref
end

function test_parse_Youngoperator1()
    str = "(1+P12)(1-P34)"
    PPP = ECWaveFunction.PseudoParticlePermutation
    coeffs_ref = [1.0, 1.0, -1.0, -1.0]
    permutations_ref = [PPP([]),
                        PPP([(1, 2)]),
                        PPP([(3, 4)]),
                        PPP([(1, 2), (3, 4)])]
    Y_ref = ECWaveFunction.YoungOperator(coeffs_ref, permutations_ref)
    Y = parse(ECWaveFunction.YoungOperator, str)
    return Y == Y_ref
end

function test_parse_Youngoperator2()
    str = "(1+P12)(1+P13+P23)(1+P45)"
    PPP = ECWaveFunction.PseudoParticlePermutation
    coeffs_ref = [1.0 for i in 1:12]
    permutations_ref = [PPP([]),
                        PPP([(1, 2)]),
                        PPP([(1, 3)]),
                        PPP([(1, 2), (1, 3)]),
                        PPP([(2, 3)]),
                        PPP([(1, 2), (2, 3)]),
                        PPP([(4, 5)]),
                        PPP([(1, 2), (4, 5)]),
                        PPP([(1, 3), (4, 5)]),
                        PPP([(1, 2), (1, 3), (4, 5)]),
                        PPP([(2, 3), (4, 5)]),
                        PPP([(1, 2), (2, 3), (4, 5)])]
    Y_ref = ECWaveFunction.YoungOperator(coeffs_ref, permutations_ref)
    Y = parse(ECWaveFunction.YoungOperator, str)
    return Y == Y_ref
end

function test_readWF()
    folder = "D3plus_param"
    param = ECWaveFunction.read_wavefuncparam(folder)
    c_n = param.n == 4
    c_masses = param.masses ≈ [0.36704829652E+04, 0.36704829652E+04, 0.36704829652E+04, 1.0, 1.0]
    c_charges = param.charges ≈ [1.0, 1.0, 1.0, -1.0, -1.0]
    c_M = param.M == 400
    c_C = (param.C[11] ≈ 0.124456114642243+9.240646041392420E-002im) && (param.C[42] ≈ 0.328701772837231-0.426789750897380im)
    c_L = param.L_flattened[5][2] ≈ -0.5625016825356735E+00
    c_B = param.B_flattened[42][5] ≈ 0.9006031655758847E+00
    c_Y = param.Y == parse(ECWaveFunction.YoungOperator, "(1+P12)(1+P13+P23)(1+P45)")
    return c_n && c_masses && c_charges && c_M && c_C && c_L && c_B && c_Y
end




@testset "Module ECWaveFunction" begin
    @test test_flattened_to_lower()
    @test test_flattened_to_symmetric()
    @test_throws ErrorException test_Transposition_wrongorder()
    @test test_get_indices()
    @test test_transposition_matrix_1()
    @test test_transposition_matrix_2()
    @test test_permutation_matrix()
    @test test_WaveFuncParamProcessed()
    @test test_identity_permutation()
    @test test_parse_Youngoperator1()
    @test test_parse_Youngoperator2()
    @test test_readWF()
end


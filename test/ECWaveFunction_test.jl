using Test
using LinearAlgebra

using MolStructSampling.ECWaveFunction

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
    transpositions_tuples = [[], [(3, 4), (1, 3)]]
    transpositions = [[ECWaveFunction.Transposition(tuple) for tuple in p] for p in transpositions_tuples]
    Y = ECWaveFunction.YoungOperator([1.0, -1.0], [ECWaveFunction.PseudoParticlePermutation(p) for p in transpositions])
    param = WaveFuncParam(n, masses, charges, M, C, L_flattened, B_flattened, Y)
    norm_consts = ECWaveFunction.calc_norm_consts_afterproj(param)

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
    c1 = param_processed.C ≈ norm_consts .* param.C
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
    param = ECWaveFunction.WaveFuncParam(folder)
    c_n = param.n == 4
    c_masses = param.masses ≈ [0.36704829652E+04, 0.36704829652E+04, 0.36704829652E+04, 1.0, 1.0]
    c_charges = param.charges ≈ [1.0, 1.0, 1.0, -1.0, -1.0]
    c_M = param.M == 400
    c_C = (param.C[11] ≈ 0.124456114642243+9.240646041392420E-002im) && (param.C[42] ≈ 0.328701772837231-0.426789750897380im)
    c_L = param.L_flattened[5][2] ≈ -0.2009259946516603E+00
    c_B = param.B_flattened[42][5] ≈ -0.2059324200753563E+00
    c_Y = param.Y == parse(ECWaveFunction.YoungOperator, "(1+P12)(1+P13+P23)(1+P45)")
    return c_n && c_masses && c_charges && c_M && c_C && c_L && c_B && c_Y
end

function test_D3plus_invariance()
    param_processed = ECWaveFunction.WaveFuncParamProcessed("D3plus_param")
    prob_dens(r) = ECWaveFunction.calc_probability_density(r, param_processed)
    dist = 1.720
    rD2 = re1 = [0.0, 0.0, dist]
    rD3 = re2 = [(sqrt(3)/2)*dist, 0.0, dist/2]
    r1 = [rD2; rD3; re1; re2]
    r2 = [rD3; rD2; re1; re2]
    r3 = [rD2; rD3; re2; re1]
    r4 = [rD3; rD2; re2; re1]
    p1 = prob_dens(r1)
    p2 = prob_dens(r2)
    p3 = prob_dens(r3)
    p4 = prob_dens(r4)
    c_equality = (p1 ≈ p2) && (p1 ≈ p3) && (p1 ≈ p4)
    r_wrongperm = [rD2; re1; rD3; re2]
    p_wrongperm = prob_dens(r_wrongperm)
    c_inequality = !(p1 ≈ p_wrongperm)
    return c_equality && c_inequality
end

# The Hamiltonian is nonrelativistic, therefore the wavefunction must be real (or: can only have a global,
# no local, phase).
function test_globalphase()
    param_processed = ECWaveFunction.WaveFuncParamProcessed("HDplus_param")
    wavefunction(r) = ECWaveFunction.calc_wavefunction(r, param_processed)
    # Several trial coordinates with internuclear distance close to Born-Oppenheimer distance of about 2 Bohrs:
    r1 = [0.0, 0.0, 1.9, 0.0, 0.5, 1.0]
    r2 = [0.0, 0.0, 2.2, 0.0, 0.0, 0.0]
    r3 = [0.0, 0.0, 2.0, 0.0, 0.0, 2.0]
    
    c1 = abs(angle(wavefunction(r1))) < 0.01
    c2 = abs(angle(wavefunction(r2))) < 0.01
    c3 = abs(angle(wavefunction(r3))) < 0.01

    return c1 && c2 && c3
end

function test_wavefunction()
    n = 2
    masses = [3400.0, 1.0, 1.0]
    charges = [1.0, -1.0, -1.0]
    M = 2
    Y = parse(ECWaveFunction.YoungOperator, "(1-P12)")
    C_linear = rand(2)+rand(2)*im
    L_flattened = [rand(3) for k in 1:M]
    B_flattened = [rand(3) for k in 1:M]
    L = [[flattened[1] 0; flattened[2] flattened[3]] for flattened in L_flattened]

    A = [L[k]*L[k]' for k in 1:M]
    B = [[flattened[1] flattened[2]; flattened[2] flattened[3]] for flattened in B_flattened]
    C = [A[k] + B[k]*im for k in 1:M]
    param = ECWaveFunction.WaveFuncParam(n, masses, charges, M, C_linear, L_flattened, B_flattened, Y)
    norm_consts = ECWaveFunction.calc_norm_consts_afterproj(param)
    param_processed = ECWaveFunction.WaveFuncParamProcessed(param)
    wavefunction(r) = ECWaveFunction.calc_wavefunction(r, param_processed)

    R1 = rand(3)
    R2 = rand(3)
    R3 = rand(3)
    r1 = R2 - R1
    r2 = R3 - R1

    R1_perm = R2
    R2_perm = R1
    R3_perm = R3
    r1_perm = R2_perm - R1_perm
    r2_perm = R3_perm - R1_perm

    Phi1 = norm_consts[1]*exp(-(C[1][1,1]*r1'*r1 + 2*C[1][1,2]*r1'*r2 + C[1][2,2]*r2'*r2))
    Phi2 = norm_consts[2]*exp(-(C[2][1,1]*r1'*r1 + 2*C[2][1,2]*r1'*r2 + C[2][2,2]*r2'*r2))
    Phi1_perm = norm_consts[1]*exp(-(C[1][1,1]*r1_perm'*r1_perm + 2*C[1][1,2]*r1_perm'*r2_perm + C[1][2,2]*r2_perm'*r2_perm))
    Phi2_perm = norm_consts[2]*exp(-(C[2][1,1]*r1_perm'*r1_perm + 2*C[2][1,2]*r1_perm'*r2_perm + C[2][2,2]*r2_perm'*r2_perm))

    wavefunction_ref = C_linear[1]*(Phi1-Phi1_perm) + C_linear[2]*(Phi2-Phi2_perm)

    r = [r1; r2]
    return wavefunction_ref ≈ wavefunction(r)
end

function test_wavefunction2()
    foldername = "HDplus_param_chopped"
    param = ECWaveFunction.WaveFuncParam(foldername)
    L = [[flattened[1] 0; flattened[2] flattened[3]] for flattened in param.L_flattened]
    norm_consts = [(2^2 *(Lk[1,1]*Lk[2,2])^2 / (pi^2))^0.75 for Lk in L]
    A = [L[k]*L[k]' for k in 1:param.M]
    B = [[flattened[1] flattened[2]; flattened[2] flattened[3]] for flattened in param.B_flattened]
    C = [A[k] + B[k]*im for k in 1:param.M]

    param_processed = ECWaveFunction.WaveFuncParamProcessed(param)
    wavefunction(r) = ECWaveFunction.calc_wavefunction(r, param_processed)

    R1 = rand(3)
    R2 = rand(3)
    R3 = rand(3)
    r1 = R2 - R1
    r2 = R3 - R1

    Phi1 = exp(-(C[1][1,1]*r1'*r1 + 2*C[1][1,2]*r1'*r2 + C[1][2,2]*r2'*r2))
    Phi2 = exp(-(C[2][1,1]*r1'*r1 + 2*C[2][1,2]*r1'*r2 + C[2][2,2]*r2'*r2))

    wavefunction_ref = param_processed.C[1]*Phi1 + param_processed.C[2]*Phi2
    r = [r1; r2]
    return wavefunction_ref ≈ wavefunction(r)
end

function test_normconst()
    L = [1.0 0; 2 3]
    return ECWaveFunction.calc_norm_const(L) ≈ 2.6393808814892727
end

"""
In this test, we numerically integrate overlap matrix elements between
unnormalized explicitly correlated Gaussian functions for the HD+ molecule.
"""
function test_numeric_integration()
    C = [1 0; 0 1]

    nperdim = 100   # number of intervals / evaluation points
    endr = 4.0
    endtheta = pi

    integral = ECWaveFunction.calc_overlap_3part(C, C, endr, endr, endtheta, nperdim)
    ref = (pi/2)^3
    return abs(ref-integral) < 0.001
end

"""
In this test, we numerically integrate overlap matrix elements between
normalized explicitly correlated Gaussian functions for the HD+ molecule.
Note: With this test, I realized that the order of the basis functions 
in the vector of linear expansion coefficients is inverted.
"""
function test_overlap_numeric()
    function overlap(k,l, param)
        norm_const_k = ECWaveFunction.calc_norm_const(param, k)
        Ck = ECWaveFunction.get_C_matrix(param, k)
        norm_const_l = ECWaveFunction.calc_norm_const(param, l)
        Cl = ECWaveFunction.get_C_matrix(param, l)

        nperdim = 100   # number of intervals / evaluation points

        endr1 = 5.0
        endr2 = 5.0
        endtheta = pi

        return norm_const_k*norm_const_l*ECWaveFunction.calc_overlap_3part(Ck, Cl, endr1, endr2, endtheta, nperdim)
    end

    param = ECWaveFunction.WaveFuncParam("HDplus_param")

    S_1_1 = overlap(1,1,param)
    S_1_2 = overlap(1,2,param)
    S_1_3 = overlap(1,3,param)
    S_2_2 = overlap(2,2,param)
    S_26_91 = overlap(26,91,param)
    S_78_7 = overlap(78,7,param)

    # The reference values are taken directly from the printout of the
    # overlap matrix from Ludwik's code.
    S_ref_1_1 = 0.1000000000000000E+01 
    S_ref_1_2 = -0.1671667245349336E-03 -0.1296978079959740E+00im
    S_ref_1_3 = -0.1644834988085512E-01 -0.5964543743141933E-01im
    S_ref_2_2 = 0.1000000000000000E+01 
    S_ref_26_91 = 0.5504153118944703E-02 -0.1057583150521015E+00im
    S_ref_78_7 =0.1503629803375769E+00 +0.1744409121917275E+00im

    c1 = abs(S_1_1-S_ref_1_1)<0.001
    c2 = abs(S_1_2-S_ref_1_2)<0.001
    c3 = abs(S_1_3-S_ref_1_3)<0.001
    c4 = abs(S_2_2-S_ref_2_2)<0.001
    c5 = abs(S_26_91-S_ref_26_91)<0.001
    c6 = abs(S_78_7-S_ref_78_7)<0.001

    return c1 && c2 && c3 && c4 && c5 && c6
end

function test_potential_energy()
    r = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    charges = [1, 2, -1]
    Epot = ECWaveFunction.calc_potential_energy(r, charges)
    return Epot ≈ (2-1-sqrt(2))
end

function test_overlap_unnormalized()
    param = ECWaveFunction.WaveFuncParam("HDplus_param")
    k = 13
    norm_const_k = ECWaveFunction.calc_norm_const(param, k)
    Ck = ECWaveFunction.get_C_matrix(param, k)
    overlap_kk = ECWaveFunction.calc_overlap_unnormalized(Ck, Ck)

    return (norm_const_k^2 * overlap_kk) ≈ 1.0
end

function test_overlap_unnormalized2()
    Lk = [1 0; 2 3]
    Ak = Lk*Lk'
    Bk = [4 5; 5 6]
    Ck = Ak + Bk*im
    norm_const_k = ECWaveFunction.calc_norm_const(Lk)
    overlap_kk = ECWaveFunction.calc_overlap_unnormalized(Ck, Ck)

    return (norm_const_k^2 * overlap_kk) ≈ 1.0
end

function test_wavefunction_value_HDplus()
    param_processed = ECWaveFunction.WaveFuncParamProcessed("HDplus_param")
    wf(r) = ECWaveFunction.calc_wavefunction(r, param_processed)
    return wf([0.0, 0.0, 2.0, 0.0, 0.0, 2.0]) ≈ (0.07450263634420635 + 0.000122352872978414im)
end

function test_wavefunction_value_D3plus()
    param_processed = ECWaveFunction.WaveFuncParamProcessed("D3plus_param")
    wf(r) = ECWaveFunction.calc_wavefunction(r, param_processed)
    dist_avg = 1.72   # from Cafiero, Adamowicz 2004
    D2 = [0.0, 0.0, dist_avg]
    D3 = [(sqrt(3)/2)*dist_avg, 0.0, 0.5*dist_avg]
    e1 = 0.5 .* D2
    e2 = 0.5 .* D3
    r = [D2; D3; e1; e2]

    return wf(r) ≈ (-0.013810604415407693 + 0.005248954597497664im)
end

function test_overlap_projectedbasis()
    param = ECWaveFunction.WaveFuncParam("D3plus_param")
    range = 1:3
    S = ECWaveFunction.calc_overlap_projectedbasis(param, range)
    S_diag = deepcopy([S[k,k] for k in range])
    for row in range
        S[row,:] = S[row,:] .* (1/sqrt(S_diag[row]))
    end
    for col in range
        S[:,col] = S[:,col] .* (1/sqrt(S_diag[col]))
    end

    # The following reference values are taken from printout of Ludwik's ocelote program
    S_ref_1_1 = 0.1000000000000000E+01 
    S_ref_1_2 = -0.6792524461245966E-01 +0.1842111495950207E+00im
    S_ref_1_3 = 0.2256232118917643E+00 +0.1324910273402387E+00im
    S_ref_2_3 =0.9460973793085009E-03 -0.9594352950993022E-01im

    c1 = (S_ref_1_1 ≈ S[1,1])
    c2 = (S_ref_1_2 ≈ S[1,2])
    c3 = (S_ref_1_3 ≈ S[1,3])
    c4 = (S_ref_2_3 ≈ S[2,3])
    return c1 && c2 && c3 && c4
end

function test_overlap_projectedbasis_D3()
    param = ECWaveFunction.WaveFuncParam("D3_param")
    range = 1:4
    S = ECWaveFunction.calc_overlap_projectedbasis(param, range)

    S_diag = deepcopy([S[k,k] for k in range])
    for row in range
        S[row,:] = S[row,:] .* (1/sqrt(S_diag[row]))
    end
    for col in range
        S[:,col] = S[:,col] .* (1/sqrt(S_diag[col]))
    end

    # The following reference values are taken from printout of Ludwik's ocelote program
    S_ref_1_1 = 0.1000000000000000E+01 
    S_ref_1_2 = 0.1713070445612793E+00 - 0.2927552733774612E-01im
    S_ref_1_3 = 0.1567207141859616E+00 + 0.9823447722497833E-01im
    S_ref_1_4 = 0.1390492527836689E+00 + 0.1412139469393235E+00im
    S_ref_2_3 = 0.3777187525106950E+00 + 0.2282214119639189E+00im
    S_ref_2_4 = 0.1272939621672173E+00 + 0.1659339179551543E+00im
    S_ref_3_4 = 0.4084984377243480E+00 + 0.2376335293952030E+00im

    c1 = (S_ref_1_1 ≈ S[1,1])
    c2 = (S_ref_1_2 ≈ S[1,2])
    c3 = (S_ref_1_3 ≈ S[1,3])
    c4 = (S_ref_1_4 ≈ S[1,4])
    c5 = (S_ref_2_3 ≈ S[2,3])
    c6 = (S_ref_2_4 ≈ S[2,4])
    c7 = (S_ref_3_4 ≈ S[3,4])

    return c1 && c2 && c3 && c4 && c5 && c6 && c7
end

function test_D3_overlapcontribution()
    param = ECWaveFunction.WaveFuncParam("D3_param")
    k = 1

    L = ECWaveFunction.flattened_to_lower(param.n, param.L_flattened[k])
    A = L*L'
    B = ECWaveFunction.flattened_to_symmetric(param.n, param.B_flattened[k])
    C = A+B*im

    P = [1    0    0    0    0;
    0    1    0    0    0;
    0    0    0    1    0;
    0    0    1    0    0;
    0    0    0    0    1]

    ref_value = 0.5008780577377846E+00 -0.2215755867623245E-17im   # value printed within ocelote
    return ECWaveFunction.calc_overlap_normalized(C, P'*C*P) ≈ ref_value
end

function test_permutation_matrices()
    param = ECWaveFunction.WaveFuncParam("D3_param")
    P_matrices = [ECWaveFunction.permutation_matrix_pseudo(param.n, p) for p in param.Y.permutations]
    Id = [1.0  0.0  0.0  0.0  0.0;
          0.0  1.0  0.0  0.0  0.0;
          0.0  0.0  1.0  0.0  0.0;
          0.0  0.0  0.0  1.0  0.0;
          0.0  0.0  0.0  0.0  1.0]
    P12 = [-1.0  0.0  0.0  0.0  0.0;
           -1.0  1.0  0.0  0.0  0.0;
           -1.0  0.0  1.0  0.0  0.0;
           -1.0  0.0  0.0  1.0  0.0;
           -1.0  0.0  0.0  0.0  1.0]
    P13 = [1.0  -1.0  0.0  0.0  0.0;
           0.0  -1.0  0.0  0.0  0.0;
           0.0  -1.0  1.0  0.0  0.0;
           0.0  -1.0  0.0  1.0  0.0;
           0.0  -1.0  0.0  0.0  1.0]
    P23 = [0.0  1.0  0.0  0.0  0.0;
           1.0  0.0  0.0  0.0  0.0;
           0.0  0.0  1.0  0.0  0.0;
           0.0  0.0  0.0  1.0  0.0;
           0.0  0.0  0.0  0.0  1.0]
    P46 = [1.0  0.0  0.0  0.0  0.0;
           0.0  1.0  0.0  0.0  0.0;
           0.0  0.0  0.0  0.0  1.0;
           0.0  0.0  0.0  1.0  0.0;
           0.0  0.0  1.0  0.0  0.0]
    P45 = [1.0  0.0  0.0  0.0  0.0;
           0.0  1.0  0.0  0.0  0.0;
           0.0  0.0  0.0  1.0  0.0;
           0.0  0.0  1.0  0.0  0.0;
           0.0  0.0  0.0  0.0  1.0]
    
    c1 = P_matrices[1]   ≈ Id
    c2 = P_matrices[2]   ≈ P12
    c3 = P_matrices[3]   ≈ P13
    c4 = P_matrices[4]   ≈ P13*P12
    c5 = P_matrices[5]   ≈ P23
    c6 = P_matrices[6]   ≈ P23*P12
    c7 = P_matrices[7]   ≈ P46
    c8 = P_matrices[8]   ≈ P46*P12
    c9 = P_matrices[9]   ≈ P46*P13
    c10 = P_matrices[10] ≈ P46*P13*P12
    c11 = P_matrices[11] ≈ P46*P23
    c12 = P_matrices[12] ≈ P46*P23*P12
    c13 = P_matrices[13] ≈ P45
    c14 = P_matrices[14] ≈ P45*P12
    c15 = P_matrices[15] ≈ P45*P13
    c16 = P_matrices[16] ≈ P45*P13*P12
    c17 = P_matrices[17] ≈ P45*P23
    c18 = P_matrices[18] ≈ P45*P23*P12
    c19 = P_matrices[19] ≈ P45*P46
    c20 = P_matrices[20] ≈ P45*P46*P12
    c21 = P_matrices[21] ≈ P45*P46*P13
    c22 = P_matrices[22] ≈ P45*P46*P13*P12
    c23 = P_matrices[23] ≈ P45*P46*P23
    c24 = P_matrices[24] ≈ P45*P46*P23*P12
    return c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9 && c10 && c11 && c12 && c13 && c14 && c15 && c16 && c17 && c18 && c19 && c20 && c21 && c22 && c23 && c24
end




@testset "ECWaveFunction.jl" begin
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
    @test test_D3plus_invariance()
    @test test_globalphase()
    @test test_wavefunction()
    @test test_wavefunction2()
    @test test_normconst()
    @test test_numeric_integration()
    @test test_overlap_numeric()
    @test test_potential_energy()
    @test test_overlap_unnormalized()
    @test test_overlap_unnormalized2()
    @test test_wavefunction_value_HDplus()
    @test test_overlap_projectedbasis()
    @test test_wavefunction_value_D3plus()
    @test test_overlap_projectedbasis_D3()
    @test test_D3_overlapcontribution()
    @test test_permutation_matrices()
end


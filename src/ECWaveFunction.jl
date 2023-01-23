module ECWaveFunction

using LinearAlgebra, DelimitedFiles

import Base.length, Base.parse, Base.*, Base.==

export WaveFuncParam, flattened_to_lower, flattened_to_symmetric
    
include("Auxiliary.jl")

struct Transposition
    indices::Tuple{Int64, Int64}
    Transposition(indices) = indices[1] > indices[2] ? error("Wrong order of indices in transposition") : new(indices)
end

function get_indices(t::Transposition)
    return t.indices[1], t.indices[2]
end

struct PseudoParticlePermutation
    transpositions::Vector{Transposition}
end

function PseudoParticlePermutation(indices::Vector{Tuple{Int64, Int64}})
    transpositions = [Transposition(tuple) for tuple in indices]
    return PseudoParticlePermutation(transpositions)
end

function (==)(p1::PseudoParticlePermutation, p2::PseudoParticlePermutation)
    l1 = length(p1.transpositions)
    l2 = length(p2.transpositions)
    if l1 != l2
        return false
    end
    transpositions_equal = [p1.transpositions[i] == p2.transpositions[i] for i in 1:l1]
    overall_equality = true
    for check in transpositions_equal
        overall_equality = overall_equality && check
    end
    return overall_equality
end

"""
    coeffs: prefactors of the permutation operators
    permutations: actual permutations whose linear combination is taken
"""
struct YoungOperator
    coeffs::Vector{Float64}
    permutations::Vector{PseudoParticlePermutation}
end

length(Y::YoungOperator) = length(Y.coeffs)

function (==)(Y1::YoungOperator, Y2::YoungOperator)
    return (Y1.coeffs ≈ Y2.coeffs) && (Y1.permutations == Y2.permutations)
end

function (*)(Y1::YoungOperator, Y2::YoungOperator)
    coeffs = [c1*c2 for c1 in Y1.coeffs, c2 in Y2.coeffs]
    permutations = [p1*p2 for p1 in Y1.permutations, p2 in Y2.permutations]
    return YoungOperator(reshape(coeffs, :), reshape(permutations, :))
end

function (*)(p1::PseudoParticlePermutation, p2::PseudoParticlePermutation)
    return PseudoParticlePermutation([p1.transpositions; p2.transpositions])
end

"""
Convert a string of the form "1+P13+P23" into a Young operator.
"""
function sumstring_to_YoungOperator(str)
    coeffs = Vector{Float64}()
    permutations = Vector{PseudoParticlePermutation}()
    parts = Vector{String}()
    start = 1
    for i in 1:length(str)
        if (str[i] == '+') || (str[i] == '-')
            append!(parts, [str[start:i-1]])
            start = i
        end
    end
    append!(parts, [str[start:end]])

    for part in parts
        if part[1] == '-'        # e.g. part = "-P12"
            append!(coeffs, [-1.0])
            part = part[2:end]   # drop prefactor from the string
        elseif part[1] == '+'   # e.g. part = "+P12"
            append!(coeffs, [1.0])
            part = part[2:end]   # drop prefactor from the string
        else                     # e.g. part = "1"
            append!(coeffs, [1.0])
        end
        if part == "1"
            append!(permutations, [PseudoParticlePermutation([])])
        end
        if part[1] == 'P'
            transposition = parse(Transposition, part)
            append!(permutations, [PseudoParticlePermutation([transposition])])
        end
    end
    return YoungOperator(coeffs, permutations)
end

"""
Parse strings of the form "P12".
"""
function parse(::Type{Transposition}, str)
    i1 = parse(Int64, str[2])
    i2 = parse(Int64, str[3])
    return Transposition((i1, i2))
end

"""
Parse strings of the form "(1+P12)(1+P13+P23)(1+P45)".
"""
function parse(::Type{YoungOperator}, str)
    indices_sums = x = Vector{Tuple{Int64,Int64}}()
    start = -1   # need to define here to make it visible in second block
    for i in 1:length(str)
        if str[i] == '('
            start = i+1
        end
        if str[i] == ')'
            stop = i-1
            append!(indices_sums, [(start, stop)])
        end
    end
    factors = [sumstring_to_YoungOperator(str[ind[1]:ind[2]]) for ind in indices_sums]
    Y_total = YoungOperator([1.0], [PseudoParticlePermutation([])])   # initialize as identity operator
    for factor in factors
        Y_total = Y_total*factor
    end
    return Y_total
end

"""
    n: Number of quasiparticles (= one less than the number of actual particles)
    M: Number of explicitly correlated basis functions
    C: The vector of linear expansion coefficients
    L_flattened: For each of the M basis functions: vector of n(n+1)/2 elements of the lower-triangular Cholesky factor L
    B_flattened: For each of the M basis functions: vector of n(n+1)/2 unique elements of the symmetric matrix B
    Y: The product of Young operators (projector applied to trial wavefunction)

"""
struct WaveFuncParam
    n::Int64
    masses::Vector{Float64}
    charges::Vector{Float64}
    M::Int64
    C::Vector{ComplexF64}
    L_flattened::Vector{Vector{Float64}}
    B_flattened::Vector{Vector{Float64}}
    Y::YoungOperator
end

"""
Reads wavefunction parameters from the files inout.txt, coeffs, and gauss_param located in the specified folder.
"""
function read_wavefuncparam(folder::String)
    inout_lines = open(readlines, "$folder/inout.txt")
    N = parse(Int64, split(inout_lines[1])[2])   # total number of particles
    n = N-1     # number of pseudoparticles
    masses = [parse(Float64, replace(split(inout_lines[2])[i+1], "D"=>"E")) for i in 1:N]
    charges = [parse(Float64, replace(split(inout_lines[3])[i+1], "D"=>"E")) for i in 1:N]
    Y = parse(YoungOperator, split(inout_lines[4])[2])   # product of Young operators
    M = parse(Int64, split(inout_lines[5])[2])           # number of complex ECG basis functions

    coeffs_array = readdlm("$folder/coeffs")
    C = [coeffs_array[i, 2] + coeffs_array[i, 3]*im for i in 1:M]   # linear coefficients

    gauss_param_array = readdlm("$folder/gauss_param")
    dim_flattened = n*(n+1)÷2
    L_flattened = [[parse(Float64, replace(gauss_param_array[bf, 1+i], "D"=>"E")) for i in 1:dim_flattened] for bf in 1:M]
    B_flattened = [[parse(Float64, replace(gauss_param_array[bf, 1+dim_flattened+i], "D"=>"E")) for i in 1:dim_flattened] for bf in 1:M]
    return WaveFuncParam(n, masses, charges, M, C, L_flattened, B_flattened, Y)
end


"""
    n: Number of quasiparticles (= one less than the number of actual particles)
    M: Number of explicitly correlated basis functions
    C: The vector of linear expansion coefficients
    A[k, i]: A matrix of the ith permutation applied to the kth basis function
    B[k, i]: B matrix of the ith permutation applied to the kth basis function
    p_coeffs: The coefficients of the permutations in the Young operator
"""
struct WaveFuncParamProcessed
    n::Int64
    masses::Vector{Float64}
    charges::Vector{Float64}
    M::Int64
    C::Vector{ComplexF64}
    A::Matrix{Matrix{Float64}}
    B::Matrix{Matrix{Float64}}
    p_coeffs::Vector{Float64}
end

function WaveFuncParamProcessed(param::WaveFuncParam)
    n = param.n
    M = param.M
    Y = param.Y
    p_matrices = [permutation_matrix_pseudo(n, p) for p in param.Y.permutations]
    id3 = Matrix{Float64}(I, 3, 3)      # 3x3 identity matrix
    A = Matrix{Matrix{Float64}}(undef, M, length(Y))
    B = Matrix{Matrix{Float64}}(undef, M, length(Y))
    for k in 1:M
        L = flattened_to_lower(n, param.L_flattened[k])
        Ak = L*L'
        Bk = flattened_to_symmetric(n, param.B_flattened[k])
        for i in 1:length(Y)
            Aki = p_matrices[i]'*Ak*p_matrices[i]
            Bki = p_matrices[i]'*Bk*p_matrices[i]
            A[k, i] = Aki ⊗ id3
            B[k, i] = Bki ⊗ id3
        end
    end
    return WaveFuncParamProcessed(n, param.masses, param.charges, M, param.C, A, B, Y.coeffs)
end

"""
Returns the transposition matrix acting on pseudoparticle coordinates.
"""
function transposition_matrix_pseudo(n::Integer, transposition::Transposition)
    i, j = get_indices(transposition)    # i is smaller than j
    if i==1
        return transposition_matrix_pseudo_ref(n, j)
    else
        return transposition_matrix_pseudo_other(n, transposition)
    end
end

"""
Returns the transposition matrix acting on pseudoparticle coordinates
for the case where the reference particle (index 1) is involved.
"""
function transposition_matrix_pseudo_ref(n::Integer, j::Int)
    T = Matrix{Float64}(I, n, n)
    T[:, j-1] .= -1.0
    return T
end

"""
Returns the transposition matrix acting on pseudoparticle coordinates
for the case where the reference particle (index 1) is not involved.
"""
function transposition_matrix_pseudo_other(n::Integer, transposition::Transposition)
    T = Matrix{Float64}(I, n, n)
    i, j = get_indices(transposition)
    col_im1 = T[:, i-1]
    col_jm1 = T[:, j-1]
    T[:, i-1] = col_jm1
    T[:, j-1] = col_im1
    return T
end

"""
Returns the permutation matrix acting on pseudoparticle coordinates.
"""
function permutation_matrix_pseudo(n::Integer, p::PseudoParticlePermutation)
    transposition_matrices = [transposition_matrix_pseudo(n, t) for t in p.transpositions]
    matrix_product = Matrix{Float64}(I, n, n)   # if there are no transpositions: return identity
    for matrix in transposition_matrices
        matrix_product = matrix_product*matrix
    end
    return matrix_product
end

"""
Takes the flattened form of a lower-triangular matrix and assembles the full 2D matrix

n: number of rows and columns of the matrix
flattened: flattened form of the matrix with n(n+1)/2 elements

"""
function flattened_to_lower(n, flattened)
    matrix = zeros(n,n)
    counter = 1
    for col in 1:n
        for row in col:n
            matrix[row, col] = flattened[counter]
            counter += 1
        end
    end
    return matrix
end

"""
Takes the flattened form of a symmetric matrix and assembles the full matrix

n: number of rows and columns of the matrix
flattened: flattened form of the matrix with n(n+1)/2 elements

"""
function flattened_to_symmetric(n, flattened)
    matrix = zeros(n,n)
    counter = 1
    for col in 1:n
        for row in col:n
            matrix[row, col] = flattened[counter]
            matrix[col, row] = flattened[counter]
            counter += 1
        end
    end
    return matrix
end

"""
Calculates the positive definite A matrix from the wavefunction parameters
"""
function calc_A(param::WaveFuncParam)
    L = flattened_to_lower(param.n, param.L_flattened)
    return L*L'
end

function calc_probability_density(r::Vector{Float64}, par::WaveFuncParamProcessed)
    permuted_basis_values = [exp(-r'*par.A[k, i]*r -r'*par.B[k, i]*r*im) for k in 1:par.M, i in 1:length(par.p_coeffs)]
    projected_basis_values = [par.p_coeffs'*permuted_basis_values[k, :] for k in 1:par.M]
    wavefunction_value = par.C'*projected_basis_values
    return abs2(wavefunction_value)
end


end

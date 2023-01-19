module ECWaveFunction

using LinearAlgebra

import Base.length

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

"""
    coeffs: prefactors of the permutation operators
    permutations: actual permutations whose linear combination is taken
"""
struct YoungOperator
    coeffs::Vector{Float64}
    permutations::Vector{PseudoParticlePermutation}
end

length(Y::YoungOperator) = length(Y.coeffs)

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
    M::Int64
    C::Vector{Float64}
    L_flattened::Vector{Vector{Float64}}
    B_flattened::Vector{Vector{Float64}}
    Y::YoungOperator
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
    M::Int64
    C::Vector{Float64}
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
    return WaveFuncParamProcessed(n, M, param.C, A, B, Y.coeffs)
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

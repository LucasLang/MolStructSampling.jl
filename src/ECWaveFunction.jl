module ECWaveFunction

export WaveFuncParam, flattened_to_lower, flattened_to_symmetric
    
struct PseudoParticlePermutation
    transpositions::Vector{Tuple{Int64, Int64}}
end

"""
    coeffs: prefactors of the permutation operators
    permutations: actual permutations whose linear combination is taken
"""
struct YoungOperator
    coeffs::Vector{Float64}
    permutations::Vector{PseudoParticlePermutation}
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
    perm_coeffs: The coefficients of the permutations in the Young operator
"""
struct WaveFuncParamProcessed
    n::Int64
    M::Int64
    C::Vector{Float64}
    A::Matrix{Matrix{Float64}}
    B::Matrix{Matrix{Float64}}
    perm_coeffs::Vector{Float64}
end

function WaveFuncParamProcessed(param::WaveFuncParam)
    n = param.n
    M = param.M
    C = param.C
    perm_len = length(Y.coeffs)
    A = Matrix{Matrix{Float64}}(undef, M, perm_len)
    B = Matrix{Matrix{Float64}}(undef, M, perm_len)
    for k in 1:M
        for i in 1:perm_len
            L = flattened_to_lower(n, param.L_flattened[k])
            A = L*L'
            B = flattened_to_symmetric(n, param.B_flattened[k])
            A[k, i] = CONTINUE LATER
            B[k, i] = CONTINUE LATER
        end
    end
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


end
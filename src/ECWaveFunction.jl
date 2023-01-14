module ECWaveFunction

export WaveFuncParam, flattened_to_lower, flattened_to_symmetric
    
"""
   n: Number of quasiparticles (= one less than the number of actual particles)
   L_flattened: n(n+1)/2 elements of the lower-triangular Cholesky factor L
   B_flattened: n(n+1)/2 unique elements of the symmetric matrix B

"""
struct WaveFuncParam
    n::Int64
    L_flattened::Vector{Float64}
    B_flattened::Vector{Float64}
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
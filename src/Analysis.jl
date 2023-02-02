module Analysis

using LinearAlgebra

export calc_partial_means

function calc_partial_means(vec::Vector{T}) where T<:Real
    N = length(vec)
    partial_means = Vector{Float64}(undef, N)
    partial_means[1] = vec[1]
    for i in 2:N
        partial_means[i] = ((i-1)*partial_means[i-1]+vec[i])/i
    end
    return partial_means
end

"""
This function transforms the single vector containing all pseudoparticle coordinates
to a vector of 3-component vectors for the individual particles, including the reference
particle in the first position.
"""
function coordinates_single2multiple_vectors(r::Vector{T}) where T <: Real
    n = length(r)÷3
    N = n+1
    r_separate = Vector{Vector{T}}(undef, N)
    r_separate[1] = [0.0, 0.0, 0.0]   # coordinates of the reference particle
    for i in 1:n
        start = 1+(i-1)*3
        stop = 3+(i-1)*3
        r_separate[i+1] = r[start:stop]
    end
    return r_separate
end

"""
Calculates the center of mass of the nuclei in a given set of coordinates.
The two inputs only contain the coordinates and masses of the nuclei, not of the electrons
"""
function calc_nuclear_COM(r_separate_nuc::Vector{Vector{T1}}, masses::Vector{T2}) where {T1 <: Real, T2 <: Real}
    Nnuc = length(r_separate_nuc)
    m_tot = sum(masses)
    #return (1/m_tot) .* sum([masses[i] .* r_separate_nuc[i] for i in 1:Nnuc])
    return (1/m_tot) .* sum(masses .* r_separate_nuc)
end

"""
Calculates coordinates with respect to a new coordinate origin (which itself is specified
with respect to the old origin).
"""
function shift2neworig(r_separate::Vector{Vector{T1}}, neworig::Vector{T2}) where {T1 <: Real, T2 <: Real}
    return [vec-neworig for vec in r_separate]
end


"""
For a system with 3 particles (typically nuclei), this function determines an orthonormal basis
that is defined as follows:
b1 is the normalized vector from p1 to p2
b2 is the Gram-Schmidt orthogonalized vector from p1 to p3
b3 = b1 x b2 (cross-product)
This definition ensures that the vector from p1 to p2 is in b1 direction, that the whole 3-particle
system lies in the b1-b2 plane, and that the coordinate system is right-handed.
"""
function determine_basis_inplane_3particle(r::Vector{Vector{T}}) where T <: Real
    v1 = r[2] - r[1]
    v2 = r[3] - r[1]

    u1 = v1
    u2 = v2 - ((v2'*u1)/(u1'*u1))*u1   # Gram-Schmidt orthogonalization

    b = Vector{Vector{Float64}}(undef, 3)
    b[1] = u1/sqrt(u1'*u1)    # normalization
    b[2] = u2/sqrt(u2'*u2)    # normalization
    b[3] = b[1] × b[2]

    B = Matrix{Float64}(undef, 3, 3)
    for i in 1:3
        B[:, i] = b[i]
    end
    return B
end

"""
columns of matrix B are coordinates of the new basis vectors in terms of the old basis vectors.
The basis is assumed to be orthonormal, i.e., the matrix B must be an orthogonal matrix!
"""
function transform_newbasis(r_separate::Vector{Vector{T1}}, B::Matrix{T2}) where {T1 <: Real, T2 <: Real}
    return [B'*vec for vec in r_separate]
end

end
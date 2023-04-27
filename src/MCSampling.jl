module MCSampling

export MCrun

include("Auxiliary.jl")

"""
    prob_dens: The probability density function
    n: Number of pseudoparticles
    nsteps: The number of Monte Carlo steps
    r_start: 3n-element vector with the initial pseudoparticle coordinates.

    Returns an array of all the pseudoparticle coordinates for the different steps.
    (this does NOT contain the starting coordinates)
"""
function MCrun(prob_dens::Function, n::Integer, nsteps::Integer, r_start::Vector{Float64}, widths::Vector{Float64})
    saved_r = Array{Float64}(undef, 3n, nsteps)
    saved_P = Vector{Float64}(undef, nsteps)
    accepted_rejected = zeros(Int64, n, 2)    # First col: number of accepted, sec col: number of rejected
    r_current = r_start
    P_current = prob_dens(r_current)
    for step in 1:nsteps
        pp_index, r_prop = propose_next_coordinates(r_current, n, widths)
        P_prop = prob_dens(r_prop)
        A = min(1, P_prop/P_current)
        if rand() < A
            r_current = r_prop
            P_current = P_prop
            accepted_rejected[pp_index, 1] += 1
        else
            accepted_rejected[pp_index, 2] += 1
        end
        saved_r[:, step] = r_current
        saved_P[step] = P_current
    end
    return saved_r, saved_P, accepted_rejected
end

"""
    width: the length of the three edges of the cube from which we draw the displacement
"""
function get_random_displacement(width)
    return [width*(rand()-0.5) for i in 1:3]
end

"""
    r_current: Current pseudoparticle coordinates
    n: Number of pseudoparticles
    widths: The jumping widths for the different pseudoparticles

    Returns a proposition for the next pseudoparticle coordinates
"""
function propose_next_coordinates(r_current::Vector{Float64}, n::Integer, widths::Vector{Float64})
    pp_index = rand(1:n)
    displacement = get_random_displacement(widths[pp_index])
    unitvec = zeros(n)
    unitvec[pp_index] = 1.0
    r_prop = r_current + unitvec ⊗ displacement
    return pp_index, r_prop
end

end

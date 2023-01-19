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
    r_current = r_start
    P_current = prob_dens(r_current)
    for step in 1:nsteps
        r_prop = propose_next_coordinates(r_current, n, widths)
        P_prop = prob_dens(r_prop)
        A = min(1, P_prop/P_current)
        if rand() < A
            r_current = r_prop
            P_current = P_prop
        end
        saved_r[:, step] = r_current
    end
    return saved_r
end

"""
    width: the width of the three sides of the cube from which we draw the displacement
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
    return r_current + unitvec ⊗ displacement
end

end
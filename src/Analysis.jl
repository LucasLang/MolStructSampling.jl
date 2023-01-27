module Analysis

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

end
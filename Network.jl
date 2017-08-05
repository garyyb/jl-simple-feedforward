module FeedForward
import Core
mutable struct Network
    num_layers::Int64
    sizes::Array{Int, 1}
    biases::Array{Vector{Float64}, 1}
    weights::Array{Matrix{Float64}, 1}

    function Network(sizes::Array{Int, 1})
        num_layers = length(sizes)

        biases = Vector{Float64}[]
        weights = Matrix{Float64}[]

        for i = 1:num_layers - 1
            num_neurons = sizes[i + 1]
            bias = zeros(Float64, num_neurons)
            weight = zeros(Float64, num_neurons, sizes[i])

            push!(biases, bias)
            push!(weights, weight)
        end

        new(num_layers, sizes, biases, weights)
    end
end

export Network
end

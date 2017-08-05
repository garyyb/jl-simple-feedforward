module FeedForward
import Core
mutable struct Network
    num_layers::Int64
    sizes::Array{Int, 1}
    biases::Array{Vector{Float64}, 1}
    weights::Array{Matrix{Float64}, 1}
    activator::Function
    activator_derivative::Function
    cost_derivative::Function

    function Network(sizes::Array{Int, 1}, activator::Function,
                     activator_derivative::Function, cost_derivative::Function)
        num_layers = length(sizes)

        biases = Vector{Float64}[]
        weights = Matrix{Float64}[]

        for i = 1:num_layers - 1
            num_neurons = sizes[i + 1]
            bias = zeros(Float64, num_neurons)
            weight = zeros(Float64, num_neurons, sizes[i])
            randn!(bias)
            randn!(weight)
            push!(biases, bias)
            push!(weights, weight)
        end

        new(num_layers, sizes, biases, weights,
            activator, activator_derivative, cost_derivative)
    end
end

function backprop(n::Network, initial::Vector{Float64}, output::Vector{Float64})
    bias_changes = Vector{Float64}[]
    weight_changes = Matrix{Float64}[]

    num_changes = n.num_layers - 1
    for i = 1:num_changes
        push!(bias_changes, zeros(n.biases[i]))
        push!(weight_changes, zeroes(n.weights[i]))
    end

    # FORWARD!
    activaton = initial
    activations = [activation]
    z_vectors = Vector{Float64}[]

    for (bias_v, weight_m) in zip(n.biases, n.weights)
        z = weight_m * activation + bias_v
        push!(z_vectors, z)
        activation = n.activator(z)
        push!(activations, activation)
    end

    # RETREAT!
    num_activations = length(activations)
    num_zs = length(z_vectors)
    error = n.cost_derivative(activations[num_activations], output) *
            n.activator_derivative(z_vectors[num_zs])

    bias_changes[num_changes] = error
    weight_changes[num_changes] = error*(activations[num_activations - 1]')

    for i = 1:num_layers - 2
        z = z_vectors[num_zs - i]
        error = (n.weights[num_layers - i + 1]') * error * n.activator_derivative(z)
        bias_changes[num_changes - i] = error
        weight_changes[num_changes - i] = error*activations[num_activations - i - 1]'
    end

    return (bias_changes, weight_changes)
end

export Network
end

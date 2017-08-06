include("Network.jl")
import FeedForward

function sigmoid(x::Vector{Float64})
    return 1.0./(1.0 .+ exp(-1 .* x))
end

function sigmoid_prime(x::Vector{Float64})
    return sigmoid(x).*(1 .- sigmoid(z))
end

function cost_derivative(x::Vector{Float64}, y::Vector{Float64})
    return x - y
end

function main()
    network = FeedForward.Network([2, 2, 1], sigmoid, sigmoid_prime, cost_derivative)
    println(network.sizes)
    println(network.biases)
    println(network.weights)
end

main()

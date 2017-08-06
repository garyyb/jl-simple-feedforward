include("Network.jl")
import FeedForward

function sigmoid(x::Vector{Float64})
    return 1.0./(1.0 .+ exp.(-1 .* x))
end

function sigmoid_prime(x::Vector{Float64})
    return sigmoid(x).*(1 .- sigmoid(x))
end

function cost_derivative(x::Vector{Float64}, y::Vector{Float64})
    return x - y
end

function main()
    network = FeedForward.Network([2, 2, 1], sigmoid, sigmoid_prime, cost_derivative)
    println("Starting Biases:")
    println(network.biases)
    println("Starting Weights")
    println(network.weights)
    flush(STDOUT)

    in1 = [0.0;0.0]
    in2 = [0.0;1.0]
    in3 = [1.0;0.0]
    in4 = [1.0;1.0]

    out1 = [0.0]
    out2 = [0.0]
    out3 = [0.0]
    out4 = [1.0]

    data = [(in1, out1), (in2, out2), (in3, out3), (in4, out4)]

    println(data)

    FeedForward.SGD(network, data, 1000, 2, 5.0)
    println("Ending Biases:")
    println(network.biases)
    println("Ending Weights")
    println(network.weights)

    println("Input 0,0 :", FeedForward.evaluate(network, [0.0, 0.0])[1])
    println("Input 0,1 :", FeedForward.evaluate(network, [0.0, 1.0])[1])
    println("Input 1,0 :", FeedForward.evaluate(network, [1.0, 0.0])[1])
    println("Input 1,1 :", FeedForward.evaluate(network, [1.0, 1.0])[1])

    println("Input 0,0 :", FeedForward.evaluate(network, [0.0, 0.0])[1] < 0.5 ? 0 : 1)
    println("Input 0,1 :", FeedForward.evaluate(network, [0.0, 1.0])[1] < 0.5 ? 0 : 1)
    println("Input 1,0 :", FeedForward.evaluate(network, [1.0, 0.0])[1] < 0.5 ? 0 : 1)
    println("Input 1,1 :", FeedForward.evaluate(network, [1.0, 1.0])[1] < 0.5 ? 0 : 1)
end

main()

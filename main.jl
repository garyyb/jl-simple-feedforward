include("Network.jl")
import FeedForward

function sigmoid()

end

function sigmoid_prime()

end

function cost_derivative()

end

function main()
    network = FeedForward.Network([2, 2, 1], sigmoid, sigmoid_prime, cost_derivative)
    println(network.sizes)
    println(network.biases)
    println(network.weights)
end

main()

include("Network.jl")
import FeedForward

function main()
    network = FeedForward.Network([2, 2, 1])
    println(network.sizes)
    println(network.biases)
    println(network.weights)
end

main()

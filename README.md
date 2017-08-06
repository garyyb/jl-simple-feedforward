# Simple Feedforward Neural Network
This is an implementation of a very simple feedforward network in Julia, for the
purpose of practice.

It only supports a single activation function. It's also rather unoptimized.

To use the module, include Network.jl and do
```Julia
import FeedForward
```

To initialise the Network,

```Julia
n = Network(sizes, activator, activator_derivative, cost_derivative)
```
where sizes is an array of the layer sizes, and the other three are the respective
functions.

To train,
```Julia
SGD(n, data, epochs, batch_size, training_rate)
```
where data is an array of 2-tuples of Float64 Vectors containing the input and
desired output.

To evaluate (feedforward),
```Julia
evaluate(n, data)
```
where data is the input vector. The function returns a vector containing the values
in the output layers.

An example of XOR learning is in main.jl.

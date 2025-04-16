First:

- need to make the tests better
- make the zero_grad() function and module
- need to make the tanh backprop better
- need to make the NN more efficient
- leaf flag on which gradients to calculate

Extensions:

- can make this work better with vectors and batches
- create some basic numpy function like numpy dot product
- can try to write some rudiementary CUDA code

- train a small network on MNIST short
- finally, try to apply learnings to C++ code
- finally, try to add vector parallization

Also:

- make the leaf, requires_grad optimization

improvements:

- a) make the is_leaf
- b) clean up @variable.h
- c) fix the requires_grad double calculating
- d) fix the unnecessary shared_ptr making???

<!-- - fix relu -->
<!-- - fix mapping with > ==, etc. -->

- create matmul
- make mnist test run

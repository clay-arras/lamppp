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

- create matmul
- make mnist test run
- TODO: need to make generalized tensors with the silly trick TODO IMPORTANT

sum_test.py
unary_test.py
matmul_test.py
binary_test.py

add some tests for each operation
make it row based instead of column based

add reshaping, 2+d operators, x.reshape(new_shape)), broadcasting
add more pytorch operators (minimum, mean, sigmoid, tanh, x.sqrt(), x.abs(), x.clamp(min, max), x.floor(), x.ceil(),
add leaf DO THIS LAST + requires_grad for backward

getting documentation with
adding benchmarks for speed
add nn module

next step: refactor max
next: add templates and different data types
also: look into lazy eval AND how I CAN INCORPORATE THAT INTO EIGEN TYPES FOR SPEED

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1

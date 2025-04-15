**jacobian basics**
jacobian is just a local derivative in chain rule, but for vectors

- e.g. df/dx

jacobian vector products

jacobian chain rule, tensor chain rule

jacobian for higher dimensional matrices: to calculate the jacobian, you flatten the matrix of both operations into a 1d matrix, then you have a N\*M jacobian

**how to operate on high-dimensional tensors with 2d matrices backend**

- batch processing
- flattening and reshaping (into and out of 2D matrices; with Eigen::Map it's O(1))

resource: https://cs231n.stanford.edu/handouts/derivatives.pdf

matrix operations: matmul
reduction operations: sum, mean, max, min
creation operatioons: zeros, ones, rand, tensor
element-wise operations: +, -, \*, /
unary operations: relu, tanh

add some debug stuff (like cout, refactor some code)

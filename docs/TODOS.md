`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1`

## easy

- add stacktrace, assert, message
- refactor the namings to all be consistent (less or le or less_than); also reconsider namespaces
- add extra stuff (these aren't strictly necessary)
  - power
  - negation
  - add summation with -1 arg
  - element wise access (.get)
  - implement to_vector
- move basic and binary ops together
- change size to numel
- refactor the binary broadcasting backward bit

## medium

- refactor the autograd functions
- make the context be saved from forward execution
- get benchmarks up again
- type broadcasting doesn't work with backward

## hard

- add CPU implementations with OMP
- add better tests including broadcasting tests and codegen; also use pytest instead

  - test broadcasting with autograd
  - test how reshaping, squeeze and expand dims effect gradients
  - test compound operations with autograd
  - test different data types (namely int64, int16, float64) with autograd

## automatic tests

- a) generate a list of variables, set A where len(A) = N
- b) generate a set of ALL of the possible edges between A, s.t. len(E) = N\*(N-1)/2
- c) each union object has a "head"; on merge, we connect s.t. newHead = op(prevHead, newModule)
- d) iterate through E, merging using reshape and expand_dims.
  - reshape checker: pad shorter shape
  - go from left to right; if both 1s, then skip; if a % b == 0, then s.t. the new shape is b, with a/b carried over
  - fallback is to make the shape (N, 1) and (1, M)
- e) after each merge, have an option to apply a unary operation x% of the time (between keepDims=false reduct, unary, and transpose)
- f) stop when all objects have been connected into one "HEAD" variable

`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1`

### current todo

<!-- - refactor the testing folder with Cmake -->
<!-- - refactor the tests to test both CPU and CUDA sequentially -->
<!-- - refactor the tests to test all data types -->


## easy

testing: 
- test .to and make sure it has an implementation in Variable, 
- test type promotion AND backwards type promotion

benchmarking
- make regular binary operations different from broadcasting operations (see if it's speedup)
- benchmark more operations

for release:
- start adding documenatation with Doxygen
- make the code compile cuda optional
- relu, sigmoid, tanh -- define this in nn module
<!-- - add prod, neg, and pow backwards -->

## medium

for cleanliness:
<!-- - refactor: add offsetUtil passed into metaHandler as a dispatch -->
- refactor the autograd functions
<!-- - refactor to have << operator with the macro LMP_CHECK -->
<!-- - refactor: move matrix cuda and cpu to their respective @matrix.cuh and @matrix.hpp -->
- split the copy, empty, fill, and resize files

## hard

- fix github tests
- add better tests with hypothesis

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

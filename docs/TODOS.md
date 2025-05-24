`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1`

### current todo

<!-- - benchmark the backward functions -->
<!-- - make the context be saved from forward execution -->
<!-- - refactor the binary broadcasting backward bit -->

<!-- - TODO: isn't context just grad.data()??????? -->
<!-- - add CPU implementations with OMP -->

- fix github tests
- refactor the tests to test both CPU and CUDA sequentially
- add extra stuff (these aren't strictly necessary)
  - power
  - negation

## easy

benchmarking

- make regular binary operations different from broadcasting operations (see if it's speedup)
- benchmark more operations

for release:

- remove all outside dependencies
- start adding documenatation
- make the code compile cuda optional

## medium

for cleanliness:

- refactor: add offsetUtil passed into metaHandler as a dispatch
- refactor: move matrix cuda and cpu to their respective @matrix.cuh and @matrix.hpp
- refactor the autograd functions
- refactor to have << operator with the macro LMP_CHECK

testing functionality:

- memory leak somewhere
- test .to and make sure it has an implementation in Variable
- type upcasting doesn't work with backward

## hard

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

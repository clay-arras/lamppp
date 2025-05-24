`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1`

### current todo

<!-- - benchmark the backward functions -->
<!-- - make the context be saved from forward execution -->
<!-- - refactor the binary broadcasting backward bit -->

<!-- - TODO: isn't context just grad.data()??????? -->
<!-- - add CPU implementations with OMP -->

- memory leak somewhere
- make regular binary operations different from broadcasting operations (see if it's speedup)
- refactor: add offsetUtil passed into metaHandler as a dispatch
- refactor: move matrix cuda and cpu to their respective @matrix.cuh and @matrix.hpp

## easy

- add extra stuff (these aren't strictly necessary)
  - power
  - negation

## medium

- fix github tests
- refactor the autograd functions
- type upcasting doesn't work with backward
- refactor to have << operator with the macro LMP_CHECK

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

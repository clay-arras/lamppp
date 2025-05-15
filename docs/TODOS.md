cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1

## easy

 <!-- - add better error handling -->
 <!-- - organize repository, delete unused stuff -->
 <!-- - make better namespaces -->
 <!-- - add more data types (int, float 16 - 128) -->
 <!-- - make global macros LMP prefix -->
 <!-- - remove relu -->
 <!-- - move all implementations to .cpp file -->

- make all accessors noexcept and const???
  add extra stuff (these aren't strictly necessary)

- power
- negation
- add summation with -1 arg
- element wise access (.get)
- implement to_vector

## medium

 <!-- - refactor adding operators/methods for codegen??? -->
 <!-- - add more pytorch operators, and remove ReLU  -->
 <!-- - add reshaping, etc. reshape, squeeze, expand dims -->
 <!-- - add some methods from tensor level to variable level -->
 <!-- - add strides (for element wise access) -- not as necessary -->
 <!-- - refactor scalar to make it work with broadcasting -->

- refactor functions s.t. there's less repetitive code
- make the context be saved from forward execution
- add better tests including broadcasting tests and codegen; also use pytest instead
  - test broadcasting
  - test how reshaping, squeeze and expand dims effect gradients
  - test compound operations
  - test different data types (namely int64, int16, float64)

## hard

 <!-- - add broadcasting -->

- add CPU implementations with OMP

- does expand/squeeze dims modify grads, also reshape
- my autograd engine might be wrong, because it takes grad.grad() in GENERAL, not just what that segment of the tree is contributing???
- idea: have all kernels redirect to a central type converter?

@src/tensor/functions
@src/tensor/cuda
@src/autograd/functions

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

todos:

**core**: need to implement the following methods for Tensor/Storage for management and constructors

@copy.hpp - transfers between devices and deepcopies; DataPtr(DataPtr, DeviceType, DeviceType); IN TENSOR
@empty.hpp - allocation factory; DataPtr(size_t, DeviceType); IN STORAGE
@fill.hpp - in-place scalar fill; void(DataPtr, size_t, Scalar); IN TENSOR
@resize.hpp - growing/shrinking data allocation (realloc, cudaMalloc, copy, free); void(DataPtr, size_t, size_t); IN STORAGE + TENSOR
@to.hpp - change the data between devices; DataPtr(DataPtr, DeviceType, DeviceType); IN TENSOR

**decision**: for which classes should I create these methods for?

a) create them for Tensor only: treat Storage as just a block of memory; we don't let it have any member functions and a bare bones constructor, and implement everything else directly on the Tensor

- remove the current constructor / empty out Storage class
- implement new methods in Tensor that operates on Storage
- Storage is just functions as an empty class that groups information

b) create the core implementation for Storage, and then have methods in Tensor calling the methods in Storage. more clear delegation of responsibility. Storage AND Tensor have implementations for Alloc, Copy, Empty, etc.

- constructor only takes in the device, not the allocator
- Storage class is responsible for itself

todos:

  <!-- - move shape and dataType from StorageImpl to TensorImpl -->
  <!-- - implement the five essential function stubs with dispatch -->
  <!-- - put the implementations inside the Storage and Tensor functions -->

new refactor idea:

  <!-- - have backends as singleton -->
  <!-- - use the dispatch function to have a backend_stub function be registered -->
  <!-- - make the function kernels in backend take in the data types as well -->
  <!-- - use macros to explicitly define the possible types of N^2 (since the promoted type is one of the two types) -->
  <!-- - finally, edit the Tensor and Storage constructors s.t. they use DeviceType, not allocator -->
  <!-- - edit the TensorImpl operators s.t. it works with the new backend -->
  <!-- - refactor backend such that it takes in TensorImpls -->
  <!-- - edit backend s.t. it handles dtype promotion -->
  <!-- - also assert that they're on the same device -->
  <!-- - change PIMPLs to have print + << operator separated -->
  <!-- - remove typeguards and stick with cpp20 #pragma once -->

  <!-- - implement type -> DataType conversion -->
  <!-- - refactor StorageImpl  -->
  <!-- - add DataType in copy definition -->
  <!-- - add copy kernels for device to device and GPU allocators -->

- add new methods in Tensor for construction
- change assertions to better error handling
- maybe change the static methods in TensorImpl
- use the regular data instead of accessors data()
- GET RID OF DEFAULT CONSTRUCTORS

- refactor using empty to alloc without cudaMalloc
  using backend_fn = std::shared_ptr<AbstractBackend> (\*)();

Problem:

  <!-- - we have backend operating on Storage; however, now storage has been refactored to JUST an array of bytes; therefore, we need to add in additional information; such as typecasting, broadcasting; SOLUTION: just add extra parameters specifying the type -->
  <!-- - another decision: should we use CRTP instead of virtual here, s.t. we only hav to implement getInstance once (also it's slightly faster) -->
  <!-- - another decision: should I make backend on STORAGE ( which has raw bytes; or should I make it on TensorImpl, which has some more information that I need, i.e. dtype, shape, etc.?) -->
  <!-- - another decision: should Storage know about datatype? because if the helper kernels in storage are datatype aware, then Storage needs to as well; decision: have fill and copy be in Tensor, remove copy in storage; and just have a simple pointer for empty array in Storage -->

- do we want the mallocs to be in the host function OR do we want each thing to handle it?; the latter, use default cuda alloc OR empty

- idea: must be size in bytes not being passed through?
- only for very large arrays

- @copy.cu print out src_type and dest_type and their sizes, make sure they're distinct
- @tensor.hpp the span code might be wrong, test it in a separate console
- check if the kernels take in size or byte_size

- the problem is at the end of large tensors, it's filled with 1s or -1s

test checks

- need to check all operations -> test suites
- need to check forward and backward -> assert
- need to check shape and data -> assert

next todos:

- refactor adding operators/methods for codegen???
- add sum, mean, min, max, etc.
- add reshaping, strides, etc. (do not fuck with non-contiguous arrays, probably)
- organize repository, delete unused stuff
- add reshaping, 2+d operators, x.reshape(new_shape)), broadcasting
- add more pytorch operators (minimum, mean, sigmoid, tanh, x.sqrt(), x.abs(), x.clamp(min, max), x.floor(), x.ceil(),

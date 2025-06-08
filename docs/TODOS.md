`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_CUDA=ON -DENABLE_COVERAGE=ON`
`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_CUDA=OFF -DENABLE_COVERAGE=OFF -DCMAKE
_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++`

### current todo

- next step: try to see if its because we need memory pool (its the allocating a new cuda malloc that's taking a long time)
- benching cudaMalloc
- try figuring out what percent of the speed is from Variable

testing: 
- test type promotion AND backwards type promotion

benchmarking
<!-- - make regular binary operations different from broadcasting operations (see if it's speedup) -->
- relu, sigmoid, tanh -- define this in nn module
- add default values - quick

- fix github tests
- add better tests with hypothesis



why is Variable so slow compared to Tensor: 
DONE - when operations are done, even with requires_grad=false, we're creating the grad Tensor


- use cuda Memset instead of zeros-like
<!-- - Variables being used in the computations -- NEED to switch to tensors -->
<!-- - its creating a new class for every operation (but this shouldn't take too much time) -->

realization: there is ZERO purpose to const& a PIMPL
test the simplest cuda addition between two straight tensors, with +
<!-- LMP_CHECK(requires_grad()) << "Need to requires_grad if calling" -->

## steps
operation inline
variable op fact
create forwardfunction
call tensor opt


use perf and flamegraph
for some reason VM with cuda is much slower than Macbook CPU version
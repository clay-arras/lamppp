`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_CUDA=ON -DENABLE_COVERAGE=ON`
`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_CUDA=OFF -DENABLE_COVERAGE=OFF -DCMAKE
_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++`

`nsys profile --trace=cuda,nvtx,osrt --stats=true --sample=cpu -o bench_test ./build/benchmarks/bench_test`
`nsys stats --report=osrt_sum bench_test.nsys-rep >> log.cpu.txt`

### current todo

- next step: try to see if its because we need memory pool (its the allocating a new cuda malloc that's taking a long time)
- benching cudaMalloc
- try figuring out what percent of the speed is from Variable

testing: 
- test type promotion AND backwards type promotion

benchmarking
- relu, sigmoid, tanh -- define this in nn module
- add default values - quick

- fix github tests
- add better tests with hypothesis



# why is Variable so slow compared to Tensor: 

- look into cudaMemcpy*Async
- use cuda Memset instead of zeros-like

realization: there is ZERO purpose to const& a PIMPL
test the simplest cuda addition between two straight tensors, with +

## steps
operation inline
variable op fact
create forwardfunction
call tensor opt


use perf and flamegraph
for some reason VM with cuda is much slower than Macbook CPU version


# TODO but more -----------------
lets focus on unary for now


@native/unary_ops.hpp: has the stub dispatch
@cuda/kernels.cuh: calls TensorMetaHandler and the dispatch
@cpu/meta_handler.hpp: takes no time at all
@cuda/unary.cuh: is the dispatch, the kernel launcher, and the kernel




## benchmarks: 
Macro LMP_DISPATCH_TYPE: 1ns
TensorImpl copy (512 x 512): 20ns
TensorMetaHandler initialize for unary (512 x 512): 800ns
unary_dispatch_handler (512 x 512): 200 000ns
kernels.cu op_cuda (512 x 512): 200 000ns
full benchmark with Tensor (512 x 512): 850 000ns

TensorImpl initialize from vector (512 x 512): 890000ns
<!-- maybe it's copy_cuda? -- copy cuda is not used in this -->


## ---

new discrepancy: the speed between grad and no-grad for variable is around 8x
#include "lamppp/inductor/nvrtc/nvrtc_backend.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <mutex>
#include <string>
#include <vector>

#include "lamppp/common/assert.hpp"
#include "lamppp/inductor/nvrtc/codegen.hpp"
#include "lamppp/inductor/nvrtc/fused_graph.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/lazy/lazy_backend.hpp"
#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/lazy/realize.hpp"
#include "lamppp/tensor/storage.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

#define NVRTC_CHECK(x)                                      \
  do {                                                      \
    const nvrtcResult nvrtc_r = (x);                        \
    LMP_CHECK(nvrtc_r == NVRTC_SUCCESS)                     \
        << "NVRTC error: " << nvrtcGetErrorString(nvrtc_r); \
  } while (0)

#define CU_CHECK(x)                                                     \
  do {                                                                  \
    const CUresult cu_r = (x);                                          \
    if (cu_r != CUDA_SUCCESS) {                                         \
      const char* cu_s = nullptr;                                       \
      cuGetErrorString(cu_r, &cu_s);                                    \
      LMP_CHECK(false) << "CUDA driver error: " << (cu_s ? cu_s : "?"); \
    }                                                                   \
  } while (0)

#define CUDART_CHECK(x)                                            \
  do {                                                             \
    const cudaError_t cudart_r = (x);                              \
    LMP_CHECK(cudart_r == cudaSuccess)                             \
        << "CUDA runtime error: " << cudaGetErrorString(cudart_r); \
  } while (0)

namespace lmp::inductor {

namespace {

struct LoadedKernel {
  CUmodule module;
  CUfunction func;
};

// Compile the generated source string to a CUBIN for THIS device's arch via
// NVRTC, then load it into the current (cudart primary) context.
LoadedKernel compile_and_load(const std::string& src) {
  nvrtcProgram prog;
  NVRTC_CHECK(
      nvrtcCreateProgram(&prog, src.c_str(), "fused.cu", 0, nullptr, nullptr));

  int dev = 0;
  CUDART_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp props;
  CUDART_CHECK(cudaGetDeviceProperties(&props, dev));
  const std::string arch = "--gpu-architecture=sm_" +
                           std::to_string(props.major) +
                           std::to_string(props.minor);
  const char* opts[] = {arch.c_str(), "--std=c++17"};

  const nvrtcResult cres = nvrtcCompileProgram(prog, 2, opts);
  if (cres != NVRTC_SUCCESS) {
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(prog, log.data());
    LMP_CHECK(false) << "NVRTC compile failed:\n"
                     << log << "\n--- generated source ---\n"
                     << src;
  }

  size_t cubin_size = 0;
  NVRTC_CHECK(nvrtcGetCUBINSize(prog, &cubin_size));
  std::vector<char> cubin(cubin_size);
  NVRTC_CHECK(nvrtcGetCUBIN(prog, cubin.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));

  static std::once_flag init_flag;
  std::call_once(init_flag, [] { CU_CHECK(cuInit(0)); });

  LoadedKernel k;
  CU_CHECK(cuModuleLoadData(&k.module, cubin.data()));
  CU_CHECK(cuModuleGetFunction(&k.func, k.module, kFusedKernelName));
  return k;
}

void launch(CUfunction f, std::vector<void*>& args, size_t n) {
  const unsigned int block = 256;
  const unsigned int grid = static_cast<unsigned int>((n + block - 1) / block);
  CU_CHECK(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, nullptr, args.data(),
                          nullptr));
  CU_CHECK(cuCtxSynchronize());
}

}  // namespace

void NVRTCInductorBackend::realize(tensor::TensorImpl* impl) {
  if (!impl->is_deferred())
    return;

  // Non-fusible root (e.g. a broadcasting binary op): the fused kernel's flat
  // same-shape index model doesn't apply. Realize its inputs and run it eagerly.
  if (!impl->lazy_op()->is_fusible()) {
    for (const std::shared_ptr<tensor::TensorImpl>& in :
         impl->lazy_op()->inputs)
      tensor::realize(in.get());
    impl->lazy_op()->run_eager(*impl);
    return;
  }

  // Fusible root: partition into one group, codegen one kernel, JIT, launch.
  FusedGraph g = build_fused_graph(impl);  // realizes all boundary inputs
  const std::string src = codegen_kernel(g);

  const size_t n = impl->numel();
  const size_t bytes = LMP_DISPATCH_ALL_TYPES(
      impl->type(), [&] { return n * sizeof(scalar_t); });

  // Allocate the output FIRST: this cudart call guarantees the primary CUDA
  // context is current on this thread before any driver-API call below.
  tensor::Storage out(bytes, tensor::DeviceType::CUDA);

  if (n == 0) {  // empty tensor: nothing to launch
    impl->set_realized(out);
    return;
  }

  LoadedKernel k = compile_and_load(src);

  // Kernel args are (out, in0..in{K-1}, n). cuLaunchKernel wants a pointer to
  // each argument *value*, so the value lvalues must stay alive (and the vector
  // holding the input device pointers must not reallocate) until after launch.
  void* out_ptr = out.data();
  std::vector<void*> in_ptrs;
  in_ptrs.reserve(g.inputs.size());
  for (tensor::TensorImpl* leaf : g.inputs)
    in_ptrs.push_back(leaf->storage().data());
  size_t n_arg = n;

  std::vector<void*> args;
  args.reserve(g.inputs.size() + 2);
  args.push_back(&out_ptr);
  for (void*& p : in_ptrs)
    args.push_back(&p);
  args.push_back(&n_arg);

  launch(k.func, args, n);
  CU_CHECK(cuModuleUnload(k.module));  // no compile cache yet: unload now

  impl->set_realized(out);
}

// Master switch for kernel fusion. Three separate concerns, kept distinct:
//   - LMP_ENABLE_CUDA  : whether this TU (the NVRTC backend) is compiled at all
//   - LMP_ENABLE_FUSION: whether the lazy backend registers (this gate)
//   - backend(device)  : the op-shim's per-op defer decision (reads registration)
// When LMP_ENABLE_FUSION is off, no backend registers, backend(CUDA) is null,
// and every op stays on the eager path (bit-identical to a no-inductor build).
// NOTE(future): when fusion is off we should go further and not build/link the
// inductor library at all, rather than building it and compiling out only the
// registrar. Left as a follow-up.
#ifdef LMP_ENABLE_FUSION
LMP_REGISTER_LAZY_BACKEND(tensor::DeviceType::CUDA, NVRTCInductorBackend)
#endif

}  // namespace lmp::inductor

#undef NVRTC_CHECK
#undef CU_CHECK
#undef CUDART_CHECK

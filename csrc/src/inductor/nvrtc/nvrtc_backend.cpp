#include "lamp3/inductor/nvrtc/nvrtc_backend.hpp"

namespace lmp::inductor {

namespace {

struct LoadedKernel {
  CUmodule module;
  CUfunction func;
};

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

}

void NVRTCInductorBackend::realize(tensor::TensorImpl* impl) {
  if (!impl->is_deferred())
    return;

  if (!impl->lazy_op()->is_fusible()) {
    for (const std::shared_ptr<tensor::TensorImpl>& in :
         impl->lazy_op()->inputs)
      tensor::lazy::realize(in.get());
    impl->lazy_op()->run_eager(*impl);
    return;
  }

  FusedGraph g = build_fused_graph(impl);
  const std::string src = codegen_kernel(g);

  const size_t n = impl->numel();
  const size_t bytes = LMP_DISPATCH_ALL_TYPES(
      impl->type(), [&] { return n * sizeof(scalar_t); });

  tensor::Storage out(bytes, tensor::DeviceType::CUDA);

  if (n == 0) {
    impl->set_realized(out);
    return;
  }

  LoadedKernel k = compile_and_load(src);

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
  CU_CHECK(cuModuleUnload(k.module));

  impl->set_realized(out);
}

LMP_REGISTER_LAZY_BACKEND(tensor::DeviceType::CUDA, NVRTCInductorBackend)

}

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "lamppp/inductor/nvrtc/fused_graph.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::inductor {

inline constexpr const char* kFusedKernelName = "fused_kernel";

std::string codegen_kernel(const FusedGraph& g);

}

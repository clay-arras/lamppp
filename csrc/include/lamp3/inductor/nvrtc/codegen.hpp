#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "lamp3/inductor/nvrtc/fused_graph.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/lazy/lazy_function.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::inductor {

inline constexpr const char* kFusedKernelName = "fused_kernel";

std::string codegen_kernel(const FusedGraph& g);

}

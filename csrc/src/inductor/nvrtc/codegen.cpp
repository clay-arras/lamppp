#include "lamppp/inductor/nvrtc/codegen.hpp"

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "lamppp/inductor/nvrtc/fused_graph.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::inductor {
namespace {

// Replace positional placeholders {0},{1},... in an op's codegen_expr template
// with the operand strings in `args`.
std::string substitute(const std::string& tmpl,
                       const std::vector<std::string>& args) {
  std::string out = tmpl;
  for (size_t j = 0; j < args.size(); ++j) {
    const std::string ph = "{" + std::to_string(j) + "}";
    for (size_t pos = out.find(ph); pos != std::string::npos;
         pos = out.find(ph, pos + args[j].size())) {
      out.replace(pos, ph.size(), args[j]);
    }
  }
  return out;
}

}  // namespace

std::string codegen_kernel(const FusedGraph& g) {
  // ---- name every value ----
  // boundary input at slot N -> "s{N}" (loaded from param "in{N}")
  // interior node at order index M -> "v{M}"
  std::unordered_map<tensor::TensorImpl*, std::string> name;
  name.reserve(g.inputs.size() + g.order.size());
  for (size_t i = 0; i < g.inputs.size(); ++i)
    name[g.inputs[i]] = "s" + std::to_string(i);
  for (size_t m = 0; m < g.order.size(); ++m)
    name[g.order[m]] = "v" + std::to_string(m);

  std::ostringstream os;

  // ---- signature: (out, in0..in{K-1}, n) ----
  os << "extern \"C\" __global__ void " << kFusedKernelName << "(\n";
  os << "    " << tensor::to_cname(g.output->type()) << "* out";
  for (size_t i = 0; i < g.inputs.size(); ++i)
    os << ",\n    const " << tensor::to_cname(g.inputs[i]->type()) << "* in"
       << i;
  os << ",\n    size_t n) {\n";

  // ---- flat index guard ----
  os << "  size_t i = blockIdx.x * blockDim.x + threadIdx.x;\n";
  os << "  if (i >= n) return;\n";

  // ---- load boundary inputs (one typed load each) ----
  for (size_t i = 0; i < g.inputs.size(); ++i)
    os << "  " << tensor::to_cname(g.inputs[i]->type()) << " s" << i << " = in"
       << i << "[i];\n";

  // ---- interior nodes in evaluation order ----
  for (size_t m = 0; m < g.order.size(); ++m) {
    tensor::TensorImpl* node = g.order[m];
    tensor::LazyFunction* fn = node->lazy_op().get();
    // Per-op promotion: compute dtype == this node's output dtype (type_upcast,
    // stamped by infer_output). Cast every operand to it before the op.
    const std::string ct = tensor::to_cname(node->type());
    std::vector<std::string> args;
    args.reserve(fn->inputs.size());
    for (const std::shared_ptr<tensor::TensorImpl>& in : fn->inputs)
      args.push_back("static_cast<" + ct + ">(" + name.at(in.get()) + ")");
    os << "  " << ct << " v" << m << " = "
       << substitute(fn->codegen_expr(), args) << ";\n";
  }

  // ---- store result ----
  os << "  out[i] = " << name.at(g.output) << ";\n";
  os << "}\n";

  return os.str();
}

}  // namespace lmp::inductor

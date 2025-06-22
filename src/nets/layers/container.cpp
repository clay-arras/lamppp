#include "lamppp/nets/any.hpp"
#include "lamppp/nets/layers/container.hpp"
#include <string>
#include <utility>

namespace lmp::nets {

SequentialImpl::SequentialImpl(std::vector<AnyModule> layers)
    : layers_(std::move(layers)) {
      for (size_t i = 0; i < layers_.size(); i++) {
        register_module("[Sequential] Layer " + std::to_string(i),
                        layers_[i].getImpl());
      }
    };

std::any SequentialImpl::forward(const std::vector<std::any>& in) const {
  LMP_CHECK(layers_.size() >= 1) << "Must have at least one layer";
  std::any out = layers_[0].call(in);
  for (size_t i = 1; i < layers_.size(); i++) {
    out = layers_[i].call({out});
  }
  return out;
}

}
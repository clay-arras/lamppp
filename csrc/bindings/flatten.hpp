#pragma once

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace lmp::bindings {

/**
 * @brief Recursively walk a nested python list/tuple, flattening it into a
 * row-major @p data buffer while inferring the @p shape at each depth. Mirrors
 * the pure-python `_flatten_array` helper in rushlite.
 */
inline void flatten_nested(const py::handle& obj, size_t depth,
                           std::vector<double>& data,
                           std::vector<size_t>& shape) {
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    auto seq = py::reinterpret_borrow<py::sequence>(obj);
    size_t len = seq.size();
    if (shape.size() == depth) {
      shape.push_back(len);
    } else if (shape[depth] != len) {
      throw std::runtime_error("rushlite: array must be uniform");
    }
    for (const auto& item : seq) {
      flatten_nested(item, depth + 1, data, shape);
    }
  } else {
    data.push_back(obj.cast<double>());
  }
}

/**
 * @brief Flatten a nested python list/tuple into (row-major data, shape).
 */
inline std::pair<std::vector<double>, std::vector<size_t>> flatten_array(
    const py::handle& obj) {
  std::vector<double> data;
  std::vector<size_t> shape;
  flatten_nested(obj, 0, data, shape);
  return {std::move(data), std::move(shape)};
}

}  // namespace lmp::bindings

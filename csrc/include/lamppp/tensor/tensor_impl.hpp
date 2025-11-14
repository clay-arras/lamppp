#pragma once

#include <iostream>
#include <numeric>
#include <vector>
#include "data_type.hpp"
#include "device_type.hpp"
#include "dispatch_type.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/utils/align_utils.hpp"
#include "lamppp/tensor/native/memory_ops.hpp"
#include "lamppp/tensor/storage.hpp"

namespace lmp::tensor {

/// @internal
/**
 * @brief  Main implementation class for Tensor object
 *
 * @details `TensorImpl` contains a few core members: `type_`, `shape_`, and `data_`
 * Note that similar to Pytorch, Tensor/TensorImpl is not responsible for the 
 * low-level data storage -- note that `TensorImpl` has no member called `device_`.
 * That is managed by `Storage`.
 *
 * @see Tensor, Storage
 */
class TensorImpl {
 public:
  /**
   * @brief Construct a TensorImpl from a vector of data
   * 
   * @tparam T The element type of the input data vector
   * @param data   Flat vector containing the tensor data in row-major order
   * @param shape  Dimensions of the tensor, e.g. {28, 28} for a 2D tensor
   * @param device Target device where the tensor will be stored (CPU/GPU)
   * @param dtype  Data type for the tensor elements (may differ from T)
   * 
   * @throws std::runtime_error if data.size() != product of shape dimensions
   * 
   * @details This constructor allocates storage on the specified device and
   * copies the input data.
   *
   * @note Note that the input data's type T does NOT have to be the same as dtype. 
   * i.e. inputting dtype = DataType::Float64, but data = std::vector<int>{...} is valid
   */
  template <typename T>
  explicit TensorImpl(const std::vector<T>& data,
                      const std::vector<size_t>& shape, DeviceType device,
                      DataType dtype)
      : data_(LMP_DISPATCH_ALL_TYPES(
            dtype,
            [&] { return Storage(data.size() * sizeof(scalar_t), device); })),
        shape_(shape),
        type_(dtype),
        strides_(std::vector<detail::stride_t>(shape.size())),
        numel_(shape.empty() ? 0
                             : std::accumulate(shape.begin(), shape.end(), 1,
                                               std::multiplies<>())) {
    LMP_CHECK(data.size() == numel_) <<
              "Size mismatch, product of shape must equal num elements";
    DataType src_dtype = TypeMeta<T>::value;
    ops::copy_stub()(DeviceType::CPU, device, data.data(),
                                data_.data(), numel_, src_dtype, type_);
    update_strides();
  }
  /// @internal
  /// @note: this should not be used by the user.
  explicit TensorImpl(Storage storage, const std::vector<size_t>& shape,
                      DataType dtype);
  /// @endinternal

  void* data() const noexcept;
  DataType type() const noexcept;
  DeviceType device() const noexcept;
  const std::vector<size_t>& shape() const noexcept;
  const std::vector<detail::stride_t>& strides() const noexcept;
  size_t numel() const noexcept;

  TensorImpl reshape(std::vector<size_t> new_shape);
  TensorImpl squeeze(size_t dim);
  TensorImpl expand_dims(size_t dim);
  Scalar index(const std::vector<size_t>& idx);

  void copy(const TensorImpl& other) const;
  void fill(Scalar item) const;
  void print(std::ostream& os) const;

 private:
  friend class Tensor;

  /**
  * @brief a simple function to recalculate the strides 
  *
  * @note after each operation that involves changing the shape, update_strides_()
  * MUST be called for the broadcasting to work correctly. 
  */
  void update_strides();

  DataType type_;
  Storage data_;
  size_t numel_;
  std::vector<size_t> shape_;
  std::vector<detail::stride_t> strides_;
};
/// @endinternal

}  // namespace lmp::tensor
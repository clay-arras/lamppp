#include "lamppp/tensor/native/memory_ops.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(copy_fn, copy_stub);
LMP_DEFINE_DISPATCH(empty_fn, empty_stub);
LMP_DEFINE_DISPATCH(fill_fn, fill_stub);
LMP_DEFINE_DISPATCH(resize_fn, resize_stub);

Tensor to(const Tensor& a, DeviceType to_device) {
  LMP_CHECK(a.device() != to_device)
      << "Device argument must be different from current device.";

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&]() {
    Storage new_storage(a.numel() * sizeof(scalar_t), to_device);
    copy_stub()(a.device(), to_device, a.data(), new_storage.data(), a.numel(),
                a.type(), a.type());
    TensorImpl new_impl(new_storage, a.shape(), a.type());
    return detail::UnsafeTensorAccessor::fromImpl(
        std::make_shared<TensorImpl>(new_impl));
  });
}

}  // namespace lmp::tensor::ops

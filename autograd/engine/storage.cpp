#include "autograd/engine/storage.hpp"
#include <algorithm>
#include <numeric>

namespace autograd {

class StorageImpl {
public:
    void* data_ptr;
    size_t size;
    DataType type;
    std::vector<size_t> shape;

    explicit StorageImpl(const std::vector<size_t>& shape, DataType type)
        : shape(shape), type(type) {
        size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        DISPATCH_ALL_TYPES(type, [&]{
            data_ptr = static_cast<void *>(new scalar_t[size]);
        });
    }

    template<typename T>
    explicit StorageImpl(const std::vector<T>& data, const std::vector<size_t>& shape, DataType type)
        : shape(shape), type(type) {
        size = data.size();
        DISPATCH_ALL_TYPES(type, [&]{
            data_ptr = static_cast<void *>(new scalar_t[size]);
            std::transform(data.begin(), data.end(), static_cast<scalar_t*>(data_ptr),
                [](const T& val) { return static_cast<scalar_t>(val); });
        });
    }

    // Copy constructor
    StorageImpl(const StorageImpl& other) 
        : shape(other.shape), type(other.type), size(other.size) {
        DISPATCH_ALL_TYPES(type, [&]{
            data_ptr = static_cast<void *>(new scalar_t[size]);
            std::memcpy(data_ptr, other.data_ptr, size * sizeof(scalar_t));
        });
    }

    ~StorageImpl() {
        DISPATCH_ALL_TYPES(type, [&]{
            delete[] static_cast<scalar_t*>(data_ptr);
        });
    }
};

// Storage implementation

Storage::Storage(const std::vector<size_t>& shape, DataType type)
    : impl(std::make_unique<StorageImpl>(shape, type)) {}

template<typename T>
Storage::Storage(const std::vector<T>& data, const std::vector<size_t>& shape, DataType type)
    : impl(std::make_unique<StorageImpl>(data, shape, type)) {
}

Storage::~Storage() = default;

Storage::Storage(const Storage& other)
    : impl(std::make_unique<StorageImpl>(*other.impl)) {}

Storage& Storage::operator=(const Storage& other) {
    if (this != &other) {
        impl = std::make_unique<StorageImpl>(*other.impl);
    }
    return *this;
}

Storage::Storage(Storage&& other) noexcept = default;

Storage& Storage::operator=(Storage&& other) noexcept = default;

void* Storage::data() const {
    return impl->data_ptr;
}

size_t Storage::size() const {
    return impl->size;
}

DataType Storage::type() const {
    return impl->type;
}

const std::vector<size_t>& Storage::shape() const {
    return impl->shape;
}

template Storage::Storage(const std::vector<float>&, const std::vector<size_t>&, DataType);
template Storage::Storage(const std::vector<double>&, const std::vector<size_t>&, DataType);
template Storage::Storage(const std::vector<int>&, const std::vector<size_t>&, DataType);
template Storage::Storage(const std::vector<int64_t>&, const std::vector<size_t>&, DataType);

} // namespace autograd 
#ifndef _VALUE_MEMORY_POOL_H_
#define _VALUE_MEMORY_POOL_H_

#include <mutex>
#include <vector>

/**
 * @brief A class that manages a pool of Value objects for efficient memory usage.
 */
class ValueMemoryPool {
 private:
  static thread_local std::vector<void*> local_pool_;
  std::vector<void*> global_pool_;
  std::mutex global_mutex_;

  size_t block_size_;
  size_t initial_local_size_;

  void initialize_local_pool() const {
    local_pool_.reserve(initial_local_size_);
    for (size_t i = 0; i < initial_local_size_; i++) {
      local_pool_.push_back(::operator new(block_size_));
    }
  }

 public:
  /**
   * @brief Default constructor for ValueMemoryPool.
   */
  explicit ValueMemoryPool(size_t size, size_t data_size);

  /**
   * @brief Allocates a new Value object and adds it to the pool.
   * @return A pointer to the newly created Value object.
   */
  void* allocate();

  /**
   * @brief Deallocates a Value object, removing it from the pool.
   * @param value A pointer to the Value object to deallocate.
   */
  void deallocate(void* value);

  /**
   * @brief Resizes the memory pool to a new size.
   * @param new_size The new size for the memory pool.
   */
  void resize(size_t new_size);
};

#endif  // _VALUE_MEMORY_POOL_H_

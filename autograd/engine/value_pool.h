#ifndef _VALUE_MEMORY_POOL_H_
#define _VALUE_MEMORY_POOL_H_

#include <vector>
#include <mutex>

/**
 * @brief A class that manages a pool of Value objects for efficient memory usage.
 */
class ValueMemoryPool {
 private:
  std::vector<void*> pool_;  ///< Pool of shared pointers to Value objects.
  std::mutex mutex_;

 public:
  /**
   * @brief Default constructor for ValueMemoryPool.
   */
  explicit ValueMemoryPool(size_t size);

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

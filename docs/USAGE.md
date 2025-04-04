Based on analyzing your code, I'll explain why SharedValue matrix operations are so slow and provide optimizations ordered by effectiveness.

## Root Causes of 700x Slowdown

The SharedValue implementation creates significant overhead for matrix operations because:

1. **Every operation builds a computational graph** - Each element-wise operation allocates memory and constructs the autodiff structure
2. **Excessive heap allocations** - Each operation creates new shared_ptr instances
3. **Pointer indirection** - Accessing values requires dereferencing pointers
4. **Reference counting overhead** - std::shared_ptr has atomic counter operations
5. **unordered_set usage** - Hash-based containers add lookup overhead

## Most Effective Optimizations

1. **Implement a No-Grad Mode (Highest Impact)**

```cpp
class SharedValue {
private:
    static thread_local bool no_grad_mode_;
public:
    static void set_no_grad(bool val) { no_grad_mode_ = val; }

    SharedValue operator*(const SharedValue& other) const {
        if (no_grad_mode_) {
            // Fast path: just do the computation
            return SharedValue(value_->data * other.value_->data);
        }
        // Regular path with graph building
        return SharedValue(value_ * other.value_);
    }
};

// Usage
SharedValue::set_no_grad(true);
auto result = mat1 * mat2; // Fast computation without autodiff
SharedValue::set_no_grad(false);
```

2. **Memory Pooling for Value Objects**

```cpp
class ValuePool {
private:
    std::vector<std::unique_ptr<Value[]>> blocks_;
    Value* free_list_ = nullptr;

public:
    Value* allocate() {
        if (!free_list_) {
            // Allocate a new block
            const size_t BLOCK_SIZE = 1024;
            auto block = std::make_unique<Value[]>(BLOCK_SIZE);

            // Set up free list
            for (size_t i = 0; i < BLOCK_SIZE-1; ++i) {
                new (&block[i]) Value();
                *reinterpret_cast<Value**>(&block[i]) = &block[i+1];
            }
            new (&block[BLOCK_SIZE-1]) Value();
            *reinterpret_cast<Value**>(&block[BLOCK_SIZE-1]) = nullptr;

            free_list_ = &block[0];
            blocks_.push_back(std::move(block));
        }

        Value* result = free_list_;
        free_list_ = *reinterpret_cast<Value**>(free_list_);
        return result;
    }

    void deallocate(Value* ptr) {
        *reinterpret_cast<Value**>(ptr) = free_list_;
        free_list_ = ptr;
    }
};
```

3. **Specialized Matrix Operations**

```cpp
// Add specialized matrix multiplication
template<typename Derived1, typename Derived2>
Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>
fast_matrix_multiply(const Eigen::MatrixBase<Derived1>& a,
                     const Eigen::MatrixBase<Derived2>& b) {
    // Extract raw data
    Eigen::MatrixXd a_data(a.rows(), a.cols());
    Eigen::MatrixXd b_data(b.rows(), b.cols());

    // Copy data to raw matrices
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j)
            a_data(i, j) = a(i, j).getData();

    for (int i = 0; i < b.rows(); ++i)
        for (int j = 0; j < b.cols(); ++j)
            b_data(i, j) = b(i, j).getData();

    // Perform optimized multiplication
    Eigen::MatrixXd result = a_data * b_data;

    // Reconstruct SharedValue matrix
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>
        sv_result(result.rows(), result.cols());

    for (int i = 0; i < result.rows(); ++i)
        for (int j = 0; j < result.cols(); ++j)
            sv_result(i, j) = SharedValue(result(i, j));

    return sv_result;
}
```

4. **Replace unordered_set with vector**

```cpp
// In Value constructor
Value::Value(double data, std::vector<std::shared_ptr<Value>> children, double grad)
    : data(data), grad(grad) {
    prev.reserve(children.size());
    prev.insert(prev.end(), children.begin(), children.end());
}
```

5. **Use expression templates**

```cpp
template<typename Op, typename LHS, typename RHS>
class Expression {
    const LHS& lhs_;
    const RHS& rhs_;

public:
    Expression(const LHS& l, const RHS& r) : lhs_(l), rhs_(r) {}

    // Evaluate only when needed
    double eval() const {
        return Op::apply(lhs_.eval(), rhs_.eval());
    }

    // Convert to SharedValue when required
    operator SharedValue() const {
        return SharedValue(eval());
    }
};

// Multiplication operator
struct MultiplyOp {
    static double apply(double a, double b) { return a * b; }
};

// Overload * for expressions
template<typename LHS, typename RHS>
Expression<MultiplyOp, LHS, RHS> operator*(const LHS& lhs, const RHS& rhs) {
    return Expression<MultiplyOp, LHS, RHS>(lhs, rhs);
}
```

Start with the no-grad mode implementation as it provides the fastest path to improving performance while requiring minimal changes to your existing code.

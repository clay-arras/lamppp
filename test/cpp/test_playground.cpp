// Test: Can we have a templated function that gets called by an untemplated function
// which uses an enum + switch to dispatch to the correct template instantiation?

#include <any>
#include <iostream>
#include <type_traits>

// Define supported types via enum
enum class DataType {
  Float,
  Double
};

// Templated function that we want to call
template<typename T>
void print_type() {
  std::cout << "Type is: " << typeid(T).name() << std::endl;
}

class TestClass {
public:
  DataType type;

  TestClass(DataType type) : type(type) {};

  // Non-templated function that returns std::any
  std::any dispatch_print() {
    switch(type) {
      case DataType::Float:
        return dispatch_print_impl<float>();
      case DataType::Double:
        return dispatch_print_impl<double>();
      default:
        std::cerr << "Unsupported type" << std::endl;
        return std::any();
    }
  }

private:
  // Private templated implementation
  template <typename T>
  std::any dispatch_print_impl() {
    print_type<T>();
    return T(1);
  }
};

// Example usage:
void run_test() {
  TestClass t(DataType::Float);
  std::any result = t.dispatch_print();
  
  // To get the value back:
  float value = std::any_cast<float>(result);
  std::cout << "Value: " << value << std::endl;
}

int main() {
  run_test();
}
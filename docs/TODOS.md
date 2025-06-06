`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_CUDA=ON -DENABLE_COVERAGE=ON`

### current todo

testing: 
- test type promotion AND backwards type promotion

benchmarking
- make regular binary operations different from broadcasting operations (see if it's speedup)
- benchmark more operations
- relu, sigmoid, tanh -- define this in nn module

- fix github tests
- add better tests with hypothesis


# benchmarking
- look at pytorch benchmarks and design common
- bench cuda, different sizes, different ops? (or just float32)

op types: 
- binary, no broadcast
- binary backward

- binary, w/ broadcast
- expand backward

- unary
- unary backward

- unary w/ restrictions/extra stuff: 
  - exp, log, sqrt, tangent, clamp
- backward

- reduct ops

#### properties
sizes: 128, 256, 512, 1024
reduct: [256, 512], [256, 32], [64, 512], [64, 32]

value initializations: 
devices: cpu, cuda
data types: float32


```cpp
Variable fn() {
  return ...; // cycle for size, device, value init, etc.
}

std::vector<Function> operators;

where operators tells us how to get the ith variable

function body: 
  a) backward unary
  b) backward binary
  c) unary
  d) binary


template <typename NumOps>
struct OperatorSpec {
    std::string name;
    std::function<Variable(std::array<Variable, NumOps>)> binary_fn; 
    std::array<std::function<Variable()>, NumOps> initializers; 
};

// how to do sum? 
// how to have extra parameters?





```


second idea: make it class based instead
- have an initialization function to initialize the variables
- then have a () operator to run the variable
- have static parameters (with default settings, but option to override)
- also add register function (takes in OperatorConfig???) so that way each function is responsible for registering their own thing

benefits: easier for functions like sum, which you want register axis=1, axis=0, etc.
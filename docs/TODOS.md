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

### automatic tests


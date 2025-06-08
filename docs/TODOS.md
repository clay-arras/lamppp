`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_CUDA=ON -DENABLE_COVERAGE=ON`

### current todo

- next step: try to see if its because we need memory pool (its the allocating a new cuda malloc that's taking a long time)
- benching cudaMalloc
- 

testing: 
- test type promotion AND backwards type promotion

benchmarking
<!-- - make regular binary operations different from broadcasting operations (see if it's speedup) -->
- relu, sigmoid, tanh -- define this in nn module
- add default values - quick

- fix github tests
- add better tests with hypothesis


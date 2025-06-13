`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_CUDA=ON -DENABLE_COVERAGE=ON`
`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DENABLE_CUDA=OFF -DENABLE_COVERAGE=OFF -DCMAKE
_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++`

### current todo

testing: 
- test type promotion AND backwards type promotion

benchmarking
- relu, sigmoid, tanh -- define this in nn module
- add default values - quick

- fix github tests
- add better tests with hypothesis
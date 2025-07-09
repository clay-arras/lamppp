`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DLMP_ENABLE_CUDA=ON -DLMP_ENABLE_COVERAGE=OFF -DCMAKE_COLOR_DIAGNOSTICS=ON`
`-DC_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++`

### testing 
- test type promotion AND backwards type promotion
- fix github workflow tests
- add better tests with hypothesis

### nets
- init module: He initialization, Xavier initialization
- lmp::empty (create an empty tensor/variable) so we can use lmp::nets::init::...

- add image classification layers like maxpool, conv2d, and batchnorm
- we need to add a shape(), device(), and type() to Variable (type might fail due to promotion in backward)

- add default value to reduct s.t. it sums over EVERYTHING
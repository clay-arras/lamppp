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


### convolutions: 

<!-- - implement tensor -->
<!-- - implement cpu version     -->
<!-- - manual testing -->
<!-- - implement autograd -->


- implement conv1d2d3d
- implement transpose convolution
- implement depthwise convolution



requirements: 
- need to export a python module: already done.
- need to figure out how to LINK the cpp .so file to the module
- finally need to figure out how to export said module WITH the pybind.
- preferrably uses poetry


ok. we want poetry to STAY as our package manager
we want to replicate the pyproject.toml and CMakeLists.txt inside of scikit_build_example
we want to install that.

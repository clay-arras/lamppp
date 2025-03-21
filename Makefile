CXX = g++-14
CXXFLAGS = -std=c++17 -w -Wfatal-errors
INCLUDES = -I/usr/include/c++/14 \
          -I/usr/include \

ENGINE_SRC = test/test_engine.cpp autograd/engine.cpp autograd/nn.cpp
NN_SRC = test/test_nn.cpp autograd/engine.cpp autograd/nn.cpp
UTIL_SRC = autograd/engine.cpp autograd/nn.cpp autograd/util.cpp
MNIST_SRC = autograd/engine.cpp autograd/nn.cpp autograd/util.cpp autograd/mnist.cpp
TARGET = out 

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

test_engine: $(ENGINE_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ENGINE_SRC) -o $@

test_nn: $(NN_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(NN_SRC) -o $@

util: $(UTIL_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(UTIL_SRC) -o $@

mnist: $(MNIST_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(MNIST_SRC) -o $@

clean:
	rm -f $(TARGET) $(ENGINE_SRC) $(NN_SRC)

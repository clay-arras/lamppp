CXX = g++-14
CXXFLAGS = -std=c++17 -w -Wfatal-errors
INCLUDES = -I/usr/include/c++/14 \
          -I/usr/include/x86_64-linux-gnu/c++/14 \
          -I/usr/include \
          -I/usr/include/linux \
          -I/usr/lib/gcc/x86_64-linux-gnu/14/include/ \
          -I/usr/include/c++/14/tr1/

SRC = test/test_engine.cpp autograd/engine.cpp
TARGET = test_engine

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
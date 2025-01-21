CXX = g++-14
CXXFLAGS = -std=c++17 -w -Wfatal-errors
INCLUDES = -I/usr/include/c++/14 \
          -I/usr/include \

SRC = test/test_engine.cpp autograd/engine.cpp
TARGET = test_engine

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

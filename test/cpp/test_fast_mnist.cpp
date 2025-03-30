#include "autograd/fast_layer.h"
#include "autograd/wrapper_engine.h"

int main() {
  int nin = 108 * 108 * 3;
  FastLayer W1(nin, 1024);
  FastLayer W2(1024, 512);
  FastLayer W3(512, 256);
  FastLayer W4(256, 128);
  FastLayer W5(128, 10);
}
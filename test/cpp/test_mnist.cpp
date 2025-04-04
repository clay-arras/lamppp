#include "autograd/nn/nn.h"

int main() {
  int nin = 108 * 108 * 3;
  Layer w1(nin, 1024);
  Layer w2(1024, 512);
  Layer w3(512, 256);
  Layer w4(256, 128);
  Layer w5(128, 10);
}
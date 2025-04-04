#include "autograd/nn/fast_layer.h"

int main() {
  int nin = 108 * 108 * 3;
  FastLayer w1(nin, 1024);
  FastLayer w2(1024, 512);
  FastLayer w3(512, 256);
  FastLayer w4(256, 128);
  FastLayer w5(128, 10);
}
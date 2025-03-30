#include "autograd/nn.h"

int main() {
  int nin = 108 * 108 * 3;
  Layer W1(nin, 1024);
  Layer W2(1024, 512);
  Layer W3(512, 256);
  Layer W4(256, 128);
  Layer W5(128, 10);
}
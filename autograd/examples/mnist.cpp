#include "mnist.h"
#include "autograd/util/csv_reader.h"
#include <cassert>

int main() {
  auto [data, label] = readCSV("data/mnist_dummy.csv");

}
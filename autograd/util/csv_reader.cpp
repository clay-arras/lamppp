#include "csv_reader.h"
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
readCSV(const std::string& filename) {
  std::vector<std::vector<float>> data;
  std::vector<std::vector<float>> label;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return {data, label};
  }
  std::string line;
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;

    std::getline(ss, cell, ',');
    int label_index = stoi(cell);

    std::vector<float> one_hot(10, 0.0F);
    one_hot[label_index] = 1.0F;
    label.push_back(one_hot);

    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(cell) / 255);
    }

    data.push_back(row);
  }

  file.close();
  return {data, label};
}
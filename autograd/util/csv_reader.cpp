#include "csv_reader.h"

std::pair<std::vector<std::vector<double>>, std::vector<int>> readCSV(
    const std::string& filename) {
  std::vector<std::vector<double>> data;
  std::vector<int> label;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return {data, label};
  }
  std::string line;
  std::getline(file, line);  // skips the first cell

  while (std::getline(file, line)) {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;

    std::getline(ss, cell, ',');  // first cell is the label
    label.push_back(stoi(cell));

    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(cell) / 255);
    }

    data.push_back(row);
  }

  file.close();
  return {data, label};
}
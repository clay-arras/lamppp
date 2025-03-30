#include "csv_reader.h"
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>

/**
 * @brief Reads a CSV file and extracts data and labels.
 *
 * This function opens a specified CSV file, reads its contents, and separates
 * the data into a vector of rows and a vector of labels. The first cell of each
 * row is treated as the label, while the remaining cells are treated as data
 * points. The data points are normalized by dividing by 255.
 *
 * @param filename The name of the CSV file to read.
 * @return A pair containing a vector of data rows and a vector of labels.
 */
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
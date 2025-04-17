#ifndef _UTIL_H_
#define _UTIL_H_

#include <string>
#include <vector>

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
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
readCSV(const std::string& filename);

#endif
#ifndef _UTIL_H_
#define _UTIL_H_

#include <vector>
#include <string>
#include <iostream>
#include <ostream>

std::pair<std::vector<std::vector<double>>, std::vector<int>> readCSV(const std::string& filename);

#endif
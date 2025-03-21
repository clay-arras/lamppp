#ifndef _UTIL_H_
#define _UTIL_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

std::pair<std::vector<std::vector<double>>, std::vector<int>> readCSV(const std::string& filename);

#endif
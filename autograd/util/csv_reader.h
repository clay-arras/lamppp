#ifndef _UTIL_H_
#define _UTIL_H_

#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

std::pair<std::vector<std::vector<double>>, std::vector<int>> readCSV(
    const std::string& filename);

#endif
#pragma once

#ifndef _CSV_READER_H_
#define _CSV_READER_H_

#include <string>
#include <vector>

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
readCSV(const std::string& filename);

#endif  // _CSV_READER_H_
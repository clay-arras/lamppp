#pragma once

#ifndef _BATCH_SAMPLE_H_
#define _BATCH_SAMPLE_H_

#include <algorithm>
#include <cassert>
#include <iterator>
#include <random>
#include <vector>

template <typename X, typename Y>
void sample_batch_sample(const std::vector<X>& data,
                         const std::vector<Y>& labels, std::size_t k,
                         std::vector<X>& out_data, std::vector<Y>& out_labels) {
  assert(data.size() == labels.size());
  std::size_t n = data.size();
  k = std::min(k, n);

  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0U);

  std::vector<std::size_t> pick;
  pick.reserve(k);
  std::sample(idx.begin(), idx.end(), std::back_inserter(pick), k,
              std::mt19937{std::random_device{}()});

  out_data.reserve(k);
  out_labels.reserve(k);
  for (auto i : pick) {
    out_data.push_back(data[i]);
    out_labels.push_back(labels[i]);
  }
}

#endif  // _BATCH_SAMPLE_H_
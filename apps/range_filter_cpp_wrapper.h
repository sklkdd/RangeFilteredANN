#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include "../ParlayANN/algorithms/utils/point_range.h"
#include "../ParlayANN/algorithms/utils/euclidian_point.h"
#include "../src/prefiltering.h"
#include "../src/postfilter_vamana.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

using index_type = int32_t;

// Simplified C++ interface for RangeFilteredANN without pybind11 dependencies
template <typename T, typename Point,
          template <typename, typename, typename> class RangeSpatialIndex = PrefilterIndex,
          typename FilterType = float>
class RangeFilterCppWrapper {
public:
    using pid = std::pair<index_type, float>;
    using PR = PointRange<T, Point>;
    using SubsetRange = SubsetPointRange<T, Point>;
    using SubsetRangePtr = std::unique_ptr<SubsetRange>;
    using SpatialIndex = RangeSpatialIndex<T, Point, SubsetRange>;
    using SpatialIndexPtr = std::unique_ptr<SpatialIndex>;
    using FilterRange = std::pair<FilterType, FilterType>;
    using FilterList = parlay::sequence<FilterType>;

    // Constructor from raw data (no Python)
    RangeFilterCppWrapper(T* data, size_t n, size_t d, FilterType* filters,
                         int32_t cutoff, size_t split_factor, BuildParams build_params) {
        // Sort by filter values
        FilterList filter_values_seq(filters, filters + n);
        
        auto filter_indices_sorted = parlay::tabulate(n, [](index_type i) { return i; });
        parlay::sort_inplace(filter_indices_sorted, [&](auto i, auto j) {
            return filter_values_seq[i] < filter_values_seq[j];
        });

        // Create sorted data
        auto data_sorted = parlay::sequence<T>(n * d);
        auto decoding = parlay::sequence<size_t>(n, 0);

        parlay::parallel_for(0, n, [&](size_t sorted_id) {
            auto original_id = filter_indices_sorted[sorted_id];
            decoding[sorted_id] = original_id;
            for (size_t dim = 0; dim < d; dim++) {
                data_sorted[sorted_id * d + dim] = data[original_id * d + dim];
            }
        });

        auto sorted_filter_values = parlay::sequence<FilterType>(n);
        parlay::parallel_for(0, n, [&](size_t sorted_id) {
            sorted_filter_values[sorted_id] = filter_values_seq[filter_indices_sorted[sorted_id]];
        });

        // Create PointRange
        auto sorted_point_range = std::make_shared<PR>(data_sorted.data(), n, d, d);

        // Initialize the tree structure
        _sorted_index_to_original_point_id = decoding;
        _cutoff = cutoff;
        _filter_values = sorted_filter_values;
        _points = sorted_point_range;
        _split_factor = split_factor;

        // Build spatial indices
        _spatial_indices.push_back(std::vector<SpatialIndexPtr>(1));
        _spatial_indices.at(0).at(0) = create_index(_filter_values, 0, _filter_values.size(), _points.get(), build_params);
        _bucket_offsets.push_back({0, _filter_values.size()});

        while (_bucket_offsets.back().at(1) > cutoff) {
            auto last_num_buckets = _spatial_indices.back().size();
            _bucket_offsets.push_back(std::vector<size_t>(last_num_buckets * _split_factor + 1));
            _bucket_offsets.back().back() = _filter_values.size();
            _spatial_indices.push_back(std::vector<SpatialIndexPtr>(last_num_buckets * _split_factor));

            parlay::parallel_for(0, last_num_buckets, [&](auto last_bucket_id) {
                auto last_start = _bucket_offsets.at(_bucket_offsets.size() - 2).at(last_bucket_id);
                auto last_end = _bucket_offsets.at(_bucket_offsets.size() - 2).at(last_bucket_id + 1);
                auto last_size = last_end - last_start;

                auto large_bucket_size = (last_size + _split_factor - 1) / _split_factor;
                auto small_bucket_size = large_bucket_size - 1;
                auto num_larger_buckets = last_size - small_bucket_size * _split_factor;

                parlay::parallel_for(0, num_larger_buckets, [&](auto i) {
                    auto start = last_start + i * large_bucket_size;
                    auto end = start + large_bucket_size;
                    _bucket_offsets.back().at(last_bucket_id * _split_factor + i) = start;
                    _spatial_indices.back().at(last_bucket_id * _split_factor + i) =
                        create_index(_filter_values, start, end, _points.get(), build_params);
                });

                parlay::parallel_for(num_larger_buckets, _split_factor, [&](auto i) {
                    auto start = last_start + num_larger_buckets * large_bucket_size +
                               (i - num_larger_buckets) * small_bucket_size;
                    auto end = start + small_bucket_size;
                    _bucket_offsets.back().at(last_bucket_id * _split_factor + i) = start;
                    _spatial_indices.back().at(last_bucket_id * _split_factor + i) =
                        create_index(_filter_values, start, end, _points.get(), build_params);
                });
            });
        }
    }

    // Search function returning C++ vectors instead of numpy arrays
    std::vector<pid> optimized_postfiltering_search(const Point &query, const FilterRange &range, QueryParams query_params) {
        // Based on the original implementation in range_filter_tree.h
        if (check_empty(range)) {
            return std::vector<pid>();
        }

        size_t start = first_greater_than_or_equal_to(range.first, _filter_values);
        size_t end = first_greater_than_or_equal_to(range.second, _filter_values);
        
        if (end < _filter_values.size() && _filter_values[end] == range.second) {
            end++;
        }

        return optimized_postfilter_search_in_range(query, start, end, query_params);
    }

private:
    std::vector<std::vector<size_t>> _bucket_offsets;
    std::vector<std::vector<SpatialIndexPtr>> _spatial_indices;
    parlay::sequence<size_t> _sorted_index_to_original_point_id;
    FilterList _filter_values;
    int32_t _cutoff;
    std::shared_ptr<PR> _points;
    size_t _split_factor;

    static SpatialIndexPtr create_index(FilterList &filter_values, size_t start,
                                       size_t end, PR *points, BuildParams build_params) {
        auto filter_length = end - start;
        parlay::sequence<int32_t> subset_of_indices = parlay::tabulate<int32_t>(
            filter_length, [&](auto i) { return i + start; });
        SubsetRangePtr subset_points = points->make_subset(subset_of_indices);
        FilterList subset_of_filter_values =
            FilterList(filter_values.begin() + start, filter_values.begin() + end);

        return std::make_unique<SpatialIndex>(
            std::move(subset_points), subset_of_filter_values, build_params);
    }

    bool check_empty(const FilterRange &range) {
        return range.second < _filter_values.front() || range.first > _filter_values.back();
    }

    size_t first_greater_than_or_equal_to(const FilterType &filter_value, const FilterList &filter_values) {
        if (filter_values[0] >= filter_value) {
            return 0;
        }
        size_t start = 0;
        size_t end = filter_values.size();
        while (start + 1 < end) {
            size_t mid = (start + end) / 2;
            if (filter_values[mid] >= filter_value) {
                end = mid;
            } else {
                start = mid;
            }
        }
        return end;
    }

    std::vector<pid> optimized_postfilter_search_in_range(const Point &query, size_t start, size_t end, QueryParams query_params) {
        // Find appropriate level and buckets
        size_t level = 0;
        size_t start_bucket = 0;
        size_t end_bucket = 1;

        for (size_t l = 0; l < _bucket_offsets.size(); l++) {
            auto num_buckets = _bucket_offsets[l].size() - 1;
            size_t s_bucket = 0, e_bucket = num_buckets;

            for (size_t b = 0; b < num_buckets; b++) {
                if (_bucket_offsets[l][b] <= start && start < _bucket_offsets[l][b + 1]) {
                    s_bucket = b;
                }
                if (_bucket_offsets[l][b] < end && end <= _bucket_offsets[l][b + 1]) {
                    e_bucket = b + 1;
                    break;
                }
            }

            level = l;
            start_bucket = s_bucket;
            end_bucket = e_bucket;

            if (end_bucket - start_bucket == 1) {
                break;
            }
        }

        // Perform search in the buckets
        auto results = parlay::sequence<parlay::sequence<pid>>(end_bucket - start_bucket);
        
        parlay::parallel_for(start_bucket, end_bucket, [&](size_t bucket) {
            results[bucket - start_bucket] = _spatial_indices[level][bucket]->beam_search(query, query_params);
        });

        // Merge results
        auto merged = parlay::flatten(results);
        parlay::sort_inplace(merged, [](const pid &a, const pid &b) { return a.second < b.second; });

        // Filter to k results
        size_t k = std::min(static_cast<size_t>(query_params.k), merged.size());
        std::vector<pid> final_results(merged.begin(), merged.begin() + k);

        // Map back to original IDs
        for (auto &p : final_results) {
            p.first = _sorted_index_to_original_point_id[p.first];
        }

        return final_results;
    }
};

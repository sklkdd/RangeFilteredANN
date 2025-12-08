#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include "../ParlayANN/algorithms/utils/point_range.h"
#include "../ParlayANN/algorithms/utils/euclidian_point.h"
#include "../src/range_filter_tree.h"
#include "../src/postfilter_vamana.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

using index_type = int32_t;

// C++ wrapper for RangeFilteredANN's B-Window Search Tree
// This provides a non-Python interface to the actual RangeFilterTreeIndex algorithm
template <typename T, typename Point, typename FilterType = float>
class RangeFilterCppWrapper {
public:
    using pid = std::pair<index_type, float>;
    using FilterRange = std::pair<FilterType, FilterType>;
    using PR = PointRange<T, Point>;
    using SubsetRange = SubsetPointRange<T, Point>;
    using Index = RangeFilterTreeIndex<T, Point, PostfilterVamanaIndex, FilterType>;

    // Constructor that builds the B-WST from raw data
    RangeFilterCppWrapper(T* data, size_t n, size_t d, FilterType* filters,
                         int32_t cutoff, size_t split_factor, BuildParams build_params) {
        _n = n;
        _d = d;
        
        // Sort by filter values (required by RangeFilterTreeIndex)
        parlay::sequence<FilterType> filter_seq(filters, filters + n);
        auto filter_indices_sorted = parlay::tabulate(n, [](index_type i) { return i; });
        parlay::sort_inplace(filter_indices_sorted, [&](auto i, auto j) {
            return filter_seq[i] < filter_seq[j];
        });

        // Create sorted data
        auto data_sorted = parlay::sequence<T>(n * d);
        _decoding = parlay::sequence<size_t>(n, 0);

        parlay::parallel_for(0, n, [&](size_t sorted_id) {
            auto original_id = filter_indices_sorted[sorted_id];
            _decoding[sorted_id] = original_id;
            for (size_t dim = 0; dim < d; dim++) {
                data_sorted[sorted_id * d + dim] = data[original_id * d + dim];
            }
        });

        auto sorted_filter_values = parlay::sequence<FilterType>(n);
        parlay::parallel_for(0, n, [&](size_t sorted_id) {
            sorted_filter_values[sorted_id] = filter_seq[filter_indices_sorted[sorted_id]];
        });

        // Create PointRange with sorted data
        auto sorted_point_range = std::make_shared<PR>(data_sorted.data(), n, d, d);
        
        // Build the actual RangeFilterTreeIndex (B-WST)
        // This creates the recursive tree structure with multiple indices
        _index = std::make_unique<Index>(
            sorted_point_range,
            sorted_filter_values,
            _decoding,
            cutoff,
            split_factor,
            build_params
        );
    }

    // Optimized postfiltering search - the key RangeFilteredANN query method
    std::vector<pid> optimized_postfiltering_search(const Point &query, const FilterRange &range, QueryParams query_params) {
        // Call the actual RangeFilterTreeIndex's optimized postfiltering search
        // This finds the smallest partition containing the range and does smart postfiltering
        auto results = _index->optimized_postfiltering_search(query, range, query_params);
        
        // Results are already in original ID space (handled by RangeFilterTreeIndex)
        std::vector<pid> cpp_results;
        cpp_results.reserve(results.size());
        for (const auto& r : results) {
            cpp_results.push_back(r);
        }
        return cpp_results;
    }

private:
    size_t _n, _d;
    parlay::sequence<size_t> _decoding;
    std::unique_ptr<Index> _index;
};

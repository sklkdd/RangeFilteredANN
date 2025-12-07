#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <omp.h>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

#include "../ParlayANN/algorithms/utils/euclidian_point.h"
#include "../ParlayANN/algorithms/utils/point_range.h"
#include "../src/range_filter_tree.h"
#include "../src/postfilter_vamana.h"

// Define global for thread counter
std::atomic<int> peak_threads(1);

int main(int argc, char** argv) {
    if (argc != 13) {
        std::cerr << "Usage: " << argv[0] << " <data.bin> <filters.csv> <queries.bin> <query_filters.csv> <gt.ivecs> <R> <L> <alpha> <cutoff> <split_factor> <K> <L_search>" << std::endl;
        std::cerr << "Note: Since RangeFilteredANN doesn't support index serialization, we rebuild the index here" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    std::string filters_path = argv[2];
    std::string queries_path = argv[3];
    std::string query_filters_path = argv[4];
    std::string gt_path = argv[5];
    
    uint32_t R = std::stoi(argv[6]);
    uint32_t L = std::stoi(argv[7]);
    float alpha = std::stof(argv[8]);
    int32_t cutoff = std::stoi(argv[9]);
    size_t split_factor = std::stoi(argv[10]);
    uint32_t K = std::stoi(argv[11]);
    uint32_t L_search = std::stoi(argv[12]);

    using T = float;
    using Point = Euclidian_Point<T>;
    using Index = RangeFilterTreeIndex<T, Point, PostfilterVamanaIndex>;

    // Restrict to 1 thread for query execution
    omp_set_num_threads(1);

    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    // Load data and build index (not timed - should have been done in build phase)
    std::cout << "Loading data and rebuilding index..." << std::endl;
    auto points = std::make_shared<PointRange<T, Point>>(data_path.data());
    auto filters = read_one_float_per_line(filters_path);
    parlay::sequence<float> filter_seq(filters.begin(), filters.end());
    
    BuildParams bp(R, L, alpha);
    Index index(points, filter_seq, cutoff, split_factor, bp);

    // Load queries
    std::cout << "Loading queries..." << std::endl;
    auto queries = std::make_shared<PointRange<T, Point>>(queries_path.data());
    auto query_filters = read_two_floats_per_line(query_filters_path);

    if (queries->size() != query_filters.size()) {
        std::cerr << "Error: Queries and query filters size mismatch." << std::endl;
        return 1;
    }

    // Load ground truth
    auto gt = read_ivecs(gt_path);
    if (gt.size() != queries->size()) {
        std::cerr << "Error: Ground truth size (" << gt.size() << ") doesn't match queries (" << queries->size() << ")" << std::endl;
        return 1;
    }

    // Store query results for recall calculation
    std::vector<std::vector<unsigned int>> query_results(queries->size());

    // Start timing
    auto start_search = std::chrono::high_resolution_clock::now();

    QueryParams qp;
    qp.k = K;
    qp.beamSize = L_search;
    qp.cut = 1.35;
    qp.limit = 10000000;
    qp.degree_limit = 10000;

    // Execute queries (single-threaded)
    for (size_t i = 0; i < queries->size(); ++i) {
        Point q = (*queries)[i];
        std::pair<float, float> filter = query_filters[i];
        
        // Use optimized_postfilter method
        auto results = index.optimized_postfiltering_search(q, filter, qp);
        
        // Store results
        query_results[i].reserve(K);
        for (auto& res : results) {
            if (query_results[i].size() >= K) break;
            query_results[i].push_back(index._sorted_index_to_original_point_id.at(res.first));
        }
    }

    // Stop timing
    auto end_search = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_search - start_search;

    // Stop monitoring
    done = true;
    monitor.join();

    // Calculate recall (not timed)
    int total_correct = 0;
    for (size_t i = 0; i < queries->size(); ++i) {
        std::set<unsigned int> result_set(query_results[i].begin(), query_results[i].end());
        
        for (size_t j = 0; j < K && j < gt[i].size(); ++j) {
            if (result_set.count(gt[i][j])) {
                total_correct++;
            }
        }
    }

    float recall = (float)total_correct / (queries->size() * K);
    float qps = queries->size() / elapsed.count();

    // Print statistics
    std::cout << "Query execution completed." << std::endl;
    std::cout << "Query time (s): " << elapsed.count() << std::endl;
    std::cout << "Peak thread count: " << peak_threads.load() << std::endl;
    std::cout << "QPS: " << qps << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    
    // Print memory footprint
    peak_memory_footprint();

    return 0;
}

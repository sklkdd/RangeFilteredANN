#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <omp.h>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"
#include "range_filter_cpp_wrapper.h"

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
    using Index = RangeFilterCppWrapper<T, Point>;

    // Restrict to 1 thread for query execution
    omp_set_num_threads(1);

    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    // Load data and build index (not timed - should have been done in build phase)
    std::cout << "Loading data and rebuilding index..." << std::endl;
    FILE* fp = fopen(data_path.c_str(), "rb");
    uint32_t n, d;
    fread(&n, sizeof(uint32_t), 1, fp);
    fread(&d, sizeof(uint32_t), 1, fp);
    std::vector<T> data_vec(n * d);
    fread(data_vec.data(), sizeof(T), n * d, fp);
    fclose(fp);
    
    auto filters = read_one_float_per_line(filters_path);
    BuildParams bp(R, L, alpha);
    Index index(data_vec.data(), n, d, filters.data(), cutoff, split_factor, bp);

    // Load queries
    std::cout << "Loading queries..." << std::endl;
    fp = fopen(queries_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open " << queries_path << std::endl;
        return 1;
    }
    uint32_t num_queries, query_dim;
    fread(&num_queries, sizeof(uint32_t), 1, fp);
    fread(&query_dim, sizeof(uint32_t), 1, fp);
    if (query_dim != d) {
        std::cerr << "Error: Query dimension (" << query_dim << ") doesn't match data dimension (" << d << ")" << std::endl;
        fclose(fp);
        return 1;
    }
    std::vector<T> query_vec(num_queries * query_dim);
    fread(query_vec.data(), sizeof(T), num_queries * query_dim, fp);
    fclose(fp);
    
    auto query_filters = read_two_floats_per_line(query_filters_path);

    if (num_queries != query_filters.size()) {
        std::cerr << "Error: Queries and query filters size mismatch." << std::endl;
        return 1;
    }

    // Load ground truth
    auto gt = read_ivecs(gt_path);
    if (gt.size() != num_queries) {
        std::cerr << "Error: Ground truth size (" << gt.size() << ") doesn't match queries (" << num_queries << ")" << std::endl;
        return 1;
    }

    // Store query results for recall calculation
    std::vector<std::vector<unsigned int>> query_results(num_queries);

    // Start timing
    auto start_search = std::chrono::high_resolution_clock::now();

    QueryParams qp;
    qp.k = K;
    qp.beamSize = L_search;
    qp.cut = 1.35;
    qp.limit = 10000000;
    qp.degree_limit = 10000;

    // Execute queries (single-threaded)
    for (size_t i = 0; i < num_queries; ++i) {
        Point q(query_vec.data() + i * query_dim, query_dim, query_dim, i);
        std::pair<float, float> filter = query_filters[i];
        
        // Use optimized_postfilter method
        auto results = index.optimized_postfiltering_search(q, filter, qp);
        
        // Store results (already mapped to original IDs by wrapper)
        query_results[i].reserve(K);
        for (auto& res : results) {
            if (query_results[i].size() >= K) break;
            query_results[i].push_back(res.first);
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
    for (size_t i = 0; i < num_queries; ++i) {
        std::set<unsigned int> result_set(query_results[i].begin(), query_results[i].end());
        
        for (size_t j = 0; j < K && j < gt[i].size(); ++j) {
            if (result_set.count(gt[i][j])) {
                total_correct++;
            }
        }
    }

    double recall = static_cast<double>(total_correct) / (num_queries * K);
    double qps = num_queries / elapsed.count();

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

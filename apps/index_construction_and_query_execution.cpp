// index_construction_and_query_execution.cpp - RangeFilteredANN combined wrapper for FANNS benchmarking
// This wrapper builds a RangeFilteredANN index once and queries it multiple times
// with different L_search (beamSize) values

#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <memory>
#include <set>
#include <algorithm>
#include <omp.h>

// Define global for thread counter BEFORE including helpers that use it
std::atomic<int> peak_threads(1);

#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"
#include "range_filter_cpp_wrapper.h"

// Parse comma-separated list of integers (e.g., "4,8,16,32,64")
std::vector<int> parse_int_list(const std::string& input) {
    std::string cleaned = input;
    // Remove brackets if present
    cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(),
                  [](char c) { return c == '[' || c == ']'; }),
                  cleaned.end());

    std::vector<int> result;
    std::stringstream ss(cleaned);
    std::string token;

    while (std::getline(ss, token, ',')) {
        result.push_back(std::stoi(token));
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc != 14) {
        std::cerr << "Usage: " << argv[0] << " <data.bin> <filters.csv> <queries.bin> <query_filters.csv> "
                  << "<groundtruth.ivecs> <R> <L> <alpha> <cutoff> <split_factor> <k> <L_search_list> <threads>" << std::endl;
        std::cerr << "\n";
        std::cerr << "Arguments:\n";
        std::cerr << "  data.bin          - Database vectors in .bin format\n";
        std::cerr << "  filters.csv       - Database filter values (one per line)\n";
        std::cerr << "  queries.bin       - Query vectors in .bin format\n";
        std::cerr << "  query_filters.csv - Query ranges (min-max per line)\n";
        std::cerr << "  groundtruth.ivecs - Groundtruth in .ivecs format\n";
        std::cerr << "  R                 - Graph degree parameter\n";
        std::cerr << "  L                 - Build beam size\n";
        std::cerr << "  alpha             - Alpha parameter\n";
        std::cerr << "  cutoff            - Cutoff parameter\n";
        std::cerr << "  split_factor      - Split factor parameter\n";
        std::cerr << "  k                 - Number of neighbors to return\n";
        std::cerr << "  L_search_list     - Comma-separated list of L_search values (e.g., 10,20,50,100)\n";
        std::cerr << "  threads           - Number of threads for index construction (0 = auto)\n";
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
    std::vector<int> L_search_list = parse_int_list(argv[12]);
    int threads = std::stoi(argv[13]);

    using T = float;
    using Point = Euclidian_Point<T>;
    using Index = RangeFilterCppWrapper<T, Point>;

    std::cout << "=== RangeFilteredANN Index Construction and Query Execution ===" << std::endl;
    std::cout << "Data: " << data_path << std::endl;
    std::cout << "Filters: " << filters_path << std::endl;
    std::cout << "Queries: " << queries_path << std::endl;
    std::cout << "Query filters: " << query_filters_path << std::endl;
    std::cout << "Groundtruth: " << gt_path << std::endl;
    std::cout << "Parameters: R=" << R << ", L=" << L << ", alpha=" << alpha 
              << ", cutoff=" << cutoff << ", split_factor=" << split_factor << ", k=" << K << std::endl;
    std::cout << "L_search values: ";
    for (int ls : L_search_list) std::cout << ls << " ";
    std::cout << std::endl;

    // ========== DATA LOADING (NOT TIMED) ==========
    std::cout << "\nLoading data..." << std::endl;
    
    // Load database vectors
    FILE* fp = fopen(data_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open " << data_path << std::endl;
        return 1;
    }
    uint32_t n, d;
    fread(&n, sizeof(uint32_t), 1, fp);
    fread(&d, sizeof(uint32_t), 1, fp);
    std::vector<T> data_vec(n * d);
    fread(data_vec.data(), sizeof(T), n * d, fp);
    fclose(fp);
    
    auto filters = read_one_float_per_line(filters_path);

    if (n != filters.size()) {
        std::cerr << "Error: Points (" << n << ") and filters (" << filters.size() << ") size mismatch." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << n << " database vectors (dim=" << d << ") with " << filters.size() << " filter values" << std::endl;

    // Load query vectors
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
    std::cout << "Loaded " << num_queries << " query vectors" << std::endl;
    
    // Load query filters
    auto query_filters = read_two_floats_per_line(query_filters_path);
    if (num_queries != query_filters.size()) {
        std::cerr << "Error: Queries (" << num_queries << ") and query filters (" << query_filters.size() << ") size mismatch." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << query_filters.size() << " query filter ranges" << std::endl;

    // Load ground truth
    auto gt = read_ivecs(gt_path);
    if (gt.size() != num_queries) {
        std::cerr << "Error: Ground truth size (" << gt.size() << ") doesn't match queries (" << num_queries << ")" << std::endl;
        return 1;
    }
    // Truncate to k
    for (auto& g : gt) {
        if (g.size() > K) {
            g.resize(K);
        }
    }
    std::cout << "Loaded " << gt.size() << " groundtruth entries" << std::endl;

    // ========== INDEX CONSTRUCTION (TIMED) ==========
    std::cout << "\n--- Starting index construction (TIMED) ---" << std::endl;

    // Thread count is managed by Parlay internally
    // We just monitor it
    std::atomic<bool> done_build(false);
    std::thread monitor_build(monitor_thread_count, std::ref(done_build));

    auto start_build = std::chrono::high_resolution_clock::now();

    BuildParams bp(R, L, alpha);
    Index index(data_vec.data(), n, d, filters.data(), cutoff, split_factor, bp);

    auto end_build = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_time = end_build - start_build;

    done_build = true;
    monitor_build.join();
    int build_threads = peak_threads.load();

    std::cout << "--- Index construction complete ---" << std::endl;

    // ========== QUERY EXECUTION (TIMED per L_search value) ==========
    std::cout << "\n--- Starting query execution ---" << std::endl;

    // Force single-threaded query execution
    omp_set_num_threads(1);
    
    // Start thread monitoring for query phase
    std::atomic<bool> done_query(false);
    peak_threads = 1;  // Reset
    std::thread monitor_query(monitor_thread_count, std::ref(done_query));

    // Store results for each L_search value
    std::vector<double> recall_list;
    std::vector<double> qps_list;

    QueryParams qp;
    qp.k = K;
    qp.cut = 1.35;
    qp.limit = 10000000;
    qp.degree_limit = 10000;

    for (int L_search : L_search_list) {
        qp.beamSize = L_search;
        
        std::vector<std::vector<unsigned int>> query_results(num_queries);

        auto start_query = std::chrono::high_resolution_clock::now();

        // Execute queries (single-threaded)
        for (size_t i = 0; i < num_queries; ++i) {
            Point q(query_vec.data() + i * query_dim, query_dim, query_dim, i);
            std::pair<float, float> filter = query_filters[i];
            
            auto results = index.optimized_postfiltering_search(q, filter, qp);
            
            query_results[i].reserve(K);
            for (auto& res : results) {
                if (query_results[i].size() >= K) break;
                query_results[i].push_back(res.first);
            }
        }

        auto end_query = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> query_time = end_query - start_query;

        double qps = num_queries / query_time.count();

        // Calculate recall
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

        recall_list.push_back(recall);
        qps_list.push_back(qps);
    }

    // Stop thread monitoring for query phase
    done_query = true;
    monitor_query.join();
    int query_threads = peak_threads.load();

    std::cout << "--- Query execution complete ---\n" << std::endl;

    // ========== OUTPUT RESULTS ==========
    // Output in same format as SeRF for compatibility with parsing
    peak_memory_footprint();
    printf("Maximum number of threads during index construction: %d\n", build_threads - 1);
    printf("Maximum number of threads during query execution: %d\n", query_threads - 1);
    printf("Index construction time: %.3f s\n", build_time.count());
    
    for (size_t i = 0; i < L_search_list.size(); i++) {
        printf("L_search: %d QPS: %.3f Recall: %.5f\n", L_search_list[i], qps_list[i], recall_list[i]);
    }

    return 0;
}

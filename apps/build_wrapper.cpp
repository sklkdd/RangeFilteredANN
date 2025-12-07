#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <memory>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

#include "../ParlayANN/algorithms/utils/euclidian_point.h"
#include "../ParlayANN/algorithms/utils/point_range.h"
#include "../src/range_filter_tree.h"
#include "../src/postfilter_vamana.h"

// Define global for thread counter
std::atomic<int> peak_threads(1);

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <data.bin> <filters.csv> <index_path> <R> <L> <alpha> <cutoff> <split_factor>" << std::endl;
        std::cerr << "Note: index_path is unused as RangeFilteredANN doesn't support serialization" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    std::string filters_path = argv[2];
    // std::string index_path = argv[3];  // Unused
    
    uint32_t R = std::stoi(argv[4]);
    uint32_t L = std::stoi(argv[5]);
    float alpha = std::stof(argv[6]);
    int32_t cutoff = std::stoi(argv[7]);
    size_t split_factor = std::stoi(argv[8]);

    using T = float;
    using Point = Euclidian_Point<T>;
    using Index = RangeFilterTreeIndex<T, Point, PostfilterVamanaIndex>;

    // Use all available threads for building
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "Using " << nthreads << " threads for index construction" << std::endl;

    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    // Load data BEFORE starting timer
    std::cout << "Loading data..." << std::endl;
    auto points = std::make_shared<PointRange<T, Point>>(data_path.data());
    auto filters = read_one_float_per_line(filters_path);

    if (points->size() != filters.size()) {
        std::cerr << "Error: Points (" << points->size() << ") and filters (" << filters.size() << ") size mismatch." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << points->size() << " points with " << filters.size() << " filter values" << std::endl;

    // Convert vector to parlay::sequence
    parlay::sequence<float> filter_seq(filters.begin(), filters.end());

    // Start timing
    auto start_build = std::chrono::high_resolution_clock::now();

    BuildParams bp(R, L, alpha);
    Index index(points, filter_seq, cutoff, split_factor, bp);

    // Stop timing
    auto end_build = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_time = end_build - start_build;

    // Stop monitoring
    done = true;
    monitor.join();

    // Note: The index cannot be easily serialized, so we don't save it
    // This is a limitation of the current RangeFilteredANN implementation
    // The Python wrapper will need to handle this by rebuilding for each query set

    // Print statistics
    std::cout << "Index construction completed." << std::endl;
    std::cout << "Build time (s): " << build_time.count() << std::endl;
    std::cout << "Peak thread count: " << peak_threads.load() << std::endl;
    
    // Print memory footprint
    peak_memory_footprint();

    return 0;
}

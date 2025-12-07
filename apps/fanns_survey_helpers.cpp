#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include <utility>

#include "global_thread_counter.h"

std::vector<std::vector<int>> read_ivecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for reading: " << filename << "\n";
        return {};
    }
    std::vector<std::vector<int>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        std::vector<int> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(int))) break;
        dataset.push_back(std::move(vec));
    }
    file.close();
    return dataset;
}

std::vector<std::pair<float, float>> read_two_floats_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<std::pair<float, float>> result;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        size_t dash_pos = line.find('-');
        if (dash_pos == std::string::npos) {
            throw std::runtime_error("Invalid format on line " + std::to_string(line_number) + ": expected 'min-max'");
        }
        
        std::string min_str = line.substr(0, dash_pos);
        std::string max_str = line.substr(dash_pos + 1);
        
        try {
            float min_val = std::stof(min_str);
            float max_val = std::stof(max_str);
            result.push_back({min_val, max_val});
        } catch (...) {
            throw std::runtime_error("Invalid number on line " + std::to_string(line_number));
        }
    }
    return result;
}

std::vector<float> read_one_float_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<float> result;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::stringstream ss(line);
        float value;
        if (!(ss >> value)) {
            throw std::runtime_error("Non-float or empty line at line " + std::to_string(line_number));
        }
        std::string extra;
        if (ss >> extra) {
            throw std::runtime_error("More than one value on line " + std::to_string(line_number));
        }
        result.push_back(value);
    }
    return result;
}

void peak_memory_footprint() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.find("VmPeak:") != std::string::npos) {
            std::cout << line << std::endl;
        }
        if (line.find("VmHWM:") != std::string::npos) {
            std::cout << line << std::endl;
        }
    }
}

void monitor_thread_count(std::atomic<bool>& done) {
    extern std::atomic<int> peak_threads;
    while (!done) {
        std::ifstream status_file("/proc/self/status");
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.find("Threads:") != std::string::npos) {
                int current_threads = std::stoi(line.substr(line.find_last_of('\t') + 1));
                int expected = peak_threads.load();
                while (current_threads > expected && !peak_threads.compare_exchange_weak(expected, current_threads)) {
                    // Loop until we successfully update or current is no longer greater
                }
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

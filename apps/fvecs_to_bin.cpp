#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.fvecs> <output.bin>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    // Read fvecs file
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        std::cerr << "Error opening input file: " << input_path << std::endl;
        return 1;
    }

    // Read first vector to get dimension
    uint32_t d;
    input.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    input.seekg(0, std::ios::beg);

    // Count vectors
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    
    size_t vec_size = sizeof(uint32_t) + d * sizeof(float);
    uint32_t n = file_size / vec_size;

    std::cout << "Reading " << n << " vectors of dimension " << d << std::endl;

    // Write binary file: [n][d][data...]
    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        std::cerr << "Error opening output file: " << output_path << std::endl;
        return 1;
    }

    output.write(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    output.write(reinterpret_cast<char*>(&d), sizeof(uint32_t));

    // Read and write vectors
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t dim_check;
        input.read(reinterpret_cast<char*>(&dim_check), sizeof(uint32_t));
        if (dim_check != d) {
            std::cerr << "Dimension mismatch at vector " << i << std::endl;
            return 1;
        }
        
        std::vector<float> vec(d);
        input.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float));
        output.write(reinterpret_cast<char*>(vec.data()), d * sizeof(float));
    }

    std::cout << "Conversion complete: " << output_path << std::endl;
    return 0;
}

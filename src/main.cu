/*
MIT License

Copyright (c) 2024 Bupt CIAGroup

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include "specseq.h"
#include <filesystem>

enum class IOFormat {
    Binary,
    Ascii
};
std::string extractFileNameWithoutExtension(const std::string& filePath) {
    size_t lastSlashPos = filePath.find_last_of("/\\");
    std::string fileName = (lastSlashPos == std::string::npos) ? filePath : filePath.substr(lastSlashPos + 1);

    size_t lastDotPos = fileName.find_last_of('.');
    if (lastDotPos == std::string::npos) {
        return fileName; // No extension found
    }
    return fileName.substr(0, lastDotPos);
}


void printGpuInfo() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CheckCudaError(err, "Failed to get device count");

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return;
    }

    std::cout << "+" << std::string(50, '-') << "+" << std::endl;
    std::cout << "| " << std::left << std::setw(30) << "Property" << "Value" << "              |" << std::endl;
    std::cout << "+" << std::string(50, '-') << "+" << std::endl;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, device);
        CheckCudaError(err, "Failed to get device properties");

        std::cout << "| Device " << device << ": " << deviceProp.name << std::string(48 - 10 - std::strlen(deviceProp.name), ' ') << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(30) << "  Compute capability" << deviceProp.major << "." << deviceProp.minor << std::string(17 - std::to_string(deviceProp.major).length() - std::to_string(deviceProp.minor).length(), ' ') << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(30) << "  Total global memory" << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::string(15 - std::to_string(deviceProp.totalGlobalMem / (1024 * 1024)).length(), ' ') << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(30) << "  Shared memory per block" << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::string(15 - std::to_string(deviceProp.sharedMemPerBlock / 1024).length(), ' ') << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(30) << "  Registers per block" << deviceProp.regsPerBlock << std::string(18 - std::to_string(deviceProp.regsPerBlock).length(), ' ') << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(30) << "  Warp size" << deviceProp.warpSize << std::string(18 - std::to_string(deviceProp.warpSize).length(), ' ') << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(30) << "  Memory clock rate" << deviceProp.memoryClockRate / 1000 << " MHz" << std::string(14 - std::to_string(deviceProp.memoryClockRate / 1000).length(), ' ') << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(30) << "  Memory bus width" << deviceProp.memoryBusWidth << " bits" << std::string(13 - std::to_string(deviceProp.memoryBusWidth).length(), ' ') << " |" << std::endl;
        std::cout << "| " << std::left << std::setw(30) << "  Number of multiprocessors" << deviceProp.multiProcessorCount << std::string(18 - std::to_string(deviceProp.multiProcessorCount).length(), ' ') << " |" << std::endl;
        std::cout << "+" << std::string(50, '-') << "+" << std::endl;
    }

    cudaDeviceProp prop;
    for (int i = 0; i < deviceCount; ++i) {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    }
}

bool processFiles(const std::string& inputFile, IOFormat inputFormat,
    std::vector<std::vector<int32_t>>& hv_csc, std::vector<std::vector<int32_t>>& hv_neighbor, std::vector<int32_t>& hv_dim) {
    printGpuInfo();
    if (inputFormat == IOFormat::Ascii) {
        std::ifstream input_stream(inputFile.c_str());
        if (input_stream.fail()) {
            std::cerr << "Error: Failed to read from input stream." << std::endl;
            return false;
        }
        std::string cur_line;
        int32_t cur_col = -1;
        std::vector<int32_t> temp_col;
        hv_csc.clear();hv_csc.push_back(std::vector<int32_t>());
        hv_dim.clear();hv_dim.push_back(0);
        while (getline(input_stream, cur_line)) {
            cur_line.erase(cur_line.find_last_not_of(" \t\n\r\f\v") + 1);
            if (!cur_line.empty() && cur_line[0] != '#') {
                cur_col++;
                std::stringstream ss(cur_line);
                int32_t temp_dim;
                ss >> temp_dim;
                int32_t cdim = temp_dim;
                int32_t temp_int64_t;
                temp_col.clear();
                while (ss >> temp_int64_t) {
                    temp_col.push_back(static_cast<int32_t>(temp_int64_t + 1));
                    
                }
                std::sort(temp_col.begin(), temp_col.end());
                temp_col.erase(std::unique(temp_col.begin(), temp_col.end()), temp_col.end());
                hv_csc.push_back(temp_col);
                hv_dim.push_back(cdim);
            }
            
        }
        input_stream.close();
    }
    else if (inputFormat == IOFormat::Binary) {
        std::ifstream input_stream(inputFile.c_str(), std::ios::binary);
        if (input_stream.fail()) {
            std::cerr << "Error: Failed to read from input stream." << std::endl;
            return false;
        }
        hv_csc.clear();hv_csc.push_back(std::vector<int32_t>());
        hv_dim.clear();hv_dim.push_back(0);
        int64_t num_cols;
        input_stream.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

        for (int64_t i = 0; i < num_cols; ++i) {
            int64_t temp_dim;
            input_stream.read(reinterpret_cast<char*>(&temp_dim), sizeof(temp_dim));
            int32_t cdim = temp_dim;

            int64_t col_size;
            input_stream.read(reinterpret_cast<char*>(&col_size), sizeof(col_size));

            std::vector<int64_t> temp_col(col_size);
            input_stream.read(reinterpret_cast<char*>(temp_col.data()), col_size * sizeof(int64_t));

            std::sort(temp_col.begin(), temp_col.end());
            temp_col.erase(std::unique(temp_col.begin(), temp_col.end()), temp_col.end());

            std::vector<int32_t> temp_col_int32_t;
            temp_col_int32_t.reserve(temp_col.size());
            for (const auto& value : temp_col) {
                temp_col_int32_t.push_back(static_cast<int32_t>(value));
            }

            hv_csc.push_back(temp_col_int32_t);
            hv_dim.push_back(cdim);
        }
        input_stream.close();
    }

    printf("-----------------------------------------------Reading  completed-----------------------------------------------\n");

    return true;
}

void printHelp() {
    std::cout << "Usage: program <input_file> <output_file> <input_format> <output_format> <model> <ss_block_size> <block_size_growth_rate> <level1_capacity> <level2_capacity> [<level3_capacity>]\n";
    std::cout << "input_format: ascii or binary\n";
    std::cout << "output_format: ascii or binary\n";
    std::cout << "dualize: \n";
    std::cout << "  0 - No dualize\n";
    std::cout << "  1 - Enable dualize\n";
    std::cout << "model: \n";
    std::cout << "  0 - No optimizations\n";
    std::cout << "  1 - Enable high-dimensional clearing theorem\n";
    std::cout << "  2 - Enable edge collapsing\n";
    std::cout << "  3 - Enable both high-dimension guided clearing theorem and edge collapsing\n";
    std::cout << "max_iteration[Only effective when model equals 1 or 3.]: The strength of the high-dimension guided clearance theorem increases with larger parameters, leading to higher returns, but the costs also rise accordingly.\n";
    std::cout << "ss_block_size: Block size for ss (default 102400)\n";
    std::cout << "block_size_growth_rate: Block size growth rate (default 1.05)\n";
    std::cout << "queue_size: maximum size addition (default 65536)\n";
    std::cout << "level1_capacity: Capacity for level 1 array\n";
    std::cout << "level2_capacity: Capacity for level 2 array\n";
    std::cout << "Example: ./specseq++ input.txt output.txt ascii ascii 0 1 10 1024000 1.05 65536 1000 2000\n";
}


void writeVectorToFile(const std::vector<std::pair<int64_t, int64_t>>& vec, const std::string& filename, IOFormat format) {
    std::ofstream file;

    if (format == IOFormat::Binary) {
        file.open(filename, std::ios::out | std::ios::binary);
    }
    else {
        file.open(filename, std::ios::out);
    }

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    if (format == IOFormat::Ascii) {
        // Write size in ASCII format
        file << vec.size() << std::endl;

        // Write each pair in ASCII format
        for (const auto& p : vec) {
            file << p.first << " " << p.second << std::endl;
        }
    }
    else {
        // Write size in binary format
        int64_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));

        // Write each pair in binary format
        for (const auto& p : vec) {
            file.write(reinterpret_cast<const char*>(&p.first), sizeof(p.first));
            file.write(reinterpret_cast<const char*>(&p.second), sizeof(p.second));
        }
    }

    file.close();
    printf("-----------------------------------------------Writing  completed-----------------------------------------------\n");
}

// 打印矩阵函数，用于调试
void printMatrix(const std::vector<std::vector<int64_t>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& col : row) {
            std::cout << col << " ";
        }
        std::cout << std::endl;
    }
}

void dualizeMatrix(std::vector<std::vector<int64_t>>& hv_csc, std::vector<int32_t>& hv_dim) {
    // 创建一个新的矩阵来存储对偶化后的结果
    int32_t n = hv_csc.size();
    std::vector<std::vector<int64_t>> transposed(n);

    // Iterate over each column in the original matrix
    for (int col = 0; col < n; ++col) {
        for (int rowIndex : hv_csc[col]) {
            // Calculate the new row and column indices for the transposed matrix
            int newRow = n - 1 - col;
            int newCol = n - 1 - rowIndex;

            // Add the new row index to the corresponding column in the transposed matrix
            transposed[newCol].push_back(newRow);
        }
    }

    // Sort each column in the transposed matrix
    for (auto& col : transposed) {
        std::sort(col.begin(), col.end());
    }
    hv_csc = transposed;
    for (int col = 0; col < n && col < n - col - 1; ++col) {
        std::swap(hv_dim[col], hv_dim[n - col - 1]);
    }
    return;
}

int main(int argc, char* argv[]) {
    if (argc == 2 && (std::string(argv[1]) == "-help" || std::string(argv[1]) == "-h")) {
        printHelp();
        return 0;
    }

    if (argc < 13) {
        std::cerr << "Invalid number of arguments.\n";
        printHelp();
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    std::string inputFormatStr = argv[3];
    std::string outputFormatStr = argv[4];
    int32_t dualize = std::stoi(argv[5]);
    int32_t model = std::stoi(argv[6]);
    int32_t highclear_maxneighbors = std::stoi(argv[7]);
    int32_t ssBlockSize = std::stoi(argv[8]);
    float blockSizeGrowthRate = std::stof(argv[9]);

    int32_t queuSize = std::stoi(argv[10]);
    int32_t level_capacity[4] = { 0 };
    level_capacity[1] = std::stoi(argv[11]);
    level_capacity[2] = std::stoi(argv[12]);
    // if (argc >= 13) level_capacity[3] = std::stoi(argv[12]);
    // level_capacity[3] = INT_MAX;
    IOFormat inputFormat, outputFormat;

    if (inputFormatStr == "ascii") {
        inputFormat = IOFormat::Ascii;
    }
    else if (inputFormatStr == "binary") {
        inputFormat = IOFormat::Binary;
    }
    else {
        std::cerr << "Invalid input format. Format should be 'ascii' or 'binary'.\n";
        printHelp();
        return 1;
    }

    if (outputFormatStr == "ascii") {
        outputFormat = IOFormat::Ascii;
    }
    else if (outputFormatStr == "binary") {
        outputFormat = IOFormat::Binary;
    }
    else {
        std::cerr << "Invalid output format. Format should be 'ascii' or 'binary'.\n";
        printHelp();
        return 1;
    }

    std::cout << "------------------------------------------------------BUPT------------------------------------------------------\n";
    std::vector<std::vector<int32_t>> hv_csc;
    std::vector<std::vector<int32_t>> hv_neighbor;
    std::vector<int32_t> hv_dim;

    
    processFiles(inputFile, inputFormat, hv_csc, hv_neighbor, hv_dim);
    // if (dualize) {
    //     dualizeMatrix(hv_csc, hv_dim);
    //     // printMatrix(hv_csc);
    // }

    SpecSeqPlusPlus alg(hv_csc, hv_dim, hv_neighbor, level_capacity, dualize, model, highclear_maxneighbors, ssBlockSize, blockSizeGrowthRate, queuSize);
    alg.setDataFileName(extractFileNameWithoutExtension(inputFile));
    float InitTime = alg.GPU_Init();

    // float ComputeTime = alg.GPU_Compute_With_Iner_Block_Diff();
    // float ComputeTime = alg.GPU_Compute_With_Iner_Block_Diff_Naive();
    // float ComputeTime = alg.GPU_Compute_With_Intra_Block_Diff();
    // float ComputeTime = alg.GPU_Compute_With_Intra_Block_Diff_Naive();
    float ComputeTime = alg.GPU_Compute();
    // float ComputeTime = alg.GPU_Compute_Naive();

    printf("Initialization Time: %10.6f seconds\n", InitTime);
    printf("Computation    Time: %10.6f seconds\n", ComputeTime);
    printf("Total          Time: %10.6f seconds\n", InitTime + ComputeTime);
    std::vector<std::pair<int64_t, int64_t>> PersistencePairs = alg.GetPersistencePairs();
    writeVectorToFile(PersistencePairs, outputFile, outputFormat);
    return 0;
}

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
#pragma once
#include "bupt.h"
#include "kernel.h"
#include "timer.h"
#include <unordered_map>
#include "Flag_complex_edge_collapser.h"
template <typename KernelFunc, typename... Args>
float measureKernelTime(KernelFunc kernel, dim3 gridSize, dim3 blockSize, Args... args) {
    GPUTimer timer;
    timer.startTimer();
    kernel << <gridSize, blockSize >> > (args...);
    timer.stopTimer();
    return timer.getElapsedTime();
}
template <typename Func, typename... Args>
float measureFunctionTime(Func func, Args... args) {
    CPUTimer timer;
    timer.startTimer();
    func(args...);
    timer.stopTimer();
    return timer.getElapsedTime();
}

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        auto hash1 = std::hash<T1> {}(pair.first);
        auto hash2 = std::hash<T2> {}(pair.second);
        return hash1 ^ hash2; // 简单的哈希组合方式
    }
};

struct pair_equal {
    template <class T1, class T2>
    bool operator() (const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

class SpecSeqPlusPlus {
private:
    /* Only CPU Storage*/
    std::string DataFileName;
    std::unordered_map<int32_t, int32_t> simplex2line;
    std::unordered_map<int32_t, int32_t> line2simplex;
    int32_t MaxDim = 0;//[DimBegin,DimEnd)
    std::vector<int32_t> DimBegin; // DimBegin[i] represents the starting column index of dimension i
    std::vector<int32_t> DimEnd; // DimEnd[i] represents the ending column index of dimension i
    std::vector<int32_t> DimMaxRow; // DimMaxRow[i] represents the maximum row index that exists in the column of dimension i
    std::vector<int32_t> DimMinRow; // DimMinRow[i] represents the minimum row index that exists in the column of dimension i
    using Filtration_value = float;
    using Vertex_handle = int32_t;
    using Filtered_edge = std::tuple<Vertex_handle, Vertex_handle, Filtration_value>;
    using Filtered_edge_list = std::vector<Filtered_edge>;
    Filtered_edge_list graph;
    std::unordered_map<std::pair<int32_t, int32_t>, int32_t, pair_hash, pair_equal> EdgeId;
    int32_t	NumCols = 0;
    int32_t NumNodes = 0;
    int32_t MAX_BLOCK = 0;

    int32_t GridSize;

    int32_t* h_csc = NULL;
    int64_t* h_offt = NULL;
    int32_t* h_tail = NULL;
    int32_t* h_dim = NULL;

    bool Dualize = 0;//deault 0
    int32_t Model;
    int32_t HighClearMaxNeighbors;
    int32_t SSBlockSize;
    float BlockSizeGrowthRate;
    int32_t QueueSize;
    /* Only GPU Storage*/
    int32_t* d_ell0 = NULL; // Level 0 array (CSC)
    int32_t* d_ell1 = NULL; // Level 1 array
    int32_t* d_ell2 = NULL; // Level 2 array
    int32_t* d_ell3 = NULL; // Level 3 array
    int32_t* d_len = NULL; // Length array

    int32_t* d_rank = NULL; // Column rank
    int64_t* d_offt = NULL; // Column offset
    int32_t* d_tail = NULL; // Column tail

    int32_t* d_leftone = NULL; // Left one array
    int32_t* d_clearflag = NULL; // Clear flag array
    int32_t* d_compressflag = NULL; // Compress flag array

    int32_t* Tag = NULL; // 0: pending column, 1: pivot, 2: [meaningless], 3: empty column, 4: expired pivot
    int32_t* Low = NULL; // Low array

    int32_t* d_aux_arr = NULL; // Merge array

    // __half* MatA = NULL; // Matrix A
    // __half* MatB = NULL; // Matrix B
    // __half* MatC = NULL; // Matrix C

    /* CPU and GPU Storage */
    int32_t* CAP_COL = NULL; // Capacity column
    int32_t* stableflag = NULL; // Stability flag
    int32_t* num_uncols = NULL; // Number of uncolumns
    int32_t* unstable = NULL; // Unstable array
    int32_t* look_up = NULL; // Lookup array

    int32_t* contain_vertics = NULL; // Vertices contained in the column
    int32_t* contain_vertics_offset = NULL; // Offset for vertices contained in the column
    // int64_t* simplex_id = NULL; // Simplex ID corresponding to the column
    int64_t** binomial_coeff = NULL; // Binomial coefficient table
    HashNode* HashList = NULL; // Combination index table

    int32_t* neighbor_list = NULL; // Adjacency list
    int32_t* neighbor_offset = NULL; // Offset for adjacency list
    int32_t* neighbor_len = NULL; // Length of adjacency list

    int32_t cap1;
    int32_t cap2;
    int32_t cap3;


    int32_t* queue_col = NULL;
    int32_t* queue_tag = NULL;
    int32_t* queue_count = NULL;
    std::vector<std::pair<int64_t, int64_t>> PersistencePairs;
    //base alloc
    void MemAlloc1(int32_t* level_capacity) {
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&CAP_COL, sizeof(int32_t) * 5));
        CAP_COL[0] = level_capacity[0];
        CAP_COL[1] = level_capacity[1];
        CAP_COL[2] = level_capacity[2];
        CAP_COL[3] = level_capacity[2] * (BLOCK_SIZE / WARP_SIZE);
        CAP_COL[4] = INT_MAX;
        //GPU alloc
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_len, sizeof(int32_t*) * 5));
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_rank, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_offt, sizeof(int64_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_tail, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_leftone, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_clearflag, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_compressflag, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&Tag, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&Low, sizeof(int32_t) * NumCols));
        //GPU memset
        CHECK_CUDA_ERROR(cudaMemset(d_len, 0, sizeof(int32_t) * 5));
        CHECK_CUDA_ERROR(cudaMemset(d_leftone, -1, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMemset(d_clearflag, 0, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMemset(d_compressflag, -1, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMemset(Tag, 0, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMemset(Low, -1, sizeof(int32_t) * NumCols));
        //CPU and GPU alloc
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&stableflag, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&num_uncols, sizeof(int32_t)));
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&unstable, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&look_up, sizeof(int32_t) * NumCols));
        //CPU and GPU memset
        CHECK_CUDA_ERROR(cudaMemset(stableflag, 0, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMemset(look_up, -1, sizeof(int32_t) * NumCols));
        //Queue
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&queue_col, sizeof(int32_t) * QueueSize));
        // CHECK_CUDA_ERROR(cudaMallocManaged((void**)&queue_count, sizeof(int32_t)));

        CHECK_CUDA_ERROR(cudaMemset(queue_col, -1, sizeof(int32_t) * QueueSize));
        // CHECK_CUDA_ERROR(cudaMemset(queue_count, 0, sizeof(int32_t)));
    }
    //CSC/contain_vertics/HashList/neighbor_list/binomial_coeff alloc
    void MemAlloc2(std::vector<std::vector<int32_t>>& hv_csc, std::vector<int32_t>& hv_dim, std::vector<std::vector<int32_t>>& hv_neighbor) {
        //CSC----------------------------------------------------------------------------------------------------------------
        h_offt = (int64_t*)malloc(sizeof(int64_t) * NumCols);
        h_tail = (int32_t*)malloc(sizeof(int32_t) * NumCols);
        h_dim = (int32_t*)malloc(sizeof(int32_t) * NumCols);
        int64_t offset = 0;
        for (int cur_col = 0;cur_col < hv_csc.size();cur_col++) {
            h_dim[cur_col] = hv_dim[cur_col];
            offset += hv_csc[cur_col].size();
            if (DimBegin[h_dim[cur_col]] == -1)DimBegin[h_dim[cur_col]] = cur_col;
            DimEnd[h_dim[cur_col]] = cur_col + 1;
        }
        h_csc = (int32_t*)malloc(sizeof(int32_t) * offset);
        offset = 0;
        for (int32_t cur_col = 0; cur_col < NumCols; cur_col++) {
            h_tail[cur_col] = -1;
            h_offt[cur_col] = offset;
            for (int32_t i = 0; i < hv_csc[cur_col].size(); i++) {
                h_csc[offset++] = hv_csc[cur_col][i];
                h_tail[cur_col]++;
            }
            if ((Model & 2) && h_dim[cur_col] == 1) {
                graph.push_back(Filtered_edge(hv_csc[cur_col][0], hv_csc[cur_col][1], graph.size()));
                EdgeId[std::make_pair(hv_csc[cur_col][0], hv_csc[cur_col][1])] = cur_col;
            }
        }
        //----------------------------------------------------------------------------------------------------------------------
        if (Model & 1) {
            //contain_vertics-------------------------------------------------------------------------------------------------------
            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&contain_vertics_offset, sizeof(int32_t) * NumCols));
            offset = 0;
            for (int32_t i = 0; i < NumCols; i++) {
                contain_vertics_offset[i] = offset;
                offset += h_dim[i] + 1;
            }
            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&contain_vertics, sizeof(int32_t) * offset));
            //----------------------------------------------------------------------------------------------------------------------
            //HashList--------------------------------------------------------------------------------------------------------------
            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&HashList, sizeof(HashNode) * NumCols));
            //----------------------------------------------------------------------------------------------------------------------
            //neighbor_list---------------------------------------------------------------------------------------------------------
            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&neighbor_len, sizeof(int32_t) * (NumNodes)));
            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&neighbor_offset, sizeof(int32_t) * (NumNodes)));
            offset = 0;
            for (int32_t i = 0; i < NumNodes; i++) {
                sort(hv_neighbor[i].begin(), hv_neighbor[i].end(), std::greater<int>());
                neighbor_len[i] = hv_neighbor[i].size();
                neighbor_offset[i] = offset;
                offset += hv_neighbor[i].size();
            }
            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&neighbor_list, sizeof(int32_t) * (offset)));
            for (int32_t i = 0; i < NumNodes; i++) {
                for (int32_t j = 0; j < hv_neighbor[i].size(); j++)
                    neighbor_list[neighbor_offset[i] + j] = hv_neighbor[i][j];
            }
            //----------------------------------------------------------------------------------------------------------------------
            //binomial_coeff--------------------------------------------------------------------------------------------------------
            CHECK_CUDA_ERROR(cudaMallocManaged((void**)&binomial_coeff, sizeof(int64_t*) * (DimEnd[0] + 1)));
            int32_t p_dim = (MaxDim + 1) / MAXKEY + 1;
            for (int32_t i = 0; i <= DimEnd[0]; i++) {
                CHECK_CUDA_ERROR(cudaMallocManaged((void**)&binomial_coeff[i], sizeof(int64_t) * (p_dim + 1)));
                CHECK_CUDA_ERROR(cudaMemset(binomial_coeff[i], 0, sizeof(int64_t) * (p_dim + 1)));
            }
            for (int64_t i = 0; i <= DimEnd[0]; ++i) {
                binomial_coeff[i][0] = 1;
                for (int64_t j = 1; j < std::min(i, int64_t(p_dim + 1)); ++j)
                    binomial_coeff[i][j] = binomial_coeff[i - 1][j - 1] + binomial_coeff[i - 1][j];
                if (i <= p_dim) binomial_coeff[i][i] = 1;
                if (binomial_coeff[i][std::min(i >> 1, int64_t(p_dim))] < 0) {
                    throw std::overflow_error("simplex index " + std::to_string((uint64_t)i) + " in filtration is larger than maximum index " + std::to_string((int64_t(1) << (8 * sizeof(int64_t) - 1 - 8)) - 1));
                }
            }
            //----------------------------------------------------------------------------------------------------------------------        
        }
    }
    //CPU CSC=> GPU d_ell0
    void MemAlloc3() {
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_ell0, sizeof(int32_t) * (h_offt[NumCols - 1] + h_tail[NumCols - 1] + 1)));
        std::cout << (1.0 * (h_offt[NumCols - 1] + h_tail[NumCols - 1] + 1) * 4 / 1024 / 1024 / 1024) << "GB\n";
        CHECK_CUDA_ERROR(cudaMemcpy(d_ell0, h_csc, sizeof(int32_t) * (h_offt[NumCols - 1] + h_tail[NumCols - 1] + 1), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_tail, h_tail, sizeof(int32_t) * NumCols, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_offt, h_offt, sizeof(int64_t) * NumCols, cudaMemcpyHostToDevice));
    }
    //
    void MemAlloc4() {

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_aux_arr, 1ll * sizeof(int32_t) * QueueSize * WARP_EXTRA_SIZE));
        std::cout << (1.0 * QueueSize * WARP_EXTRA_SIZE * 4 / 1024 / 1024 / 1024) << "GB\n";

        std::cout << (QueueSize * WARP_EXTRA_SIZE) << "\n";
        std::cout << (QueueSize / (BLOCK_SIZE / WARP_SIZE) * BLOCK_EXTRA_SIZE) << "\n";
        CHECK_CUDA_ERROR(cudaMemset(d_aux_arr, 0, 1ll * sizeof(int32_t) * QueueSize * WARP_EXTRA_SIZE));

        cap1 = num_uncols[0] + 1;
        cap2 = (num_uncols[0] / (CAP_COL[2] / CAP_COL[1])) / 2 + 1;
        cap3 = (num_uncols[0] / (CAP_COL[3] / CAP_COL[1])) / 2 + 1;
        // if (CAP_COL[3])cap3 = (num_uncols[0] / (CAP_COL[3] / CAP_COL[1])) + 1;

        int64_t siz1 = 1ll * sizeof(int32_t) * CAP_COL[1] * cap1;
        int64_t siz2 = 1ll * sizeof(int32_t) * CAP_COL[2] * cap2;
        int64_t siz3 = 1ll * sizeof(int32_t) * CAP_COL[3] * cap3;


        printf("CAP:%d %d %d %d\n", CAP_COL[1], CAP_COL[2], CAP_COL[3], CAP_COL[4]);
        printf("%d %d %d\n", cap1, cap2, cap3);
        printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║ Level 1 array capacity: %10" PRId64 " | Level 2 array capacity: %10" PRId64 " | Level 3 array capacity: %10" PRId64 " ║\n", int64_t(cap1), int64_t(cap2), int64_t(cap3));
        printf("║ Level 1 array mem:    %10.2fMB | Level 2 array mem:    %10.2fMB | Level 3 array mem:    %10.2fMB ║\n", siz1 / 1024.0 / 1024.0, siz2 / 1024.0 / 1024.0, siz3 / 1024.0 / 1024.0);
        printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_ell1, siz1));
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_ell2, siz2));
        // if (siz3)
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_ell3, siz3));

    }
    void DualizeMatrix() {
        // printGPUArrayRange(d_ell0, 0, (h_offt[NumCols - 1] + h_tail[NumCols - 1]));
        // printGPUArrayRange(d_tail, 0, NumCols - 1);
        // printGPUArrayRange(d_offt, 0, NumCols - 1);


        int32_t* d_col_counts = NULL;
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_col_counts, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMemset(d_col_counts, 0, sizeof(int32_t) * NumCols));
        countNonZeros << <GridSize, BLOCK_SIZE >> > (d_ell0, d_tail, d_offt, d_col_counts, NumCols);
        cudaDeviceSynchronize();

        int64_t* d_new_offt = NULL;
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_new_offt, sizeof(int64_t) * NumCols));
        computeOffsets << <GridSize, BLOCK_SIZE >> > (d_col_counts, d_new_offt, NumCols);
        cudaDeviceSynchronize();
        thrust::exclusive_scan(thrust::device, d_new_offt, d_new_offt + NumCols, d_new_offt);

        int32_t* d_new_ell0 = NULL;
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_new_ell0, sizeof(int32_t) * (getDeviceValue(d_new_offt + NumCols - 1) + getDeviceValue(d_col_counts + NumCols - 1))));
        CHECK_CUDA_ERROR(cudaMemset(d_col_counts, 0, sizeof(int32_t) * NumCols));
        dualizeMatrixKernel << <GridSize, BLOCK_SIZE >> > (d_ell0, d_tail, d_offt, d_new_ell0, d_col_counts, d_new_offt, NumCols);
        cudaDeviceSynchronize();
        replaceGPUArrays(d_ell0, d_new_ell0);
        replaceGPUArrays(d_tail, d_col_counts);
        replaceGPUArrays(d_offt, d_new_offt);
        for (int col = 0; col < NumCols; ++col) {
            thrust::sort(d_ell0 + d_offt[col], d_ell0 + d_offt[col] + d_tail[col]);
            d_tail[col]--;
        }


        int32_t* new_stableflag = NULL;
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&new_stableflag, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMemset(new_stableflag, 0, sizeof(int32_t) * NumCols));
        int32_t* new_look_up = NULL;
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&new_look_up, sizeof(int32_t) * NumCols));
        CHECK_CUDA_ERROR(cudaMemset(new_look_up, -1, sizeof(int32_t) * NumCols));
        dualizeAuxArray << <GridSize, BLOCK_SIZE >> > (stableflag, look_up, new_stableflag, new_look_up, d_ell0, d_offt, Low, d_tail, NumCols);
        cudaDeviceSynchronize();

        replaceGPUArrays(stableflag, new_stableflag);
        replaceGPUArrays(look_up, new_look_up);

        for (int col = 1;col < NumCols && col < NumCols - col;++col) {
            std::swap(h_dim[col], h_dim[NumCols - col]);
        }
        // int cnt0 = 0, cnt3 = 0;
        // for (int i = 1;i < NumCols;i++) {
        //     if (stableflag[i] == 0)cnt0++;
        //     else if (stableflag[i] == 3)cnt3++;
        //     else std::cout << "err\n";
        // }
        // std::cout << cnt0 << " " << cnt3 << "\n";
        // printGPUArrayRange(d_ell0, 0, (getDeviceValue(d_new_offt + NumCols - 1) + getDeviceValue(d_col_counts + NumCols - 1)));
        // printGPUArrayRange(d_tail, 0, NumCols - 1);
        // printGPUArrayRange(d_offt, 0, NumCols - 1);
        // printGPUArrayRange(stableflag, 0, NumCols - 1);
        // printGPUArrayRange(look_up, 0, NumCols - 1);
        // printGPUArrayRange(Low, 0, NumCols - 1);
    }
    //
    void ColumnShrink() {
        DimBegin.assign(MaxDim + 1, -1);
        DimEnd.assign(MaxDim + 1, -1);
        DimMaxRow.assign(MaxDim + 1, -1);
        DimMinRow.assign(MaxDim + 1, -1);
        num_uncols[0] = 0;
        // printGPUArrayRange(stableflag, 0, NumCols - 1);
        // printCPUArrayRange(h_dim, 0, NumCols - 1);

        for (int32_t i = 1; i < NumCols; i++) {
            if (stableflag[i] == 0) {
                unstable[num_uncols[0]++] = i;

                if (DimBegin[h_dim[i]] == -1)DimBegin[h_dim[i]] = num_uncols[0] - 1;
                DimEnd[h_dim[i]] = num_uncols[0];

                MaxDim = std::max(MaxDim, h_dim[i]);
                if (DimMaxRow[h_dim[i]] < d_ell0[d_offt[i] + d_tail[i]] + 1)
                    DimMaxRow[h_dim[i]] = d_ell0[d_offt[i] + d_tail[i]] + 1;

            }
            // if (i == 67674) {
            //     printf("dim %d DimMinRow%d", h_dim[67674], DimMinRow[h_dim[67674]]);
            //     for(int j=d_offt[67674];j<d_offt[67675];j++){
            //         std::cout<<d_ell0[j]<<",";
            //     }
            // }
            if (d_tail[i] != -1) {
                if (DimMinRow[h_dim[i]] == -1)
                    DimMinRow[h_dim[i]] = d_ell0[d_offt[i]];
                else
                    DimMinRow[h_dim[i]] = std::min(DimMinRow[h_dim[i]], d_ell0[d_offt[i]]);
            }

        }
        // printCPUArrayRange(DimBegin, 0, MaxDim);
        // printCPUArrayRange(DimEnd, 0, MaxDim);
        // printCPUArrayRange(DimMinRow, 0, MaxDim);
        // printCPUArrayRange(DimMaxRow, 0, MaxDim);
        for (int32_t i = 0; i <= MaxDim; i++) {
            if (DimMaxRow[i] == -1)continue;
            int32_t DimCols = DimEnd[i] - DimBegin[i];
            MAX_BLOCK = std::max(MAX_BLOCK, ((DimCols % BLOCK_SIZE == 0) ? (DimCols / BLOCK_SIZE) : (DimCols / BLOCK_SIZE + 1)));
        }
    }
    void GUDHICollapse() {
        std::cout << graph.size() << " col \n";
        auto remaining_edges = Gudhi::collapse::flag_complex_collapse_edges(graph);
        // remaining_edges.clear();
        graph = std::move(remaining_edges);
        std::cout << DimEnd[1] - DimBegin[1] << " col \n";
        std::cout << DimEnd[1] << " " << DimBegin[1] << "  \n";
        for (int i = DimBegin[1];i < DimEnd[1];i++) {
            stableflag[i] = 3;
        }
        for (auto filtered_edge_from_collapse : graph) {
            stableflag[EdgeId[std::make_pair(std::get<0>(filtered_edge_from_collapse), std::get<1>(filtered_edge_from_collapse))]] = 0;
            // std::cout << () << " " << () << " " << DimBegin[1] + int32_t(std::get<2>(filtered_edge_from_collapse)) << "\n";
        }
    }
    float EdgeCollapse() {
        float GPU_time = 0, CPU_time = 0, t = 0;
        t = measureFunctionTime([this]() { this->GUDHICollapse(); });
        CPU_time += t;
        for (int32_t cur_dim = 1;cur_dim <= MaxDim; cur_dim++) {
            int32_t DimCols = DimEnd[cur_dim] - DimBegin[cur_dim];
            dim3 gridSize = (DimCols % BLOCK_SIZE == 0) ? (DimCols / BLOCK_SIZE) : (DimCols / BLOCK_SIZE + 1);
            dim3 blockSize = BLOCK_SIZE;
            t = measureKernelTime(SimplexCollapse, gridSize, blockSize,
                d_ell0, d_offt, d_tail, cur_dim, DimBegin[cur_dim], DimEnd[cur_dim], stableflag);
            GPU_time += t;
        }
        return GPU_time + CPU_time;
    }
    void Formatted(std::vector<std::vector<int32_t>>& hv_csc, std::vector<int32_t>& hv_dim, std::vector<std::vector<int32_t>>& hv_neighbor) {

        int32_t NUMCOLS = hv_csc.size();
        int32_t MAXDIM = *max_element(hv_dim.begin(), hv_dim.end());
        std::vector<int32_t> DimCols[MAXDIM + 1];
        for (int colIdx = 0; colIdx < NUMCOLS; colIdx++) {
            DimCols[hv_dim[colIdx]].push_back(colIdx);
        }
        NumNodes = DimCols[0].size();

        std::vector<std::vector<int32_t>> R_CSC;
        std::vector<int32_t> R_DIM;
        int32_t lineId = -1;
        for (int dim = 0; dim <= MAXDIM; dim++) {
            for (auto colIdx : DimCols[dim]) {

                line2simplex[colIdx] = ++lineId;
                // std::cout<<colIdx<<":"<<line2simplex[colIdx] <<"\n";
                simplex2line[lineId] = colIdx;
                std::vector<int32_t> temp_col;
                for (auto rowIdx : hv_csc[colIdx]) {
                    // std::cout<<rowIdx<<" "<<line2simplex[rowIdx]<<"\n";
                    temp_col.push_back(line2simplex[rowIdx]);
                }
                R_CSC.push_back(temp_col);
                R_DIM.push_back(dim);
                if (dim == 0) {
                    hv_neighbor.push_back(std::vector<int32_t>());
                }
                else if (dim == 1) {
                    // std::cout<<hv_neighbor.size()<<" "<<line2simplex[temp_col[0]]<<" "<<line2simplex[temp_col[1]]<<"\n";
                    hv_neighbor[temp_col[0]].push_back(temp_col[1]);
                    hv_neighbor[temp_col[1]].push_back(temp_col[0]);
                }
            }

        }
        for (int i = 0;i < hv_neighbor.size();i++) {
            for (int j = 0;j < hv_neighbor[i].size();j++) {
                hv_neighbor[i][j] = line2simplex[hv_neighbor[i][j]];
            }
        }
        std::swap(hv_csc, R_CSC);
        std::swap(hv_dim, R_DIM);
    }
public:
    SpecSeqPlusPlus() {}
    SpecSeqPlusPlus(std::vector<std::vector<int32_t>>& hv_csc, std::vector<int32_t>& hv_dim, std::vector<std::vector<int32_t>>& hv_neighbor, int32_t* level_capacity, int32_t dualize, int32_t model, int32_t highclear_maxneighbors, int32_t ssBlockSize, float blockSizeGrowthRate, int32_t queuSize) {
        Formatted(hv_csc, hv_dim, hv_neighbor);
        std::cout << "pass\n";
        NumCols = hv_csc.size();
        NumNodes = hv_neighbor.size();
        Model = model;
        HighClearMaxNeighbors = highclear_maxneighbors;
        SSBlockSize = ssBlockSize;
        BlockSizeGrowthRate = blockSizeGrowthRate;
        QueueSize = queuSize;
        Dualize = dualize;
        // if (Dualize)MaxDim = hv_dim[0];
        // else MaxDim = hv_dim[hv_dim.size() - 1];
        MaxDim = hv_dim[hv_dim.size() - 1];
        GridSize = (NumCols % BLOCK_SIZE == 0) ? (NumCols / BLOCK_SIZE) : (NumCols / BLOCK_SIZE + 1);
        DimBegin.resize(MaxDim + 1);DimBegin.assign(MaxDim + 1, -1);
        DimEnd.resize(MaxDim + 1);DimEnd.assign(MaxDim + 1, -1);
        DimMaxRow.resize(MaxDim + 1);DimMaxRow.assign(MaxDim + 1, -1);
        DimMinRow.resize(MaxDim + 1);DimMinRow.assign(MaxDim + 1, -1);
        printf("  NumCols: %10d                  NumNodes: %10d                  Maxdim: %10d\n", NumCols, NumNodes, MaxDim);
        MemAlloc1(level_capacity);
        std::cout << "malloc1 done\n";
        MemAlloc2(hv_csc, hv_dim, hv_neighbor);
        std::cout << "malloc2 done\n";
        MemAlloc3();
        std::cout << "malloc3 done\n";
    }
    float GPU_Init() {
        float GPU_time = 0, CPU_time = 0, t = 0, EdgeCollapse_time = 0;

        // ljl highclear time
        float highclear_time = 0;
        if (MaxDim >= 1 && (Model & 2)) {
            EdgeCollapse_time = EdgeCollapse();
        }


        for (int32_t cur_dim = 0; (Model & 1) && cur_dim <= MaxDim; cur_dim++) {
            int32_t DimCols = DimEnd[cur_dim] - DimBegin[cur_dim];
            dim3 blockSize = BLOCK_SIZE;
            dim3 gridSize = (DimCols % BLOCK_SIZE == 0) ? (DimCols / BLOCK_SIZE) : (DimCols / BLOCK_SIZE + 1);

            t = measureKernelTime(init_contain_vertics, gridSize, blockSize,
                d_ell0, d_offt, d_tail, cur_dim, DimBegin[cur_dim], DimEnd[cur_dim], contain_vertics, contain_vertics_offset, binomial_coeff, HashList, neighbor_len, neighbor_list, neighbor_offset);
            GPU_time += t;
            std::cout << cur_dim << ":" << t << "\n";
            // printGPUArrayRange(contain_vertics,contain_vertics_offset[DimBegin[cur_dim]],contain_vertics_offset[DimEnd[cur_dim]-1]+cur_dim);
            thrust::sort(HashList + DimBegin[cur_dim], HashList + DimEnd[cur_dim]);

            if (cur_dim == MaxDim) {
                t = measureKernelTime(maxdim_clear_init, gridSize, blockSize,
                    NumNodes, cur_dim, DimBegin[cur_dim], DimEnd[cur_dim], HighClearMaxNeighbors, contain_vertics, contain_vertics_offset, binomial_coeff, HashList + DimBegin[cur_dim], neighbor_len, neighbor_list, neighbor_offset, d_clearflag, stableflag, Tag);
                GPU_time += t;
                highclear_time += t;
                std::cout << "highclear:" << t << "\n";
            }

        }
        // #if __cplusplus >= 201703L
#ifdef HIGHOUT
        std::cout << "highclear_time: " << highclear_time / 1000 << std::endl;
        if ((Model & 1))out_maxdimclear();
        exit(0);
#endif
        // #else
                // #error "This code requires at least C++17"
        // #endif



        t = measureKernelTime(initleft_kernel, GridSize, BLOCK_SIZE,
            CAP_COL, d_ell0, d_rank, d_offt, d_tail, Low, d_leftone, d_clearflag, look_up, NumCols);
        GPU_time += t;

        t = measureKernelTime(initlookup_kernel, GridSize, BLOCK_SIZE,
            Tag, CAP_COL, d_ell0, d_offt, d_tail, Low, d_leftone, d_clearflag, d_compressflag, stableflag, look_up, NumCols);
        GPU_time += t;

        t = measureKernelTime(clearing_kernel, GridSize, BLOCK_SIZE,
            d_tail, Low, d_clearflag, NumCols, look_up);
        GPU_time += t;

        t = measureKernelTime(compress_kernel, GridSize, BLOCK_SIZE,
            Tag, CAP_COL, d_ell0, d_offt, d_tail, Low, d_compressflag, look_up, NumCols);
        GPU_time += t;

        if (Dualize)DualizeMatrix();

        t = measureFunctionTime([this]() { this->ColumnShrink(); });
        CPU_time += t;
        printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║ Before ColumnShrink operation - Unprocessed Columns: %10d                                              ║\n", NumCols);
        printf("║ After  ColumnShrink operation - Unprocessed Columns: %10d, Maximum Blocks: %10d                  ║\n", num_uncols[0], MAX_BLOCK);
        printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
        MemAlloc4();

        // t = measureKernelTime(init_MatB, MAX_BLOCK, BLOCK_SIZE, MatB);
        // GPU_time += t;
        CPU_time /= 1000.0;
        GPU_time /= 1000.0;
        EdgeCollapse_time /= 1000.0;
        std::cout << CPU_time << " " << GPU_time << " " << EdgeCollapse_time << "\n";
        printf("-----------------------------------------------GPU_Init completed-----------------------------------------------\n");
        return GPU_time + CPU_time + EdgeCollapse_time;
    }

    float GPU_Compute() {
        bool* Flag = NULL;
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&Flag, sizeof(bool)));
        int debugmaxlen = 0;

        float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t = 0;
        int32_t DimForStart = Dualize == 1 ? 0 : MaxDim;
        int32_t DimForEnd = Dualize == 1 ? MaxDim + 1 : -1;
        int32_t DimForStep = Dualize == 1 ? 1 : -1;
        for (int32_t cur_dim = DimForStart; cur_dim != DimForEnd; cur_dim += DimForStep) {
            if (DimMaxRow[cur_dim] == -1) continue;
            int32_t ss_size = SSBlockSize;
            int32_t ss_begin = DimBegin[cur_dim], ss_end = std::min(DimBegin[cur_dim] + ss_size, DimEnd[cur_dim]);
            int32_t ss_minrow = DimMaxRow[cur_dim] - ss_size, ss_maxrow = ss_minrow + ss_size;
            GridSize = (ss_end - ss_begin + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::cout << "cur_dim: " << cur_dim << std::endl;
            std::cout << "GridSize: " << GridSize << std::endl;
            std::cout << "ss_begin: " << ss_begin << std::endl;
            std::cout << "ss_end: " << ss_end << std::endl;
            std::cout << "ss_minrow: " << ss_minrow << std::endl;
            std::cout << "ss_maxrow: " << ss_maxrow << std::endl;
            std::cout << "DimEnd[cur_dim]: " << DimEnd[cur_dim] << std::endl;
            std::cout << "DimMinRow[cur_dim]: " << DimMinRow[cur_dim] << std::endl;
            std::cout << "DimMaxRow[cur_dim]: " << DimMaxRow[cur_dim] << std::endl;
            while (ss_begin < DimEnd[cur_dim] || ss_maxrow > DimMinRow[cur_dim]) {
                CHECK_CUDA_ERROR(cudaMemset(queue_col, -1, sizeof(int32_t) * QueueSize));
                while (1) {
                    Flag[0] = 0;
                    t = measureKernelTime(BlockScan, GridSize, BLOCK_SIZE,
                        ss_begin, ss_end, ss_minrow, 1, d_rank, Low, stableflag, unstable, look_up);
                    t2 += t;
                    checkCudaLastError("kernel1");
                    t = measureKernelTime(BlockGenAddition, GridSize, BLOCK_SIZE,
                        Flag, ss_begin, ss_end, ss_minrow, 1, d_rank, Low, stableflag, unstable, look_up, queue_col, QueueSize);
                    t2 += t;
                    checkCudaLastError("kernel2");
                    thrust::device_ptr<int> dev_ptr(queue_col);
                    checkCudaLastError("dp");
                    int32_t count = thrust::count_if(dev_ptr, dev_ptr + QueueSize, thrust::placeholders::_1 != -1);
                    checkCudaLastError("thrust");
                    // std::cout << count << "P1\n";

                    if (!Flag[0] || count < QueueSize / (BLOCK_SIZE / WARP_SIZE))break;
                    t = measureKernelTime(ProcColumnPair, QueueSize, 32,
                        d_ell0, d_ell1, d_ell2, d_ell3, d_len, cap1, cap2, cap3, CAP_COL, d_rank, d_offt, d_tail, queue_col, QueueSize, Low, stableflag, look_up, d_aux_arr);
                    checkCudaLastError("kernel3");
                    t4 += t;
                    // std::cout << "add done\n";
                    // break;
                }
                CHECK_CUDA_ERROR(cudaMemset(queue_col, -1, sizeof(int32_t) * QueueSize));
                while (1) {
                    Flag[0] = 0;
                    // queue_count[0] = 0;
                    t = measureKernelTime(BlockScan, GridSize, BLOCK_SIZE,
                        ss_begin, ss_end, ss_minrow, 2, d_rank, Low, stableflag, unstable, look_up);
                    t2 += t;
                    checkCudaLastError("kernel1");
                    t = measureKernelTime(BlockGenAddition, GridSize, BLOCK_SIZE,
                        Flag, ss_begin, ss_end, ss_minrow, 2, d_rank, Low, stableflag, unstable, look_up, queue_col, QueueSize);
                    t2 += t;
                    checkCudaLastError("kernel2");
                    thrust::device_ptr<int> dev_ptr(queue_col);
                    checkCudaLastError("dp");
                    int32_t count = thrust::count_if(dev_ptr, dev_ptr + QueueSize, thrust::placeholders::_1 != -1);
                    checkCudaLastError("thrust");
                    std::cout << count << "P2\n";
                    if (!Flag[0] || count < 5)break;
                    // std::cout << "min(QueueSize / (BLOCK_SIZE / WARP_SIZE), count)" << min(QueueSize / (BLOCK_SIZE / WARP_SIZE), count) << "\n";
                    t = measureKernelTime(ProcColumnPair_121Block, min(QueueSize / (BLOCK_SIZE / WARP_SIZE), count), BLOCK_SIZE,
                        d_ell0, d_ell1, d_ell2, d_ell3, d_len, cap1, cap2, cap3, CAP_COL, d_rank, d_offt, d_tail, queue_col, QueueSize, Low, stableflag, look_up, d_aux_arr);
                    checkCudaLastError("kernel4");
                    // std::cout<<"add done\n";
                    t4 += t;
                }



                //The CPU processes ultra-long columns.
                // for (int i = DimBegin[cur_dim];i < DimEnd[cur_dim];i++) {
                //     int32_t col1Id = unstable[i];
                //     int32_t* column1 = NULL;
                //     int32_t col1Len = d_tail[col1Id] + 1;
                //     if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
                //     else if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
                //     else if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
                //     for (int j = 1;j < col1Len;j++){
                //         if(column1[j]<column1[j-1]){
                //             std::cout<<"Err";
                //             for (int j = 0;j < col1Len;j++)std::cout << column1[j] << " ";std::cout << "\n";
                //         }
                //     }
                // }
                ss_size *= BlockSizeGrowthRate;
                if (ss_maxrow <= DimMinRow[cur_dim]) {
                    ss_begin = ss_end;
                }
                else {
                    ss_minrow -= ss_size;
                    ss_maxrow -= ss_size;
                }
                if (ss_end < DimEnd[cur_dim]) {
                    ss_end += ss_size;
                    ss_end = std::min(ss_end, DimEnd[cur_dim]);
                }
                GridSize = (ss_end - ss_begin + BLOCK_SIZE - 1) / BLOCK_SIZE;

                std::cout << "GridSize: " << GridSize << std::endl;
                std::cout << "ss_begin: " << ss_begin << std::endl;
                std::cout << "ss_end: " << ss_end << std::endl;
                std::cout << "ss_minrow: " << ss_minrow << std::endl;
                std::cout << "ss_maxrow: " << ss_maxrow << std::endl;
            }
            int32_t cur_max_len = 0;
            std::cout << "cpu stage\n";
            CPUTimer timer;
            timer.startTimer();
            std::vector<std::vector<int32_t>>cpuColumn;
            std::vector<int32_t>temp_col;
            int32_t lenThreshold = 0;


            // int32_t* CCC = NULL;
            // if (d_rank[67674] == 1)CCC = &d_ell1[d_offt[67674]];
            // else if (d_rank[67674] == 0)CCC = &d_ell0[d_offt[67674]];
            // else if (d_rank[67674] == 2)CCC = &d_ell2[d_offt[67674]];
            // else if (d_rank[67674] == 3)CCC = &d_ell3[d_offt[67674]];
            // else if (d_rank[67674] == 4)CCC = cpuColumn[d_offt[67674]].data();

            // for (int i = 0;i <= d_tail[67674];i++) {
            //     std::cout << CCC[i] << ",";
            // }
            for (int i = DimBegin[cur_dim];i < DimEnd[cur_dim];i++) {
                int32_t colId = unstable[i];
                if (stableflag[colId] == 3)continue;
                // std::cout << "yes" << colId << "\n";
                int32_t colLen = d_tail[colId] + 1;
                // std::cout << "colLen" << colLen << std::endl;
                if (colLen > lenThreshold) {

                    int32_t nonZero = colLen;
                    int32_t lowIndex = 0;
                    // std::cout << "colLen > 30000" << std::endl;
                    // std::cout << "colId" << colId << "\n";
                    // 获取最大的索引
                    int32_t maxIndex = Low[colId];
                    // std::cout << "maxIndex" << maxIndex << "\n";
                    // 每段vector的大小
                    int32_t vectorSize = 64;
                    // 计算需要多少段vector
                    int32_t vectorNum = ((maxIndex - DimMinRow[cur_dim] + 1) + (vectorSize - 1)) / vectorSize;
                    // std::cout << "vectorNum" << vectorNum << "\n";
                    // 初始化长列的稠密存储
                    // if (vectorNum <= 0)printf("ERROR1\n");
                    std::vector<uint64_t>column(vectorNum, 0);
                    // std::cout << "passbegin\n";
                    int32_t col1Id = colId;
                    int32_t* column1 = NULL;
                    // if (d_rank[col1Id] == 1 && d_offt[col1Id] > cap1 * CAP_COL[1])printf("ERROR offt 1");
                    // else if (d_rank[col1Id] == 2 && d_offt[col1Id] > cap2 * CAP_COL[2])printf("ERROR offt 2");
                    // else if (d_rank[col1Id] == 3 && d_offt[col1Id] > cap3 * CAP_COL[3])printf("ERROR offt 3");
                    // else if (d_rank[col1Id] == 4 && d_offt[col1Id] >= cpuColumn.size())printf("ERROR offt 4");

                    if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
                    else if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
                    else if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
                    else if (d_rank[col1Id] == 3)column1 = &d_ell3[d_offt[col1Id]];
                    else if (d_rank[col1Id] == 4)column1 = cpuColumn[d_offt[col1Id]].data();

                    lowIndex = vectorNum - 1;
                    // 使用稠密模式存储
                    // std::cout << "pass2\n";
                    for (int32_t k = 0;k < colLen;k++) {
                        int32_t vectorIndex = (column1[k] - DimMinRow[cur_dim]) / vectorSize;
                        int16_t index = (column1[k] - DimMinRow[cur_dim]) % vectorSize;
                        // if (vectorIndex < 0 || vectorIndex >= column.size())printf("ERROR vindex\n");
                        // if (index < 0 || index >= 64)printf("%d %d %d ERROR index\n", column1[k], DimMinRow[cur_dim], index);
                        column[vectorIndex] |= (uint64_t(1) << index);
                    }
                    while (Low[colId] != -1 && look_up[Low[colId]] != -1) {
                        int32_t pivotcolId = look_up[Low[colId]];
                        if (colId < pivotcolId) {
                            stableflag[look_up[Low[colId]]] = 0;
                            look_up[Low[colId]] = colId;
                            stableflag[colId] = 3;
                            stableflag[Low[colId]] = 3;
                            break;
                        }

                        int32_t col2Id = pivotcolId;
                        int32_t* column2 = NULL;
                        int32_t col2Len = d_tail[col2Id] + 1;
                        // if (d_rank[col2Id] == 1 && d_offt[col2Id] > cap1 * CAP_COL[1])printf("ERROR offt 1");
                        // else if (d_rank[col2Id] == 2 && d_offt[col2Id] > cap2 * CAP_COL[2])printf("ERROR offt 2");
                        // else if (d_rank[col2Id] == 3 && d_offt[col2Id] > cap3 * CAP_COL[3])printf("ERROR offt 3");
                        // else if (d_rank[col2Id] == 4 && d_offt[col2Id] >= cpuColumn.size())printf("ERROR offt 4");

                        if (d_rank[col2Id] == 1)column2 = &d_ell1[d_offt[col2Id]];
                        else if (d_rank[col2Id] == 0)column2 = &d_ell0[d_offt[col2Id]];
                        else if (d_rank[col2Id] == 2)column2 = &d_ell2[d_offt[col2Id]];
                        else if (d_rank[col2Id] == 3)column2 = &d_ell3[d_offt[col2Id]];
                        else if (d_rank[col2Id] == 4)column2 = cpuColumn[d_offt[col2Id]].data();



                        // if (col1Id == 500696)
                        // printf("col2Id: %d ", col2Id);
                        // if (col1Id == 500696)
                        // printf("col2Len: %d\n", col2Len);


                        // openMP 并行约减

                        // if (col1Id == 500696)std::cout << col1Id << "+" << col2Id << " low" << Low[col1Id] << " len" << nonZero << " lowindex" << lowIndex << "minrow" << DimMinRow[cur_dim] << " \n";
                        // if (col1Id == 500696)for (int k = 0;k < column.size();k++)if (column[k])std::cout << k << ":", printBinary(column[k]);
                        // if (col1Id == 500696)
                        // printf("=>\n");
                        // #pragma omp parallel for
                        for (int k = 0;k < col2Len;k++) {
                            int32_t vectorIndex = (column2[k] - DimMinRow[cur_dim]) / vectorSize;
                            int16_t index = (column2[k] - DimMinRow[cur_dim]) % vectorSize;
                            uint64_t bitmask = (uint64_t(1) << index);
                            // if (vectorIndex < 0 || vectorIndex >= column.size())printf("ERROR\n");
                            // if (index < 0 || index >= 64)printf("ERROR\n");
                            // #pragma omp atomic
                            column[vectorIndex] ^= bitmask;

                            // if (col1Id == 500696)std::cout << vectorIndex << ":", printBinary((uint64_t(1) << index));
                            if (column[vectorIndex] & bitmask)nonZero++;
                            else nonZero--;
                        }

                        // if (col1Id == 500696)
                        // for (int k = 0;k < column.size();k++)if (column[k])std::cout << k << ":", printBinary(column[k]);

                        // 是否约减完毕
                        if (nonZero == 0)Low[col1Id] = -1;
                        else {
                            for (int k = lowIndex;k >= 0;k--) {
                                if (column[k]) {
                                    Low[col1Id] = k * vectorSize + (63 - __builtin_clzll(uint64_t(column[k]))) + DimMinRow[cur_dim];
                                    lowIndex = k;
                                    vectorNum = k + 1;
                                    break;
                                }
                            }
                            // for (int k = lowIndex + 1;k < column.size();k++) {
                            //     if (column[k]) {
                            //         std::cout << "err\n";
                            //         exit(0);
                            //     }
                            // }
                        }
                        // std::cout << col1Id << "+" << col2Id << " low" << Low[col1Id] << " len" << nonZero << " lowindex" << lowIndex << "minrow" << DimMinRow[cur_dim] << " \n\n";
                        // 514518+456883
                        // if (col2Id == 456883 && col1Id == 514518)exit(0);

                        // int test = 0;
                        // for (auto k : column) {
                        //     test += __builtin_popcountll(k);
                        // }
                        // if (test != nonZero) {
                        //     std::cout << col1Id << "\n";
                        //     std::cout << "err\n";
                        //     exit(0);
                        // }

                    }
                    // if (col1Id == 500696)exit(0);
                    // if (col1Id == 502602)exit(0);
                    //2024.10.10.11:44 上文已修改
                    // 把稠密存储存回稀疏存储
                    std::vector<int32_t>temp_col;
                    int temp_index = 0;
                    for (int k = 0;k < vectorNum;k++) {
                        while (column[k]) {
                            uint64_t lowbit = column[k] & (-column[k]);
                            // std::cout << lowbit << " <-\n";
                            temp_col.push_back(k * vectorSize + __builtin_ctzll(lowbit) + DimMinRow[cur_dim]);
                            column[k] -= lowbit;
                        }
                    }

                    if (nonZero > CAP_COL[d_rank[col1Id]]) {
                        d_rank[col1Id]++;
                        while (temp_col.size() > CAP_COL[d_rank[col1Id]])d_rank[col1Id]++;
                        if (d_rank[col1Id] == 1) {
                            if (d_len[d_rank[col1Id]] > cap1) {
                                printf("Level 1 array's capbility is not enough\n");
                                d_rank[col1Id] = 4;cpuColumn.push_back(temp_col);d_offt[col1Id] = cpuColumn.size() - 1;
                            }
                            else d_offt[col1Id] = 1ll * (d_len[d_rank[col1Id]]++) * CAP_COL[d_rank[col1Id]];
                        }
                        else if (d_rank[col1Id] == 2) {
                            if (d_len[d_rank[col1Id]] >= cap2) {
                                printf("Level 2 array's capbility%d is not enough %d\n", cap2, d_len[d_rank[col1Id]]);
                                d_rank[col1Id] = 4;cpuColumn.push_back(temp_col);d_offt[col1Id] = cpuColumn.size() - 1;
                            }
                            else d_offt[col1Id] = 1ll * (d_len[d_rank[col1Id]]++) * CAP_COL[d_rank[col1Id]];
                        }
                        else if (d_rank[col1Id] == 3) {
                            if (d_len[d_rank[col1Id]] >= cap3) {
                                printf("Level 2 array's capbility is not enough\n");
                                d_rank[col1Id] = 4;cpuColumn.push_back(temp_col);d_offt[col1Id] = cpuColumn.size() - 1;
                            }
                            else d_offt[col1Id] = 1ll * (d_len[d_rank[col1Id]]++) * CAP_COL[d_rank[col1Id]];
                        }
                        else if (d_rank[col1Id] == 4) {
                            cpuColumn.push_back(temp_col);
                            d_offt[col1Id] = cpuColumn.size() - 1;
                        }
                        else {
                            printf("$d Level array's level is not enough on cpu\n", d_rank[col1Id]);
                        }
                        if (d_offt[col1Id] < 0)printf("ERROR");
                    }

                    d_tail[col1Id] = temp_col.size() - 1;
                    if (temp_col.size() == 0)stableflag[col1Id] = 3;

                    if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
                    else if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
                    else if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
                    else if (d_rank[col1Id] == 3)column1 = &d_ell3[d_offt[col1Id]];
                    else if (d_rank[col1Id] == 4)column1 = cpuColumn[d_offt[col1Id]].data();

                    Low[col1Id] = (temp_col.size() == 0) ? -1 : temp_col[temp_col.size() - 1];
                    // std::cout << "Low[col1Id]" << Low[col1Id] << "\n";
                    // debugmaxlen = std::max(debugmaxlen, int(temp_col.size()));
                    // cur_max_len = std::max(cur_max_len, int(temp_col.size()));

                    if (d_rank[col1Id] < 4) {
                        for (int i = 0;i < temp_col.size();i++) {
                            column1[i] = temp_col[i];
                        }
                    }
                    // std::cout << "pass\n";
                    // else cpuColumn[d_offt[col1Id]].swap(temp_col);
                    // std::cout << "pass3\n";
                    if (Low[colId] != -1 && look_up[Low[colId]] == -1) {
                        look_up[Low[colId]] = colId;
                        stableflag[colId] = 3;
                        stableflag[Low[colId]] = 3;
                        // std::cout << "pivot:" << Low[colId] << "," << colId << '\n';
                    }
                    // std::cout << "passend\n";
                }
                else {
                    while (Low[colId] != -1 && look_up[Low[colId]] != -1) {
                        int32_t pivotcolId = look_up[Low[colId]];
                        if (colId < pivotcolId) {
                            stableflag[look_up[Low[colId]]] = 0;
                            look_up[Low[colId]] = colId;
                            stableflag[colId] = 3;
                            stableflag[Low[colId]] = 3;
                            break;
                        }
                        int32_t col1Id = colId;
                        int32_t col2Id = pivotcolId;

                        int32_t* column1 = NULL;
                        int32_t* column2 = NULL;
                        int32_t col1Len = d_tail[col1Id] + 1;
                        int32_t col2Len = d_tail[col2Id] + 1;
                        if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
                        else if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
                        else if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
                        else if (d_rank[col1Id] == 3)column1 = &d_ell3[d_offt[col1Id]];
                        else if (d_rank[col1Id] == 4)column1 = cpuColumn[d_offt[col1Id]].data();

                        if (d_rank[col2Id] == 1)column2 = &d_ell1[d_offt[col2Id]];
                        else if (d_rank[col2Id] == 0)column2 = &d_ell0[d_offt[col2Id]];
                        else if (d_rank[col2Id] == 2)column2 = &d_ell2[d_offt[col2Id]];
                        else if (d_rank[col2Id] == 3)column2 = &d_ell3[d_offt[col2Id]];
                        else if (d_rank[col2Id] == 4)column2 = cpuColumn[d_offt[col2Id]].data();
                        // if (col1Id == 2369149)printf("col1Id:%d col2Id:%d col1Len:%d col2Len:%d Low[colId]:%d Low[col2Id]:%d\n", col1Id, col2Id, col1Len, col2Len, Low[col1Id], Low[col2Id]);
                        temp_col.assign(col1Len + col2Len, 0);
                        auto it = std::set_symmetric_difference(column1, column1 + col1Len, column2, column2 + col2Len, temp_col.begin());
                        temp_col.resize(std::distance(temp_col.begin(), it));

                        if (temp_col.size() > CAP_COL[d_rank[col1Id]]) {
                            d_rank[col1Id]++;
                            while (temp_col.size() > CAP_COL[d_rank[col1Id]])d_rank[col1Id]++;
                            //////////////DEBUG///////////
                            if (d_rank[col1Id] == 1) {
                                if (d_len[d_rank[col1Id]] > cap1) {
                                    // printf("Level 1 array's capbility is not enough\n");
                                    d_rank[col1Id] = 4;cpuColumn.push_back(temp_col);d_offt[col1Id] = cpuColumn.size() - 1;
                                }
                                else d_offt[col1Id] = 1ll * (d_len[d_rank[col1Id]]++) * CAP_COL[d_rank[col1Id]];
                            }
                            else if (d_rank[col1Id] == 2) {
                                if (d_len[d_rank[col1Id]] >= cap2) {
                                    // printf("Level 2 array's capbility is not enough\n");
                                    d_rank[col1Id] = 4;cpuColumn.push_back(temp_col);d_offt[col1Id] = cpuColumn.size() - 1;
                                }
                                else d_offt[col1Id] = 1ll * (d_len[d_rank[col1Id]]++) * CAP_COL[d_rank[col1Id]];
                            }
                            else if (d_rank[col1Id] == 3) {
                                if (d_len[d_rank[col1Id]] >= cap3) {
                                    // printf("Level 2 array's capbility is not enough\n");
                                    d_rank[col1Id] = 4;cpuColumn.push_back(temp_col);d_offt[col1Id] = cpuColumn.size() - 1;
                                }
                                else d_offt[col1Id] = 1ll * (d_len[d_rank[col1Id]]++) * CAP_COL[d_rank[col1Id]];
                            }
                            else if (d_rank[col1Id] == 4) {
                                cpuColumn.push_back(temp_col);
                                d_offt[col1Id] = cpuColumn.size() - 1;
                            }
                            else {
                                printf("$d Level array's level is not enough on cpu\n", d_rank[col1Id]);
                            }
                            if (d_offt[col1Id] < 0)printf("ERROR");
                            //////////////DEBUG///////////
                        }

                        d_tail[col1Id] = temp_col.size() - 1;
                        if (temp_col.size() == 0)stableflag[col1Id] = 3;

                        if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
                        else if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
                        else if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
                        else if (d_rank[col1Id] == 3)column1 = &d_ell3[d_offt[col1Id]];
                        else if (d_rank[col1Id] == 4)column1 = cpuColumn[d_offt[col1Id]].data();

                        Low[col1Id] = (temp_col.size() == 0) ? -1 : temp_col[temp_col.size() - 1];

                        debugmaxlen = std::max(debugmaxlen, int(temp_col.size()));
                        cur_max_len = std::max(cur_max_len, int(temp_col.size()));
                        if (d_rank[col1Id] < 4) {
                            for (int i = 0;i < temp_col.size();i++) {
                                column1[i] = temp_col[i];
                            }
                        }
                        else cpuColumn[d_offt[col1Id]].swap(temp_col);
                    }
                    if (Low[colId] != -1 && look_up[Low[colId]] == -1) {
                        look_up[Low[colId]] = colId;
                        stableflag[colId] = 3;
                        stableflag[Low[colId]] = 3;
                    }
                }

            }

            timer.stopTimer();
            t3 += timer.getElapsedTime();
            for (int col = 0;col < NumCols;++col) {
                if (Low[col] == -1 && d_tail[col] != -1)printf("ERR1 %d\n", col);
                if (Low[col] != -1 && d_tail[col] == -1)printf("ERR2");
                if (Low[col] == -1 || d_tail[col] == -1)continue;
                stableflag[Low[col]] = 3;
                Low[Low[col]] = -1;
                d_tail[Low[col]] = -1;
            }
            printGPUArrayRange(d_len, 0, 3);
            std::cout << cpuColumn.size() << "\n";
            std::cout << "cpu stage done\n";
            std::cout << "cur_max_len:" << cur_max_len << "\n";

        }
        std::cout << "time:" << t1 << "|" << t2 << "|" << t3 << "|" << t4 << "\n";
        std::cout << "maxlen:" << debugmaxlen << "\n";
        printf("----------------------------------------------GPU_Compute completed---------------------------------------------\n");



        return (t1 + t2 + t3 + t4) / 1000.0;
    }
    float GPU_Compute_Naive() {
        bool* Flag = NULL;
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&Flag, sizeof(bool)));

        float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t = 0;
        int32_t DimForStart = Dualize == 1 ? 0 : MaxDim;
        int32_t DimForEnd = Dualize == 1 ? MaxDim + 1 : -1;
        int32_t DimForStep = Dualize == 1 ? 1 : -1;
        for (int32_t cur_dim = DimForStart; cur_dim != DimForEnd; cur_dim += DimForStep) {
            if (DimMaxRow[cur_dim] == -1) continue;
            int32_t ss_size = SSBlockSize;

            int32_t DimCols = DimEnd[cur_dim] - DimBegin[cur_dim];
            int32_t DimRows = DimMaxRow[cur_dim] - DimMinRow[cur_dim];
            int32_t SS_NUM = (DimCols % ss_size == 0) ? (DimCols / ss_size) : (DimCols / ss_size + 1);
            GridSize = (DimCols % BLOCK_SIZE == 0) ? (DimCols / BLOCK_SIZE) : (DimCols / BLOCK_SIZE + 1);
            int32_t MaxPass = SS_NUM + ((DimRows % ss_size == 0) ? (DimRows / ss_size) : (DimRows / ss_size + 1)) - 1;

            for (int32_t cur_pass = 0; cur_pass < MaxPass; cur_pass++) {

                while (1) {
                    Flag[0] = 0;
                    // queue_count[0] = 0;
                    t = measureKernelTime(BlockScanNaive, GridSize, BLOCK_SIZE,
                        DimBegin[cur_dim], DimEnd[cur_dim], DimMinRow[cur_dim], DimMaxRow[cur_dim], cur_pass, ss_size, Low, stableflag, unstable, look_up);
                    t2 += t;
                    t = measureKernelTime(BlockGenAdditionNaive, GridSize, BLOCK_SIZE,
                        Flag, DimBegin[cur_dim], DimEnd[cur_dim], DimMinRow[cur_dim], DimMaxRow[cur_dim], cur_pass, ss_size, Low, stableflag, unstable, look_up, queue_col, QueueSize);

                    t2 += t;
                    if (!Flag[0])break;
                    t = measureKernelTime(ProcColumnPair, QueueSize, 32,
                        d_ell0, d_ell1, d_ell2, d_ell3, d_len, cap1, cap2, cap3, CAP_COL, d_rank, d_offt, d_tail, queue_col, QueueSize, Low, stableflag, look_up, d_aux_arr);

                    t4 += t;
                }
            }

            printGPUArrayRange(d_len, 0, 3);

        }
        std::cout << t1 << " " << t2 << " " << t3 << " " << t4 << "\n";
        printf("----------------------------------------------GPU_Compute completed---------------------------------------------\n");

        return (t1 + t2 + t3 + t4) / 1000.0;
    }
    // #if __cplusplus >= 201703L
    bool out_maxdimclear() {
        std::ofstream outFile("./maxcleardata/" + DataFileName + "_maxclear.txt");
        // 检查文件是否成功打开
        if (!outFile) {
            std::cerr << "无法打开文件" << std::endl;
            return 1;
        }
        for (int cur_col = 1;cur_col < NumCols;cur_col++) {

            outFile << h_dim[line2simplex[cur_col]];
            if (stableflag[line2simplex[cur_col]] == 0) {
                for (int j = 0;j <= d_tail[line2simplex[cur_col]];j++) {
                    outFile << " " << simplex2line[d_ell0[h_offt[line2simplex[cur_col]] + j]] - 1;
                }
            }
            outFile << "\n";
        }
        // 关闭文件
        outFile.close();
        std::cout << "数据已写入 " << "./maxcleardata/" << DataFileName << "_maxclear.txt" << std::endl;
        return 0;
    }
    // #define MY_CPP17_FEATURES
    // #else
    //     // #error "This code requires at least C++17"
    // #endif


    void MemFree() {
        if (h_csc) free(h_csc);
        if (h_offt) free(h_offt);
        if (h_tail) free(h_tail);
        if (h_dim) free(h_dim);

        if (d_ell0) cudaFree(d_ell0);
        if (d_ell1) cudaFree(d_ell1);
        if (d_ell2) cudaFree(d_ell2);
        if (d_ell3) cudaFree(d_ell3);
        if (d_len) cudaFree(d_len);

        if (d_rank) cudaFree(d_rank);
        if (d_offt) cudaFree(d_offt);
        if (d_tail) cudaFree(d_tail);
        if (d_leftone) cudaFree(d_leftone);
        if (d_clearflag) cudaFree(d_clearflag);
        if (d_compressflag) cudaFree(d_compressflag);

        if (Tag) cudaFree(Tag);
        if (Low) cudaFree(Low);

        if (d_aux_arr) cudaFree(d_aux_arr);

        // if (MatA) cudaFree(MatA);
        // if (MatB) cudaFree(MatB);
        // if (MatC) cudaFree(MatC);

        if (CAP_COL) cudaFree(CAP_COL);
        if (stableflag) cudaFree(stableflag);
        if (num_uncols) cudaFree(num_uncols);
        if (unstable) cudaFree(unstable);
        if (look_up) cudaFree(look_up);

        if (contain_vertics) cudaFree(contain_vertics);
        if (contain_vertics_offset) cudaFree(contain_vertics_offset);
        if (binomial_coeff) cudaFree(binomial_coeff);
        if (HashList) cudaFree(HashList);
        if (neighbor_list) cudaFree(neighbor_list);
        if (neighbor_offset) cudaFree(neighbor_offset);
        if (neighbor_len) cudaFree(neighbor_len);
    }
    void setDataFileName(const std::string& newFileName) {
        DataFileName = newFileName;
    }

    std::vector<std::pair<int64_t, int64_t> >& GetPersistencePairs() {
        PersistencePairs.clear();
        for (int64_t i = 0; i < NumCols; ++i) {
            if (look_up[i] != -1) {

                if (!Dualize)PersistencePairs.emplace_back(simplex2line[i] - 1, simplex2line[look_up[i]] - 1);
                else PersistencePairs.emplace_back(simplex2line[NumCols - look_up[i]] - 1, simplex2line[NumCols - i] - 1);
            }
        }
        sort(PersistencePairs.begin(), PersistencePairs.end());
        return PersistencePairs;
    }
};


//middle record
/*
float GPU_Compute_With_Iner_Block_Diff() {
    bool* Flag = NULL;
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&Flag, sizeof(bool)));

    float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t = 0;
    int32_t DimForStart = Dualize == 1 ? 0 : MaxDim;
    int32_t DimForEnd = Dualize == 1 ? MaxDim + 1 : -1;
    int32_t DimForStep = Dualize == 1 ? 1 : -1;
    std::vector<int>blockaddcnt;
    std::vector<float>resultlog;
    for (int32_t cur_dim = DimForStart; cur_dim != DimForEnd; cur_dim += DimForStep) {
        if (DimMaxRow[cur_dim] == -1) continue;
        int32_t ss_size = SSBlockSize;
        int32_t ss_begin = DimBegin[cur_dim], ss_end = std::min(DimBegin[cur_dim] + ss_size, DimEnd[cur_dim]);
        int32_t ss_minrow = DimMaxRow[cur_dim] - ss_size, ss_maxrow = ss_minrow + ss_size;
        std::cout << DimBegin[cur_dim] << " " << DimEnd[cur_dim] << " " << DimMinRow[cur_dim] << " " << DimMaxRow[cur_dim] << "\n";
        std::vector<int>blockbegin;
        std::vector<int>blockend;
        std::vector<int>blockaddcnt;
        blockbegin.push_back(ss_begin);
        blockend.push_back(ss_end);
        blockaddcnt.assign(1, 0);
        while (ss_begin < DimEnd[cur_dim] || ss_maxrow > DimMinRow[cur_dim]) {
            std::cout << DimBegin[cur_dim] + ss_begin << " " << DimBegin[cur_dim] + ss_end << " " << ss_minrow << " " << ss_maxrow << "\n";
            while (1) {
                Flag[0] = 0;
                // queue_count[0] = 0;
                t = measureKernelTime(BlockScan, GridSize, BLOCK_SIZE,
                    ss_begin, ss_end, ss_minrow, Low, stableflag, unstable, look_up);
                t2 += t;
                t = measureKernelTime(BlockGenAddition, GridSize, BLOCK_SIZE,
                    Flag, ss_begin, ss_end, ss_minrow, Low, stableflag, unstable, look_up, queue_col, QueueSize);
                for (int i = 0;i < QueueSize;i++) {
                    if (queue_col[i] == -1)continue;

                    for (int j = 0;j < blockbegin.size();j++) {
                        if (queue_col[i] >= unstable[blockbegin[j]] && queue_col[i] <= unstable[blockend[j] - 1]) {
                            blockaddcnt[j]++;
                            break;
                        }
                    }

                }
                t2 += t;
                if (!Flag[0])break;
                t = measureKernelTime(ProcColumnPair, QueueSize, 32,
                    d_ell0, d_ell1, d_ell2, d_ell3, d_len, cap1, cap2, cap3, CAP_COL, d_rank, d_offt, d_tail, queue_col, QueueSize, Low, stableflag, look_up, d_aux_arr);

                t4 += t;

            }
            int sum = 0;
            for (int j = 0;j < blockbegin.size();j++) {
                sum += blockaddcnt[j];
            }
            std::cout << sum << "\n";
            for (int j = 0;j < blockbegin.size();j++) {
                if (sum == 0)continue;
                if (blockend[j] <= ss_begin)continue;
                resultlog.push_back(blockaddcnt[j] * 1.0 / sum);
            }


            ss_size *= BlockSizeGrowthRate;
            if (ss_maxrow <= DimMinRow[cur_dim]) {
                ss_begin = ss_end;
            }
            else {
                ss_minrow -= ss_size;
                ss_maxrow -= ss_size;
            }
            if (ss_end < DimEnd[cur_dim]) {
                blockbegin.push_back(ss_end);
                ss_end += ss_size;
                ss_end = std::min(ss_end, DimEnd[cur_dim]);
                blockend.push_back(ss_end);

            }
            blockaddcnt.assign(blockbegin.size(), 0);
            GridSize = (ss_end - ss_begin + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }
        std::cout << "block [\n";
        for (int i = 0;i < blockbegin.size();i++) {
            std::cout << blockbegin[i] << ":" << blockend[i] << " ";
        }
        std::cout << "] block\n";
        printGPUArrayRange(d_len, 0, 3);

    }
    std::cout << t1 << " " << t2 << " " << t3 << " " << t4 << "\n";
    printf("----------------------------------------------GPU_Compute completed---------------------------------------------\n");


    std::ofstream outFile("resultlog.txt");
    // 检查文件是否成功打开
    if (!outFile) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    // 将 vector 中的数据写入文件
    for (float value : resultlog) {
        outFile << value << std::endl;
    }
    // 关闭文件
    outFile.close();
    std::cout << "数据已写入 resultlog.txt" << std::endl;

    return (t1 + t2 + t3 + t4) / 1000.0;
}
float GPU_Compute_With_Iner_Block_Diff_Naive() {
    bool* Flag = NULL;
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&Flag, sizeof(bool)));

    float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t = 0;
    int32_t DimForStart = Dualize == 1 ? 0 : MaxDim;
    int32_t DimForEnd = Dualize == 1 ? MaxDim + 1 : -1;
    int32_t DimForStep = Dualize == 1 ? 1 : -1;
    std::vector<int>blockaddcnt;
    std::vector<float>resultlog;
    for (int32_t cur_dim = DimForStart; cur_dim != DimForEnd; cur_dim += DimForStep) {
        if (DimMaxRow[cur_dim] == -1) continue;
        int32_t ss_size = SSBlockSize;

        int32_t DimCols = DimEnd[cur_dim] - DimBegin[cur_dim];
        int32_t DimRows = DimMaxRow[cur_dim] - DimMinRow[cur_dim];
        int32_t SS_NUM = (DimCols % ss_size == 0) ? (DimCols / ss_size) : (DimCols / ss_size + 1);
        GridSize = (DimCols % BLOCK_SIZE == 0) ? (DimCols / BLOCK_SIZE) : (DimCols / BLOCK_SIZE + 1);
        int32_t MaxPass = SS_NUM + ((DimRows % ss_size == 0) ? (DimRows / ss_size) : (DimRows / ss_size + 1)) - 1;
        std::vector<int>blockbegin;
        std::vector<int>blockend;
        std::vector<int>blockaddcnt;
        for (int i = 0;i < SS_NUM;i++) {
            blockbegin.push_back(DimBegin[cur_dim] + i * ss_size);
            blockend.push_back(std::min(DimBegin[cur_dim] + i * ss_size + ss_size, DimEnd[cur_dim]));
        }
        for (int32_t cur_pass = 0; cur_pass < MaxPass; cur_pass++) {
            blockaddcnt.assign(SS_NUM, 0);

            while (1) {
                Flag[0] = 0;
                // queue_count[0] = 0;
                t = measureKernelTime(BlockScanNaive, GridSize, BLOCK_SIZE,
                    DimBegin[cur_dim], DimEnd[cur_dim], DimMinRow[cur_dim], DimMaxRow[cur_dim], cur_pass, ss_size, Low, stableflag, unstable, look_up);
                t2 += t;
                t = measureKernelTime(BlockGenAdditionNaive, GridSize, BLOCK_SIZE,
                    Flag, DimBegin[cur_dim], DimEnd[cur_dim], DimMinRow[cur_dim], DimMaxRow[cur_dim], cur_pass, ss_size, Low, stableflag, unstable, look_up, queue_col, QueueSize);
                int cnt = 0;
                for (int i = 0;i < QueueSize;i++) {
                    if (queue_col[i] == -1)continue;
                    cnt++;
                    for (int j = 0;j < blockbegin.size();j++) {
                        if (queue_col[i] >= unstable[blockbegin[j]] && queue_col[i] <= unstable[blockend[j] - 1]) {
                            blockaddcnt[j]++;
                            break;
                        }
                    }

                }
                // std::cout<<cnt<<"\n";
                t2 += t;
                if (!Flag[0])break;
                t = measureKernelTime(ProcColumnPair, QueueSize, 32,
                    d_ell0, d_ell1, d_ell2, d_ell3, d_len, cap1, cap2, cap3, CAP_COL, d_rank, d_offt, d_tail, queue_col, QueueSize, Low, stableflag, look_up, d_aux_arr);

                t4 += t;
            }
            int sum = 0;
            for (int j = 0;j < blockbegin.size();j++) {
                sum += blockaddcnt[j];
            }
            std::cout << sum << "\n";
            for (int j = 0;j < blockbegin.size();j++) {
                if (sum == 0)continue;
                if (DimMaxRow[cur_dim] - (j + cur_pass) * ss_size - ss_size <= 0)continue;
                resultlog.push_back(blockaddcnt[j] * 1.0 / sum);
            }
        }

        std::cout << "block [\n";
        for (int i = 0;i < blockbegin.size();i++) {
            std::cout << blockbegin[i] << ":" << blockend[i] << " ";
        }
        std::cout << "] block\n";
        printGPUArrayRange(d_len, 0, 3);

    }
    std::cout << t1 << " " << t2 << " " << t3 << " " << t4 << "\n";
    printf("----------------------------------------------GPU_Compute completed---------------------------------------------\n");


    std::ofstream outFile("resultlog.txt");
    // 检查文件是否成功打开
    if (!outFile) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    // 将 vector 中的数据写入文件
    for (float value : resultlog) {
        outFile << value << std::endl;
    }
    // 关闭文件
    outFile.close();
    std::cout << "数据已写入 resultlog.txt" << std::endl;

    return (t1 + t2 + t3 + t4) / 1000.0;
}
float GPU_Compute_With_Intra_Block_Diff() {
    bool* Flag = NULL;
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&Flag, sizeof(bool)));

    float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t = 0;
    int32_t DimForStart = Dualize == 1 ? 0 : MaxDim;
    int32_t DimForEnd = Dualize == 1 ? MaxDim + 1 : -1;
    int32_t DimForStep = Dualize == 1 ? 1 : -1;
    std::vector<int>blockaddcnt;
    std::vector<float>resultlog;
    for (int32_t cur_dim = DimForStart; cur_dim != DimForEnd; cur_dim += DimForStep) {
        if (DimMaxRow[cur_dim] == -1) continue;
        int32_t ss_size = SSBlockSize;
        int32_t ss_begin = DimBegin[cur_dim], ss_end = std::min(DimBegin[cur_dim] + ss_size, DimEnd[cur_dim]);
        int32_t ss_minrow = DimMaxRow[cur_dim] - ss_size, ss_maxrow = ss_minrow + ss_size;
        std::cout << DimBegin[cur_dim] << " " << DimEnd[cur_dim] << " " << DimMinRow[cur_dim] << " " << DimMaxRow[cur_dim] << "\n";
        std::vector<int>blockbegin;
        std::vector<int>blockend;
        std::vector<int>blockaddcnt;
        blockbegin.push_back(ss_begin);
        blockend.push_back(ss_end);
        blockaddcnt.assign(1, 0);
        while (ss_begin < DimEnd[cur_dim] || ss_maxrow > DimMinRow[cur_dim]) {
            std::cout << DimBegin[cur_dim] + ss_begin << " " << DimBegin[cur_dim] + ss_end << " " << ss_minrow << " " << ss_maxrow << "\n";
            while (1) {
                Flag[0] = 0;
                // queue_count[0] = 0;
                t = measureKernelTime(BlockScan, GridSize, BLOCK_SIZE,
                    ss_begin, ss_end, ss_minrow, Low, stableflag, unstable, look_up);
                t2 += t;
                t = measureKernelTime(BlockGenAddition, GridSize, BLOCK_SIZE,
                    Flag, ss_begin, ss_end, ss_minrow, Low, stableflag, unstable, look_up, queue_col, QueueSize);
                int cnt = 0;
                for (int i = 0;i < QueueSize;i++) {
                    if (queue_col[i] == -1)continue;
                    cnt++;
                    for (int j = 0;j < blockbegin.size();j++) {
                        if (queue_col[i] >= unstable[blockbegin[j]] && queue_col[i] <= unstable[blockend[j] - 1]) {
                            blockaddcnt[j]++;
                            break;
                        }
                    }

                }
                for (int j = 0;j < blockbegin.size();j++) {
                    if (cnt == 0)continue;
                    if (blockend[j] <= ss_begin)continue;
                    resultlog.push_back(blockaddcnt[j] * 1.0 / (blockend[j] - blockbegin[j]));
                    blockaddcnt[j] = 0;
                }

                t2 += t;
                if (!Flag[0])break;
                t = measureKernelTime(ProcColumnPair, QueueSize, 32,
                    d_ell0, d_ell1, d_ell2, d_ell3, d_len, cap1, cap2, cap3, CAP_COL, d_rank, d_offt, d_tail, queue_col, QueueSize, Low, stableflag, look_up, d_aux_arr);

                t4 += t;

            }


            ss_size *= BlockSizeGrowthRate;
            if (ss_maxrow <= DimMinRow[cur_dim]) {
                ss_begin = ss_end;
            }
            else {
                ss_minrow -= ss_size;
                ss_maxrow -= ss_size;
            }
            if (ss_end < DimEnd[cur_dim]) {
                blockbegin.push_back(ss_end);
                ss_end += ss_size;
                ss_end = std::min(ss_end, DimEnd[cur_dim]);
                blockend.push_back(ss_end);

            }
            blockaddcnt.assign(blockbegin.size(), 0);
            GridSize = (ss_end - ss_begin + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }
        std::cout << "block [\n";
        for (int i = 0;i < blockbegin.size();i++) {
            std::cout << blockbegin[i] << ":" << blockend[i] << " ";
        }
        std::cout << "] block\n";
        printGPUArrayRange(d_len, 0, 3);

    }
    std::cout << t1 << " " << t2 << " " << t3 << " " << t4 << "\n";
    printf("----------------------------------------------GPU_Compute completed---------------------------------------------\n");


    std::ofstream outFile("resultlog.txt");
    // 检查文件是否成功打开
    if (!outFile) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    // 将 vector 中的数据写入文件
    for (float value : resultlog) {
        if (value == 0)continue;
        outFile << value << std::endl;
    }
    // 关闭文件
    outFile.close();
    std::cout << "数据已写入 resultlog.txt" << std::endl;

    return (t1 + t2 + t3 + t4) / 1000.0;
}
float GPU_Compute_With_Intra_Block_Diff_Naive() {
    bool* Flag = NULL;
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&Flag, sizeof(bool)));

    float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t = 0;
    int32_t DimForStart = Dualize == 1 ? 0 : MaxDim;
    int32_t DimForEnd = Dualize == 1 ? MaxDim + 1 : -1;
    int32_t DimForStep = Dualize == 1 ? 1 : -1;
    std::vector<int>blockaddcnt;
    std::vector<float>resultlog;
    for (int32_t cur_dim = DimForStart; cur_dim != DimForEnd; cur_dim += DimForStep) {
        if (DimMaxRow[cur_dim] == -1) continue;
        int32_t ss_size = SSBlockSize;

        int32_t DimCols = DimEnd[cur_dim] - DimBegin[cur_dim];
        int32_t DimRows = DimMaxRow[cur_dim] - DimMinRow[cur_dim];
        int32_t SS_NUM = (DimCols % ss_size == 0) ? (DimCols / ss_size) : (DimCols / ss_size + 1);
        GridSize = (DimCols % BLOCK_SIZE == 0) ? (DimCols / BLOCK_SIZE) : (DimCols / BLOCK_SIZE + 1);
        int32_t MaxPass = SS_NUM + ((DimRows % ss_size == 0) ? (DimRows / ss_size) : (DimRows / ss_size + 1)) - 1;
        std::vector<int>blockbegin;
        std::vector<int>blockend;
        std::vector<int>blockaddcnt;
        for (int i = 0;i < SS_NUM;i++) {
            blockbegin.push_back(DimBegin[cur_dim] + i * ss_size);
            blockend.push_back(std::min(DimBegin[cur_dim] + i * ss_size + ss_size, DimEnd[cur_dim]));
        }
        for (int32_t cur_pass = 0; cur_pass < MaxPass; cur_pass++) {
            blockaddcnt.assign(SS_NUM, 0);

            while (1) {
                Flag[0] = 0;
                // queue_count[0] = 0;
                t = measureKernelTime(BlockScanNaive, GridSize, BLOCK_SIZE,
                    DimBegin[cur_dim], DimEnd[cur_dim], DimMinRow[cur_dim], DimMaxRow[cur_dim], cur_pass, ss_size, Low, stableflag, unstable, look_up);
                t2 += t;
                t = measureKernelTime(BlockGenAdditionNaive, GridSize, BLOCK_SIZE,
                    Flag, DimBegin[cur_dim], DimEnd[cur_dim], DimMinRow[cur_dim], DimMaxRow[cur_dim], cur_pass, ss_size, Low, stableflag, unstable, look_up, queue_col, QueueSize);
                int cnt = 0;
                for (int i = 0;i < QueueSize;i++) {
                    if (queue_col[i] == -1)continue;
                    cnt++;
                    for (int j = 0;j < blockbegin.size();j++) {
                        if (queue_col[i] >= unstable[blockbegin[j]] && queue_col[i] <= unstable[blockend[j] - 1]) {
                            blockaddcnt[j]++;
                            break;
                        }
                    }
                }
                // int sum = 0;
                // for (int j = 0;j < blockbegin.size();j++) {
                //     sum += blockaddcnt[j];
                // }
                // std::cout << sum << "\n";
                for (int j = 0;j < blockbegin.size();j++) {
                    if (cnt == 0)continue;
                    if (DimMaxRow[cur_dim] - (j + cur_pass) * ss_size - ss_size <= 0)continue;
                    resultlog.push_back(blockaddcnt[j] * 1.0 / ss_size);
                    blockaddcnt[j] = 0;
                }
                // std::cout<<cnt<<"\n";
                t2 += t;
                if (!Flag[0])break;
                t = measureKernelTime(ProcColumnPair, QueueSize, 32,
                    d_ell0, d_ell1, d_ell2, d_ell3, d_len, cap1, cap2, cap3, CAP_COL, d_rank, d_offt, d_tail, queue_col, QueueSize, Low, stableflag, look_up, d_aux_arr);

                t4 += t;
            }

        }

        std::cout << "block [\n";
        for (int i = 0;i < blockbegin.size();i++) {
            std::cout << blockbegin[i] << ":" << blockend[i] << " ";
        }
        std::cout << "] block\n";
        printGPUArrayRange(d_len, 0, 3);

    }
    std::cout << t1 << " " << t2 << " " << t3 << " " << t4 << "\n";
    printf("----------------------------------------------GPU_Compute completed---------------------------------------------\n");


    std::ofstream outFile("resultlog.txt");
    // 检查文件是否成功打开
    if (!outFile) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    // 将 vector 中的数据写入文件
    for (float value : resultlog) {
        if (value == 0)continue;
        outFile << value << std::endl;
    }
    // 关闭文件
    outFile.close();
    std::cout << "数据已写入 resultlog.txt" << std::endl;

    return (t1 + t2 + t3 + t4) / 1000.0;
}
*/
//finish
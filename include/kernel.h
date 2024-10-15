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
#define CHECK_CUDA_ERROR(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while (0)
void CheckCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


struct HashNode {
    int64_t key[MAXKEY];
    int32_t val;
    bool operator < (const HashNode& b)const {
        for (int32_t i = 0; i < MAXKEY; i++) {
            if (key[i] == b.key[i])continue;
            return key[i] < b.key[i];
        }
        return val < b.val;
    }
    bool operator > (const HashNode& b)const {
        for (int32_t i = 0; i < MAXKEY; i++) {
            if (key[i] == b.key[i])continue;
            return key[i] > b.key[i];
        }
        return val < b.val;
    }
};

__device__ void _upper_bound(int32_t* arr, int32_t len, int32_t val, int32_t arr_off, int32_t* pos) {
    int32_t l = 0;
    int32_t r = len - 1;
    int32_t mid;
    *pos = len;
    while (l <= r) {
        mid = (l + r) / 2;
        if (arr[arr_off + mid] > val) {
            *pos = mid;
            r = mid - 1;
        }
        else l = mid + 1;
    }
}
__device__ void _upper_bound(HashNode* arr, int32_t len, HashNode val, int32_t arr_off, int32_t* pos) {
    int32_t l = 0;
    int32_t r = len - 1;
    int32_t mid;
    *pos = len;
    while (l <= r) {
        mid = (l + r) / 2;
        bool greater = 0;
        for (int32_t i = 0; i < MAXKEY; i++) {
            if (arr[arr_off + mid].key[i] == val.key[i])continue;
            if (arr[arr_off + mid].key[i] > val.key[i]) {
                *pos = mid;
                r = mid - 1;
                greater = 1;
                break;
            }
            else break;
        }
        if (!greater) {
            l = mid + 1;
        }
    }
}
__device__ __forceinline__ void ptx_barrier_blocking(const int32_t name, const int32_t num_barriers) {
    asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(num_barriers) : "memory");
}
__device__ __forceinline__ void ptx_barrier_nonblocking(const int32_t name, const int32_t num_barriers) {
    asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(num_barriers) : "memory");
}
__device__ bool checkhashlist(const HashNode* MaxDimHashList, const int32_t* hash_index, const HashNode* target, const int32_t* DimEnd, int32_t col) {
    if (hash_index[0] < 0 || hash_index[0] >= DimEnd[0]) {
        // if(col==1000000){
        printf("ERROR INPUT (hash_index) %d\n", hash_index[0]);
        // }
        // 
        return false;
    }
    else {
        for (int keyid = 0; keyid < MAXKEY; keyid++) {
            if (MaxDimHashList[hash_index[0]].key[keyid] != target->key[keyid]) {
                // printf("ERROR INPUT this key %lld %lld\n", MaxDimHashList[hash_index[0]].key[keyid], target->key[keyid]);
                return false;
            }
        }
        // for (int keyid = 0;keyid < MAXKEY ;keyid++) {
        //     // if (MaxDimHashList[hash_index].key[keyid] != target.key[keyid]) {
        //         printf("ERROR INPUT %lld \n", target.key[keyid]);
        //         // is_birth = 0;
        //     // }
        // }
    }
    return true;
}
__global__ void init_contain_vertics(int32_t* d_ell0, int64_t* d_offt, int32_t* d_tail, int32_t cur_dim, int32_t DimBegin, int32_t DimEnd, int32_t* contain_vertics, int32_t* contain_vertics_offset, int64_t** binomial_coeff, HashNode* HashList, int32_t* neighbor_len, int32_t* neighbor_list, int32_t* neighbor_offset) {
    int32_t col = DimBegin + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= DimEnd)return;

    if (cur_dim >= 1) {

        int32_t len = 0;
        for (int64_t i = d_offt[col]; i <= d_offt[col] + 1; i++) {
            int32_t facet_col = d_ell0[i];
            for (int32_t j = 0; j < cur_dim; j++) {
                int32_t vertice = contain_vertics[contain_vertics_offset[facet_col] + j];
                bool not_exist = true;
                for (int32_t k = 0; k < len && not_exist; k++) {
                    if (contain_vertics[contain_vertics_offset[col] + k] == vertice) {
                        not_exist = false;
                    }
                }
                if (not_exist) {
                    int32_t insert_index = len;
                    for (int32_t k = len - 1; k >= 0; k--) {
                        if (contain_vertics[contain_vertics_offset[col] + k] >= vertice) {
                            insert_index = k;
                        }
                    }
                    for (int32_t k = len; k > insert_index; k--) {
                        contain_vertics[contain_vertics_offset[col] + k] = contain_vertics[contain_vertics_offset[col] + k - 1];
                    }
                    contain_vertics[contain_vertics_offset[col] + insert_index] = vertice;
                    len++;
                }

            }
        }
        int64_t simplex_id;

        int32_t p_begin = 0;
        for (int32_t keyid = 0; keyid < MAXKEY; keyid++) {
            simplex_id = 0;
            int32_t p_k = (cur_dim / MAXKEY);
            if (keyid <= (cur_dim % MAXKEY))p_k++;

            for (int32_t i = 0; i < p_k; i++) {
                simplex_id += binomial_coeff[contain_vertics[contain_vertics_offset[col] + p_begin + i]][i + 1];
            }
            HashList[col].key[keyid] = simplex_id;
            p_begin += p_k;
        }
        HashList[col].val = col;




    }
    else {//0维
        contain_vertics[contain_vertics_offset[col] + 0] = col;
        // simplex_id[col] = col;
        HashList[col].key[0] = col;
        for (int32_t keyid = 1; keyid < MAXKEY; keyid++)
            HashList[col].key[keyid] = 0;
        HashList[col].val = col;
        // int c;
        // for (int i = 0;i < neighbor_len[col];i++) {
        //     c = neighbor_list[neighbor_offset[col] + i];
        // }
        // int d = c;
    }

}

__global__ void SimplexCollapse(int32_t* d_ell0, int64_t* d_offt, int32_t* d_tail, int32_t cur_dim, int32_t DimBegin, int32_t DimEnd, int32_t* stableflag) {
    int32_t col = DimBegin + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= DimEnd)return;
    for (int64_t i = d_offt[col]; !stableflag[col] && i <= d_offt[col] + d_tail[col]; i++) {
        int32_t facet_col = d_ell0[i];
        stableflag[col] |= stableflag[facet_col];
    }
    if (stableflag[col])d_tail[col] = -1;
}



#ifdef MULTIKEY
__global__ void maxdim_clear_init(int32_t nodes, int32_t cur_dim, int32_t DimBegin, int32_t DimEnd, int32_t HighClearMaxNeighbors, int32_t* contain_vertics, int32_t* contain_vertics_offset, int64_t** binomial_coeff, HashNode* MaxDimHashList, int32_t* neighbor_len, int32_t* neighbor_list, int32_t* neighbor_offset, int32_t* d_clearflag, int32_t* stableflag, int32_t* Tag) {
    int32_t col = DimBegin + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= DimEnd || cur_dim == 0)return;
    bool endflag = false;
    int32_t vertice_ptr[MAXDIM + 1];
    int32_t high_vertics[MAXDIM + 2];
    int64_t simplex_id[MAXKEY];

    for (int32_t i = 0; i <= cur_dim; i++)vertice_ptr[i] = neighbor_len[contain_vertics[contain_vertics_offset[col] + i]] - 1;
    int itercnt = 0;
    while (!endflag && vertice_ptr[0] >= 0 && itercnt < HighClearMaxNeighbors) {
        int32_t extra_vertice = neighbor_list[neighbor_offset[contain_vertics[contain_vertics_offset[col] + 0]] + vertice_ptr[0]];
        bool not_find_flag = 0;
        for (int32_t i = 1; i <= cur_dim && !endflag && !not_find_flag; i++) {
            while (vertice_ptr[i] >= 0 && neighbor_list[neighbor_offset[contain_vertics[contain_vertics_offset[col] + i]] + vertice_ptr[i]] > extra_vertice) {
                vertice_ptr[i]--;
            }

            if (vertice_ptr[i] < 0) {
                endflag = true;
            }
            else if (neighbor_list[neighbor_offset[contain_vertics[contain_vertics_offset[col] + i]] + vertice_ptr[i]] != extra_vertice) {
                not_find_flag = 1;
            }
        }
        if (!not_find_flag && !endflag) {

            int32_t index = 0;
            for (int32_t i = 0; i <= cur_dim; i++) {
                if (extra_vertice <= contain_vertics[contain_vertics_offset[col] + i]) {
                    high_vertics[index] = extra_vertice;
                    // simplex_id[index % MAXKEY] += binomial_coeff[extra_vertice][index / MAXKEY + 1];
                    index++;
                    while (i <= cur_dim) {
                        high_vertics[index] = contain_vertics[contain_vertics_offset[col] + i];
                        i++;
                        // simplex_id[index % MAXKEY] += binomial_coeff[contain_vertics[contain_vertics_offset[col] + i++]][index / MAXKEY + 1];
                        index++;
                    }

                }
                else {
                    high_vertics[index] = contain_vertics[contain_vertics_offset[col] + i];
                    // simplex_id[index % MAXKEY] += binomial_coeff[contain_vertics[contain_vertics_offset[col] + i]][index / MAXKEY + 1];
                    index++;
                }
            }
            if (index <= cur_dim + 1) {
                high_vertics[index] = extra_vertice;
                // simplex_id[index % MAXKEY] += binomial_coeff[extra_vertice][index / MAXKEY + 1];
                index++;
            }
            bool is_birth = 1;
            HashNode target;

            for (int32_t del_id = 0;del_id <= cur_dim + 1 && is_birth;del_id++) {
                for (int32_t keyid = 0, p_begin = 0; keyid < MAXKEY; keyid++) {
                    simplex_id[keyid] = 0;
                    int32_t p_k = (cur_dim / MAXKEY);
                    if (keyid <= (cur_dim % MAXKEY))p_k++;
                    for (int32_t i = 0; i < p_k; i++) {
                        if (p_begin + i == del_id)p_begin++;
                        simplex_id[keyid] += binomial_coeff[high_vertics[p_begin + i]][i + 1];
                    }
                    target.key[keyid] = simplex_id[keyid];
                    p_begin += p_k;
                }
                int32_t hash_index;
                _upper_bound(MaxDimHashList, DimEnd - DimBegin, target, 0, &hash_index);
                hash_index--;
                //BUG CHECK
                if (checkhashlist(MaxDimHashList, &hash_index, &target, &DimEnd, col)) {
                    if (MaxDimHashList[hash_index].val > col) {
                        is_birth = 0;
                    }
                }
                else {
                    is_birth = 0;
                }
            }
            if (is_birth) {
                d_clearflag[col] = 1;
                stableflag[col] = 3;
                Tag[col] = 3;
            }
        }
        vertice_ptr[0]--;
        itercnt++;
    }
}
#else
__global__ void maxdim_clear_init(int32_t nodes, int32_t cur_dim, int32_t DimBegin, int32_t DimEnd, int32_t HighClearMaxNeighbors, int32_t* contain_vertics, int32_t* contain_vertics_offset, int64_t** binomial_coeff, HashNode* MaxDimHashList, int32_t* neighbor_len, int32_t* neighbor_list, int32_t* neighbor_offset, int32_t* d_clearflag, int32_t* stableflag, int32_t* Tag) {
    int32_t col = DimBegin + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= DimEnd || cur_dim == 0)return;
    bool endflag = false;
    int32_t vertice_ptr[MAXDIM + 1];
    int32_t high_vertics[MAXDIM + 2];
    int64_t simplex_id[MAXKEY];

    for (int32_t i = 0; i <= cur_dim; i++)vertice_ptr[i] = neighbor_len[contain_vertics[contain_vertics_offset[col] + i]] - 1;
    int itercnt = 0;
    while (!endflag && vertice_ptr[0] >= 0 && itercnt < HighClearMaxNeighbors) {
        int32_t extra_vertice = neighbor_list[neighbor_offset[contain_vertics[contain_vertics_offset[col] + 0]] + vertice_ptr[0]];
        bool not_find_flag = 0;
        for (int32_t i = 1; i <= cur_dim && !endflag && !not_find_flag; i++) {
            while (vertice_ptr[i] >= 0 && neighbor_list[neighbor_offset[contain_vertics[contain_vertics_offset[col] + i]] + vertice_ptr[i]] < extra_vertice) {
                vertice_ptr[i]--;
            }

            if (vertice_ptr[i] < 0) {
                endflag = true;
            }
            else if (neighbor_list[neighbor_offset[contain_vertics[contain_vertics_offset[col] + i]] + vertice_ptr[i]] != extra_vertice) {
                not_find_flag = 1;
            }
        }
        if (!not_find_flag && !endflag) {

            int32_t index = 0;
            for (int32_t i = 0; i <= cur_dim; i++) {
                if (extra_vertice <= contain_vertics[contain_vertics_offset[col] + i]) {
                    high_vertics[index] = extra_vertice;
                    simplex_id[index % MAXKEY] += binomial_coeff[extra_vertice][index / MAXKEY + 1];
                    index++;
                    while (i <= cur_dim) {
                        high_vertics[index] = contain_vertics[contain_vertics_offset[col] + i];
                        simplex_id[index % MAXKEY] += binomial_coeff[contain_vertics[contain_vertics_offset[col] + i++]][index / MAXKEY + 1];
                        index++;
                    }

                }
                else {
                    high_vertics[index] = contain_vertics[contain_vertics_offset[col] + i];
                    simplex_id[index % MAXKEY] += binomial_coeff[contain_vertics[contain_vertics_offset[col] + i]][index / MAXKEY + 1];
                    index++;
                }
            }
            if (index <= cur_dim + 1) {
                high_vertics[index] = extra_vertice;
                simplex_id[index % MAXKEY] += binomial_coeff[extra_vertice][index / MAXKEY + 1];
                index++;
            }
            bool is_birth = 1;
            HashNode target;
            for (int32_t keyid = 0, p_begin = 0; keyid < MAXKEY; keyid++) {
                simplex_id[keyid] = 0;
                int32_t p_k = ((cur_dim + 1) / MAXKEY);
                if (keyid <= ((cur_dim + 1) % MAXKEY))p_k++;
                for (int32_t i = 0; i < p_k; i++) {
                    simplex_id[keyid] += binomial_coeff[high_vertics[p_begin + i]][i + 1];
                }
                target.key[keyid] = simplex_id[keyid];
                p_begin += p_k;
            }
            int64_t idx_above = 0, idx_below = 0, idx_extra = 0;
            int32_t key_mid, key_mid_l = 0, key_mid_r = 0;
            for (int32_t keyid = 0; keyid < MAXKEY; keyid++) {
                if (keyid <= ((cur_dim + 1) % MAXKEY)) {
                    key_mid = keyid;
                    key_mid_l = key_mid_r;
                    key_mid_r += (cur_dim + 1) / MAXKEY + 1;
                }
            }

            if (key_mid + 1 < MAXKEY) {
                target.key[key_mid] = simplex_id[key_mid] - binomial_coeff[high_vertics[key_mid_r - 1]][key_mid_r - key_mid_l];
                idx_extra = binomial_coeff[high_vertics[key_mid_r - 1]][1];
                for (int32_t keyid = key_mid + 1, p_begin = key_mid_r; keyid < MAXKEY && is_birth; keyid++) {
                    int32_t p_k = ((cur_dim + 1) / MAXKEY);
                    idx_below = 0;
                    idx_above = simplex_id[keyid];
                    for (int32_t i = 0; i < p_k; i++) {
                        target.key[keyid] = idx_above - binomial_coeff[high_vertics[i + p_begin]][i + 1] + idx_below + idx_extra;
                        idx_below += binomial_coeff[high_vertics[i + p_begin]][i + 2];
                        idx_above -= binomial_coeff[high_vertics[i + p_begin]][i + 1];


                        int32_t hash_index;
                        _upper_bound(MaxDimHashList, DimEnd - DimBegin, target, 0, &hash_index);
                        hash_index--;
                        if (checkhashlist(MaxDimHashList, &hash_index, &target, &DimEnd, col)) {
                            if (MaxDimHashList[hash_index].val > col) {
                                is_birth = 0;
                            }
                        }
                        else {
                            is_birth = 0;
                        }
                    }
                    p_begin += p_k;

                    idx_extra = binomial_coeff[high_vertics[p_begin + p_k - 1]][1];
                }
                for (int32_t keyid = key_mid; keyid < MAXKEY; keyid++) {
                    target.key[keyid] = simplex_id[keyid];
                }
            }



            idx_extra = 0;
            for (int32_t keyid = key_mid, p_rbegin = key_mid_r - 1; keyid >= 0 && is_birth; keyid--) {
                int32_t p_k = (cur_dim + 1) / MAXKEY + 1;
                idx_below = simplex_id[keyid];
                idx_above = 0;
                for (int32_t i = 0; i < p_k && is_birth; i++) {
                    // printf("cut v:%d\n", high_vertics[i]);
                    target.key[keyid] = idx_above - binomial_coeff[high_vertics[p_rbegin - i]][p_k - i] + idx_below + idx_extra;
                    idx_below -= binomial_coeff[high_vertics[p_rbegin - i]][p_k - i];
                    idx_above += binomial_coeff[high_vertics[p_rbegin - i]][p_k - i - 1];
                    int32_t hash_index;

                    _upper_bound(MaxDimHashList, DimEnd - DimBegin, target, 0, &hash_index);
                    hash_index--;
                    //BUG CHECK
                    if (checkhashlist(MaxDimHashList, &hash_index, &target, &DimEnd, col)) {
                        if (MaxDimHashList[hash_index].val > col) {
                            is_birth = 0;
                        }
                    }
                    else {
                        is_birth = 0;
                    }
                }
                idx_extra = binomial_coeff[high_vertics[p_rbegin]][p_k];
                p_rbegin -= p_k;
            }


            if (is_birth) {
                d_clearflag[col] = 1;
                stableflag[col] = 3;
                Tag[col] = 3;
            }
        }
        vertice_ptr[0]--;
        itercnt++;
    }
}
#endif

__global__ void initleft_kernel(int32_t* CAP_COL, int32_t* d_ell0, int32_t* d_rank, int64_t* d_offt, int32_t* d_tail, int32_t* Low, int32_t* leftone, int32_t* d_clearflag, int32_t* look_up, int32_t num_cols) {
    // int32_t bid=blockIdx.x;
    // int32_t tid=threadIdx.x;
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_cols) {
        d_rank[col] = 0;

        Low[col] = (d_tail[col] == -1 ? -1 : d_ell0[d_offt[col] + d_tail[col]]);
        for (int32_t i = 0; i <= d_tail[col]; i++) {
            atomicMin((unsigned int32_t*)(leftone + d_ell0[d_offt[col] + i]), (unsigned int32_t)col);
        }

    }
}

__global__ void initlookup_kernel(int32_t* Tag, int32_t* CAP_COL, int32_t* d_ell0, int64_t* d_offt, int32_t* d_tail, int32_t* Low, int32_t* leftone, int32_t* d_clearflag, int32_t* compressflag, int32_t* stableflag, int32_t* look_up, int32_t num_cols) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_cols) {
        // int32_t coloff=col*CAP_COL[1];
        int32_t low = Low[col];//(d_tail[col]==-1)?-1:d_ell1[coloff+d_tail[col]];
        if (d_tail[col] == -1)stableflag[col] = 3, Tag[col] = 3;
        if (low != -1)d_clearflag[low] = 1, stableflag[low] = 3, Tag[low] = 3;
        if (low != -1 && leftone[low] == col) {//zhuyuan
            look_up[low] = col;
            stableflag[col] = 3;
            //look_up[col]=-2;//di col liei keyi yasuo
            compressflag[col] = 1;
            Tag[col] = 1;
        }
        for (int32_t i = 0; i <= d_tail[col] && look_up[col] != -2; i++) {
            if (leftone[d_ell0[d_offt[col] + i]] == col) {
                //look_up[col]=-2;
                compressflag[col] = 1;
            }
        }

    }
}

__global__ void clearing_kernel(int32_t* d_tail, int32_t* Low, int32_t* d_clearflag, int32_t num_cols, int32_t* look_up) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_cols && d_clearflag[col] == 1) {
        int32_t lowest = Low[col];
        if (lowest != -1 && look_up[lowest] != col) {
            d_tail[col] = -1;
            Low[col] = -1;
        }
        else if (lowest != -1 && look_up[lowest] == col)printf("DATA ERROR look_up[%d] == %d\n", lowest, col);
    }
}

__global__ void compress_kernel(int32_t* Tag, int32_t* CAP_COL, int32_t* d_ell0, int64_t* d_offt, int32_t* d_tail, int32_t* Low, int32_t* compressflag, int32_t* look_up, int32_t num_cols) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_cols) {
        //printf("tail[%d]:%d\n",col,d_tail[col]);
        int64_t coloff = d_offt[col];

        for (int32_t i = 0; i <= d_tail[col]; i++) {
            int32_t row = d_ell0[coloff + i];
            if (compressflag[row] == 1) {
                d_ell0[coloff + i] = -1;
            }
        }
        int32_t len = d_tail[col];
        int32_t cnt = -1;
        for (int32_t i = 0; i <= len; i++) {
            if (d_ell0[coloff + i] != -1) {
                d_ell0[coloff + (++cnt)] = d_ell0[coloff + i];
            }
        }
        d_tail[col] = cnt;
        if (cnt == -1)Tag[col] = 3, Low[col] = -1;
        else Low[col] = d_ell0[coloff + cnt];
        //printf("tail[%d]:%d\n",col,d_tail[col]);
        // d_offt[col]=col*CAP_COL[1];
    }
}
__global__ void countNonZeros(int32_t* d_ell0, int32_t* d_tail, int64_t* d_offt, int32_t* d_col_counts, int32_t n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        int startIdx = d_offt[col];
        int endIdx = startIdx + d_tail[col] + 1;

        for (int i = startIdx; i < endIdx; ++i) {
            int rowIndex = d_ell0[i];
            int newCol = n - rowIndex;
            atomicAdd(&d_col_counts[newCol], 1);
        }
    }
}
__global__ void computeOffsets(int32_t* d_col_counts, int64_t* d_new_offt, int32_t n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        d_new_offt[col] = d_col_counts[col];
    }
}
__global__ void dualizeMatrixKernel(int32_t* d_ell0, int32_t* d_tail, int64_t* d_offt,
    int32_t* d_new_ell0, int32_t* d_col_counts, int64_t* d_new_offt, int32_t n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        int startIdx = d_offt[col];
        int endIdx = startIdx + d_tail[col] + 1;

        for (int i = startIdx; i < endIdx; ++i) {
            int rowIndex = d_ell0[i];
            int newRow = n - col;
            int newCol = n - rowIndex;

            int pos = atomicAdd(&d_col_counts[newCol], 1);

            d_new_ell0[d_new_offt[newCol] + pos] = newRow;
            // printf("oldc%d oldr%d col%d  row%d [%d+%lld]= %d\n", col, rowIndex, newCol, newRow, pos, d_new_offt[newCol], newRow);
        }
    }
}
__global__ void dualizeAuxArray(int32_t* stableflag, int32_t* look_up,
    int32_t* new_stableflag, int32_t* new_look_up,
    int32_t* d_ell0, int64_t* d_offt, int32_t* Low, int32_t* d_tail, int32_t n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        int rowIndex = col;
        int colIndex = look_up[col];
        int newRow = n - colIndex;
        int newCol = n - rowIndex;
        if (colIndex != -1) {
            new_look_up[newRow] = newCol;
            new_stableflag[newCol] = 3;
        }
        if (d_tail[col] == -1) {
            new_stableflag[col] = 3;
            Low[col] = -1;
        }
        else {
            Low[col] = d_ell0[d_offt[col] + d_tail[col]];
        }
    }
}

//计算
__global__ void BlockScan(int32_t DimBegin, int32_t DimEnd, int32_t MinRow, int32_t PhaseFlag,
    int32_t* d_rank, int32_t* Low, int32_t* stableflag, int32_t* d_unstable, int32_t* look_up) {
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t unId = DimBegin + (bid << BLOCK_SIZE_BIT) + tid;
    if (unId >= DimEnd)return;
    int32_t colId = d_unstable[unId];
    if (stableflag[colId] >= PhaseFlag) return;
    if (Low[colId] < MinRow)return;
    // if(d_rank[colId]>2&&!PhaseFlag)return ;
    int32_t lowbit = Low[colId];
    int32_t pivotcolId = look_up[lowbit];
    if (pivotcolId != -1) {
        if (colId < pivotcolId) {
            int32_t oldPivotcolId = atomicMin(look_up + lowbit, colId);
            stableflag[oldPivotcolId] = 0;
        }
    }
    else {
        int32_t tryset = atomicExch(look_up + lowbit, colId);
        if (tryset != -1 && tryset < colId) {
            atomicMin(look_up + lowbit, tryset);
        }
    }
}
__global__ void BlockGenAddition(bool* Flag, int32_t DimBegin, int32_t DimEnd, int32_t MinRow, int32_t PhaseFlag,
    int32_t* d_rank, int32_t* Low, int32_t* stableflag, int32_t* d_unstable, int32_t* look_up,
    int32_t* queue_col, int32_t QueueSize) {
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t gid = (bid << BLOCK_SIZE_BIT) + tid;
    int32_t unId = DimBegin + (bid << BLOCK_SIZE_BIT) + tid;
    if (unId >= DimEnd)return;
    int32_t colId = d_unstable[unId];
    if (stableflag[colId] >= PhaseFlag)return;
    if (Low[colId] < MinRow)return;
    // if (d_rank[colId] > 2 && !PhaseFlag)return;
    int32_t lowbit = Low[colId];
    int32_t pivotcolId = look_up[lowbit];
    int32_t index = gid % QueueSize;
    if (pivotcolId != -1) {
        if (colId == pivotcolId) {
            stableflag[lowbit] = 3;//clearing
            stableflag[colId] = 3;
        }
        else {
            Flag[0] = 1;
            queue_col[index] = colId;
        }
    }
}

__global__ void BlockScanNaive(int32_t DimBegin, int32_t DimEnd, int32_t DimMinRow, int32_t DimMaxRow, int32_t Pass, int32_t SS_SIZE,
    int32_t* Low, int32_t* stableflag, int32_t* d_unstable, int32_t* look_up) {
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t unId = DimBegin + (bid << BLOCK_SIZE_BIT) + tid;
    if (unId >= DimEnd)return;
    int32_t colId = d_unstable[unId];
    if (stableflag[colId])return;
    int32_t ssid = ((bid << BLOCK_SIZE_BIT) + tid) / SS_SIZE;
    if (DimMaxRow - Pass * SS_SIZE + ssid * SS_SIZE < 0)return;
    if (Low[colId] < DimMaxRow - Pass * SS_SIZE + ssid * SS_SIZE - SS_SIZE)return;
    int32_t lowbit = Low[colId];
    int32_t pivotcolId = look_up[lowbit];
    if (pivotcolId != -1) {
        if (colId < pivotcolId) {
            int32_t oldPivotcolId = atomicMin(look_up + lowbit, colId);
            stableflag[oldPivotcolId] = 0;
        }
    }
    else {
        int32_t tryset = atomicExch(look_up + lowbit, colId);
        if (tryset != -1 && tryset < colId) {
            atomicMin(look_up + lowbit, tryset);
        }
    }
}
__global__ void BlockGenAdditionNaive(bool* Flag, int32_t DimBegin, int32_t DimEnd, int32_t DimMinRow, int32_t DimMaxRow, int32_t Pass, int32_t SS_SIZE,
    int32_t* Low, int32_t* stableflag, int32_t* d_unstable, int32_t* look_up,
    int32_t* queue_col, int32_t QueueSize) {
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t gid = (bid << BLOCK_SIZE_BIT) + tid;
    int32_t unId = DimBegin + (bid << BLOCK_SIZE_BIT) + tid;
    if (unId >= DimEnd)return;
    int32_t colId = d_unstable[unId];
    if (stableflag[colId])return;
    int32_t ssid = ((bid << BLOCK_SIZE_BIT) + tid) / SS_SIZE;
    if (DimMaxRow - Pass * SS_SIZE + ssid * SS_SIZE < 0)return;
    if (Low[colId] < DimMaxRow - Pass * SS_SIZE + ssid * SS_SIZE - SS_SIZE)return;
    // printf("%d\n",ssid);
    int32_t lowbit = Low[colId];
    int32_t pivotcolId = look_up[lowbit];
    int32_t index = gid % QueueSize;
    if (pivotcolId != -1) {
        if (colId == pivotcolId) {
            stableflag[lowbit] = 1;//clearing
            stableflag[colId] = 1;
        }
        else {
            Flag[0] = 1;
            queue_col[index] = colId;
        }
    }
}


__device__ void cal_rank_warp(int32_t* aux_arr, int32_t* column1, int32_t col1Len, int32_t* column2, int32_t col2Len, int32_t lid, int32_t* null_nums) {
    int32_t pos;
    int32_t t;
    for (int32_t i = lid; i < col1Len; i += 32) {
        _upper_bound(column2, col2Len, column1[i], 0, &pos);
        // printf("%d %d:%d+%d\n",lid,column1[i],pos,i);
        atomicXor(aux_arr + i + pos, column1[i]);
    }
    __syncwarp();
    for (int32_t i = lid; i < col2Len; i += 32) {
        _upper_bound(column1, col1Len, column2[i], 0, &pos);
        // printf("%d %d:%d+%d\n",lid,column2[i],pos,i);
        // t = 
        atomicXor(aux_arr + i + pos, column2[i]);
        // if (t != -1)atomicExch(aux_arr + i + pos, -1);
    }
    __syncwarp();
    null_nums[lid] = 0;
    int32_t partLen = (col2Len + col1Len) / 32 + 1;
    int32_t partBegin = lid * partLen;
    int32_t partEnd = min(lid * partLen + partLen, col2Len + col1Len);
    for (int32_t i = partBegin; i < partEnd; i++) {
        atomicAdd(null_nums + lid, !aux_arr[i]);
    }
    // printf("%d %d %d:%d\n", partBegin, partEnd, lid, null_nums[lid]);
    // if(lid==0){
    //     for(int i=0;i<col2Len + col1Len;i++){
    //         printf("[%d]",aux_arr[i]);
    //     }
    // }
    __syncwarp();
    int32_t value = null_nums[lid];
    for (int offset = 1; offset < 32; offset <<= 1) {
        int32_t temp = __shfl_up_sync(0xffffffff, value, offset, 32);
        if (lid >= offset) value += temp;
    }
    null_nums[lid] = value;
    __syncwarp();
}
__device__ void arr_compact_warp(int32_t* column, int32_t* aux_arr, int32_t merge_len, int32_t* null_nums, int32_t lid) {
    int32_t partLen = merge_len / 32 + 1;
    int32_t partBegin = lid * partLen;
    int32_t partEnd = min(lid * partLen + partLen, merge_len);
    int32_t null_cnt = null_nums[lid];
    for (int32_t i = partEnd - 1; i >= partBegin; i--) {
        if (aux_arr[i]) {
            column[i - null_cnt] = aux_arr[i];
        }
        else null_cnt--;
        aux_arr[i] = 0;
    }
    __syncwarp();
}
__device__ void cal_rank_block(int32_t* aux_arr, int32_t* column1, int32_t col1Len, int32_t* column2, int32_t col2Len, int32_t tid, int32_t* null_nums) {
    int32_t pos;
    int32_t t;
    for (int32_t i = tid; i < col1Len; i += blockDim.x) {
        _upper_bound(column2, col2Len, column1[i], 0, &pos);
        atomicXor(aux_arr + i + pos, column1[i]);
    }
    __syncthreads();
    for (int32_t i = tid; i < col2Len; i += blockDim.x) {
        _upper_bound(column1, col1Len, column2[i], 0, &pos);
        // t = 
        atomicXor(aux_arr + i + pos, column2[i]);
        // if (t != -1)atomicExch(aux_arr + i + pos, -1);
    }
    __syncthreads();
    null_nums[tid] = 0;
    int32_t partLen = (col2Len + col1Len) / blockDim.x + 1;
    int32_t partBegin = tid * partLen;
    int32_t partEnd = min(tid * partLen + partLen, col2Len + col1Len);
    for (int32_t i = partBegin; i < partEnd; i++) {
        atomicAdd(null_nums + tid, !aux_arr[i]);
    }
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int32_t temp;
        if (tid >= offset) {
            temp = null_nums[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            null_nums[tid] += temp;
        }
        __syncthreads();
    }
}
__device__ void arr_compact_block(int32_t* column, int32_t* aux_arr, int32_t merge_len, int32_t* null_nums, int32_t tid) {
    int32_t partLen = merge_len / blockDim.x + 1;
    int32_t partBegin = tid * partLen;
    int32_t partEnd = min(tid * partLen + partLen, merge_len);
    int32_t null_cnt = null_nums[tid];
    for (int32_t i = partEnd - 1; i >= partBegin; i--) {
        if (aux_arr[i]) {
            column[i - null_cnt] = aux_arr[i];
        }
        else null_cnt--;
        aux_arr[i] = 0;
    }
    __syncthreads();
}
__device__ void clear_aux_arr(int32_t* column, int32_t* aux_arr, int32_t merge_len, int32_t* null_nums, int32_t lid) {
    int32_t partLen = merge_len / 32 + 1;
    int32_t partBegin = lid * partLen;
    int32_t partEnd = min(lid * partLen + partLen, merge_len);
    int32_t null_cnt = null_nums[lid];
    for (int32_t i = partEnd - 1; i >= partBegin; i--) {
        aux_arr[i] = 0;
    }
    __syncwarp();
}
__device__ void clear_aux_arr_block(int32_t* column, int32_t* aux_arr, int32_t merge_len, int32_t* null_nums, int32_t tid) {
    int32_t partLen = merge_len / blockDim.x + 1;
    int32_t partBegin = tid * partLen;
    int32_t partEnd = min(tid * partLen + partLen, merge_len);
    int32_t null_cnt = null_nums[tid];
    for (int32_t i = partEnd - 1; i >= partBegin; i--) {
        aux_arr[i] = 0;
    }
    __syncthreads();
}

__global__ void ProcColumnPair(
    int32_t* d_ell0, int32_t* d_ell1, int32_t* d_ell2, int32_t* d_ell3, int32_t* d_len, int32_t cap1, int32_t cap2, int32_t cap3, int32_t* CAP_COL,
    int32_t* d_rank, int64_t* d_offt, int32_t* d_tail,
    int32_t* queue_col, int32_t QueueSize,
    int32_t* Low, int32_t* stableflag, int32_t* look_up,
    int32_t* aux_arr
) {
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t gid = bid * blockDim.x + tid;
    int32_t gwid = gid / 32;
    // if (queue_col[gwid] == -1)return;
    int32_t wid = tid / 32;
    int32_t lid = tid % 32;
    // __shared__ int32_t CompetitiveSuccess[1];
    __shared__ int32_t NullNums[32];
    __shared__ int32_t* AuxArr[1];
    __shared__ bool OOM;
    if (lid == 0)AuxArr[wid] = aux_arr + (gwid << WARP_EXTRA_SIZE_BIT);

    // for (int i = gwid % QUEUE_SIZE, j = 0;j < QUEUE_SIZE && queue_count[0];j++, i = (i + 1) % QUEUE_SIZE) {
    for (int i = bid;i < QueueSize /*&& i < queue_count[0]*/; i += gridDim.x) {
        __syncwarp();
        if (queue_col[i] == -1)continue;
        OOM = false;
        int32_t col1Id = queue_col[i];
        int32_t col2Id = look_up[Low[col1Id]];//queue_add[i];
        // if(lid==0)printf("%d+%d\n",col1Id,col2Id);
        int32_t* column1 = NULL;
        int32_t* column2 = NULL;
        int32_t col1Len = d_tail[col1Id] + 1;
        int32_t col2Len = d_tail[col2Id] + 1;
        
        
        __syncwarp();
        if (d_rank[col2Id] == 3||d_rank[col1Id] == 3) {
            // if (lid == 0) printf("skip\n");
            // printf("skip\n");
            queue_col[i] = -1;
            stableflag[col1Id] = 1;//1 indicates that this column will not be processed at the moment.
            continue;
        }

        if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
        if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
        if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
        if (d_rank[col1Id] == 3)printf("ERROR1\n");
        if (d_rank[col2Id] == 1)column2 = &d_ell1[d_offt[col2Id]];
        if (d_rank[col2Id] == 0)column2 = &d_ell0[d_offt[col2Id]];
        if (d_rank[col2Id] == 2)column2 = &d_ell2[d_offt[col2Id]];
        if (d_rank[col2Id] == 3)printf("ERROR2\n");
        
        cal_rank_warp(AuxArr[wid], column1, col1Len, column2, col2Len, lid, NullNums + (wid << WARP_SIZE_BIT));
        int32_t resultLen = col1Len + col2Len - NullNums[(wid << WARP_SIZE_BIT) + 31];

        if (lid == 0 && resultLen > d_tail[col1Id] + 1) {
            if (resultLen > CAP_COL[d_rank[col1Id]]) {

                int32_t D_Rank = d_rank[col1Id];
                D_Rank++;
                while (resultLen > CAP_COL[D_Rank])D_Rank++;
                //////////////DEBUG///////////
                if (D_Rank == 1) {
                    int32_t t_number = atomicAdd(d_len + D_Rank, 1);
                    if (t_number > cap1) {
                        printf("Level 1 array's capbility is not enough\n");
                        OOM = true;
                    }
                    else {
                        d_offt[col1Id] = 1ll * t_number * CAP_COL[D_Rank];
                        d_rank[col1Id] = D_Rank;
                    }
                }
                else if (D_Rank == 2) {
                    int32_t t_number = atomicAdd(d_len + D_Rank, 1);
                    if (t_number >= cap2) {
                        printf("Level 2 array's capbility is not enough\n");
                        OOM = true;
                    }
                    else {
                        d_offt[col1Id] = 1ll * t_number * CAP_COL[D_Rank];
                        d_rank[col1Id] = D_Rank;
                    }
                }
                // else if (d_rank[col1Id] == 3 && CAP_COL[3] > 0) {
                //     if (d_len[d_rank[col1Id]] >= cap3) {
                //         printf("Level 3 array's capbility is not enough\n");
                //     }
                // }
                else {
                    OOM = true;
                    printf("Len:%d Level array's level is not enough on GPU P1\n", resultLen);
                }
                if (d_offt[col1Id] < 0)printf("ERROR");
                //////////////DEBUG///////////
            }
        }
        __syncwarp();
        if (OOM) {
            // if (lid == 0) printf("skip\n");
            // printf("skip\n");
            queue_col[i] = -1;
            stableflag[col1Id] = 1;//1 indicates that this column will not be processed at the moment.
            clear_aux_arr(column1, AuxArr[wid], col1Len + col2Len, NullNums + (wid << WARP_SIZE_BIT), lid);
            continue;
        }
        d_tail[col1Id] = resultLen - 1;
        stableflag[col1Id] = 3 * int(!resultLen);
        if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
        if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
        if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
        // if (d_rank[col1Id] == 3)printf("ERR\n");
        //arr
        arr_compact_warp(column1, AuxArr[wid], col1Len + col2Len, NullNums + (wid << WARP_SIZE_BIT), lid);
        Low[col1Id] = (resultLen == 0) ? -1 : column1[resultLen - 1];
        if (lid == 0) {
            queue_col[i] = -1;
            // queue_tag[i] = 0;
            // inqueue[col1Id] = 0;
            // atomicSub(queue_count, 1);
        }
    }
}


__global__ void ProcColumnPair_121Block(
    int32_t* d_ell0, int32_t* d_ell1, int32_t* d_ell2, int32_t* d_ell3, int32_t* d_len, int32_t cap1, int32_t cap2, int32_t cap3, int32_t* CAP_COL,
    int32_t* d_rank, int64_t* d_offt, int32_t* d_tail,
    int32_t* queue_col, int32_t QueueSize,
    int32_t* Low, int32_t* stableflag, int32_t* look_up,
    int32_t* aux_arr
) {
    int32_t bid = blockIdx.x;
    int32_t tid = threadIdx.x;
    int32_t gid = bid * blockDim.x + tid;
    int32_t gwid = gid / 32;
    int32_t wid = tid / 32;
    int32_t lid = tid % 32;
    __shared__ int32_t NullNums[BLOCK_SIZE];
    __shared__ int32_t* AuxArr;
    __shared__ bool OOM;
    __shared__ int32_t col1Id;
    __shared__ int32_t col2Id;
    if (tid == 0)AuxArr = aux_arr + (bid << BLOCK_EXTRA_SIZE_BIT);
    __syncthreads();
    for (int i = bid % QueueSize;i < QueueSize;i += gridDim.x) {
        if (queue_col[i] == -1)continue;
        OOM = false;
        __syncthreads();
        col1Id = queue_col[i];
        col2Id = look_up[Low[col1Id]];
        __syncthreads();
        while (col2Id != -1) {
            int32_t* column1 = NULL;
            int32_t* column2 = NULL;
            int32_t col1Len = d_tail[col1Id] + 1;
            int32_t col2Len = d_tail[col2Id] + 1;

            if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
            if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
            if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
            if (d_rank[col1Id] == 3)column1 = &d_ell3[d_offt[col1Id]];
            if (d_rank[col2Id] == 1)column2 = &d_ell1[d_offt[col2Id]];
            if (d_rank[col2Id] == 0)column2 = &d_ell0[d_offt[col2Id]];
            if (d_rank[col2Id] == 2)column2 = &d_ell2[d_offt[col2Id]];
            if (d_rank[col2Id] == 3)column2 = &d_ell3[d_offt[col2Id]];
            cal_rank_block(AuxArr, column1, col1Len, column2, col2Len, tid, NullNums);
            int32_t resultLen = col1Len + col2Len - NullNums[blockDim.x - 1];



            if (tid == 0 && resultLen > d_tail[col1Id] + 1) {
                if (resultLen > CAP_COL[d_rank[col1Id]]) {
                    int32_t D_Rank = d_rank[col1Id];
                    while (resultLen > CAP_COL[D_Rank])D_Rank++;
                    //////////////DEBUG///////////
                    if (D_Rank == 1) {
                        int32_t t_number = atomicAdd(d_len + D_Rank, 1);
                        if (t_number > cap1) {
                            printf("Level 1 array's capbility is not enough\n");
                            OOM = true;
                        }
                        else {
                            d_offt[col1Id] = 1ll * t_number * CAP_COL[D_Rank];
                            d_rank[col1Id] = D_Rank;
                        }
                    }
                    else if (D_Rank == 2) {
                        int32_t t_number = atomicAdd(d_len + D_Rank, 1);
                        if (t_number >= cap2) {
                            printf("Level 2 array's capbility is not enough\n");
                            OOM = true;
                        }
                        else {
                            d_offt[col1Id] = 1ll * t_number * CAP_COL[D_Rank];
                            d_rank[col1Id] = D_Rank;
                        }
                    }
                    else if (D_Rank == 3) {
                        int32_t t_number = atomicAdd(d_len + D_Rank, 1);
                        if (t_number >= cap3) {
                            printf("Level 3 array's capbility is not enough\n");
                            OOM = true;
                        }
                        else {
                            d_offt[col1Id] = 1ll * t_number * CAP_COL[D_Rank];
                            d_rank[col1Id] = D_Rank;
                        }
                    }
                    else {
                        OOM = true;
                        printf("Len:%d Level array's level is not enough on GPU P2\n", resultLen);
                    }
                    if (d_offt[col1Id] < 0)printf("ERROR");
                    //////////////DEBUG///////////
                }
            }
            __syncthreads();
            if (OOM) {
                // queue_tag[i] = 0;
                // inqueue[col1Id] = 0;
                // if(tid==0)atomicSub(queue_count, 1);
                stableflag[col1Id] = 2;//2 indicates that this column will not be processed at the moment.
                clear_aux_arr_block(column1, AuxArr, col1Len + col2Len, NullNums, tid);
                break;
            }
            __syncthreads();
            d_tail[col1Id] = resultLen - 1;

            stableflag[col1Id] = 3 * int(!resultLen);
            if (d_rank[col1Id] == 1)column1 = &d_ell1[d_offt[col1Id]];
            if (d_rank[col1Id] == 0)column1 = &d_ell0[d_offt[col1Id]];
            if (d_rank[col1Id] == 2)column1 = &d_ell2[d_offt[col1Id]];
            if (d_rank[col1Id] == 3)column1 = &d_ell3[d_offt[col1Id]];
            //arr
            arr_compact_block(column1, AuxArr, col1Len + col2Len, NullNums, tid);

            Low[col1Id] = (resultLen == 0) ? -1 : column1[resultLen - 1];
            if (tid == 0) {
                if (Low[col1Id] != -1 && look_up[Low[col1Id]] < col1Id) {
                    col2Id = look_up[Low[col1Id]];
                }
                else col2Id = -1;
            }
            __syncthreads();

        }
        if (tid == 0)queue_col[i] = -1;
        __syncthreads();

    }
}


template <typename T>
void printGPUArrayRange(const T* d_array, int a, int b) {
    if (a > b) {
        std::cerr << "Invalid range: a should be less than or equal to b." << std::endl;
        return;
    }
    int rangeSize = b - a + 1;
    T* h_array = new T[rangeSize];
    CHECK_CUDA_ERROR(cudaMemcpy(h_array, d_array + a, rangeSize * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < rangeSize; ++i) {
        std::cout << std::right << std::setw(3) << h_array[i] << "\t";
    }
    std::cout << std::endl;
    delete[] h_array;
}
void printCPUArrayRange(const int* h_array, int a, int b) {
    if (a > b) {
        std::cerr << "Invalid range: a should be less than or equal to b." << std::endl;
        return;
    }
    int rangeSize = b - a + 1;
    for (int i = 0; i < rangeSize; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;
    delete[] h_array;
}
void printCPUArrayRange(const std::vector<int>& h_array, int a, int b) {
    if (a > b) {
        std::cerr << "Invalid range: a should be less than or equal to b." << std::endl;
        return;
    }
    if (a < 0 || b >= h_array.size()) {
        std::cerr << "Invalid range: a and b should be within the bounds of the array." << std::endl;
        return;
    }

    for (int i = a; i <= b; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
T getDeviceValue(T* deviceAddress) {
    T value;
    CHECK_CUDA_ERROR(cudaMemcpy(&value, deviceAddress, sizeof(T), cudaMemcpyDeviceToHost));
    return value;
}
template <typename T>
void replaceGPUArrays(T*& old_arr, T* new_arr) {
    if (old_arr) {
        CHECK_CUDA_ERROR(cudaFree(old_arr));
    }
    old_arr = new_arr;
}
void checkCudaLastError(const char* action) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error after " << action << ": " << cudaGetErrorString(err) << std::endl;
        // 处理错误（例如，退出程序或采取其他措施）
    }
}
void printBinary(uint64_t num) {
    // 使用 std::bitset 来输出二进制表示
    std::bitset<64> binary(num);
    std::cout << binary << std::endl;
}

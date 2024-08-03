#pragma once
#include <bits/stdc++.h>
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


#include <mma.h>

#include <thread>
#include <mutex>
#include <algorithm>


const int32_t MAXDIM = 5;

#ifdef MULTIKEY
const int32_t MAXKEY = 2;
#else
const int32_t MAXKEY = 1;
#endif
const int32_t BLOCK_SIZE_BIT = 8;
const int32_t BLOCK_SIZE = 1 << BLOCK_SIZE_BIT;
const int32_t WARP_SIZE_BIT = 5;
const int32_t WARP_SIZE = 1 << WARP_SIZE_BIT;
const int32_t WARP_EXTRA_SIZE_BIT = 15;
const int32_t WARP_EXTRA_SIZE = 1 << WARP_EXTRA_SIZE_BIT;
const int32_t BLOCK_EXTRA_SIZE_BIT = BLOCK_SIZE_BIT - WARP_SIZE_BIT + WARP_EXTRA_SIZE_BIT;
const int32_t BLOCK_EXTRA_SIZE = 1 << BLOCK_EXTRA_SIZE_BIT;



// 2^10    2^11    2^12    2^13    2^14    2^15    2^16    2^17    2^18    2^19    2^20
// 1024    2048    4096    8192    16384   32768   65536   131072  262144  524288  1048576
const int32_t QUEUE_SIZE=(1<<16);
const int32_t QUEUE_SIZE_MASK=(1<<16)-1;

const int32_t SS_BLOCK=25600;
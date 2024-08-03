#include <cuda_runtime.h>
#include <iostream>

class GPUTimer {
public:
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() {
        cudaEventRecord(start, 0);
    }

    void stopTimer() {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
    }

    float getElapsedTime() {
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        return elapsedTime;
    }

private:
    cudaEvent_t start, stop;
};

class CPUTimer {
public:
    CPUTimer() : start_time(), end_time() {}

    void startTimer() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stopTimer() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    float getElapsedTime() const {
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};
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
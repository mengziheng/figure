#include <stdio.h>
#include <random>
#include "gputimer.h"
#include <chrono>
using namespace std;

__global__ void Scan(int *histogram, int bucket_num, int exp_size)
{
    extern __shared__ int s_histogram[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < exp_size)
    {
        // expand array to power of 2
        if (tid < bucket_num)
            s_histogram[tid] = histogram[tid];
        else
            s_histogram[tid] = 0;

        int count = 1;
        for (unsigned int s = exp_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                s_histogram[(tid + 1) * (1 << count) - 1] = s_histogram[(tid + 1) * (1 << count) - 1 - (1 << (count - 1))] + s_histogram[(tid + 1) * (1 << count) - 1];
                count++;
            }
            __syncthreads();
        }
        s_histogram[exp_size - 1] = 0;
        count--;
        int tmp_1;
        int tmp_2;
        for (unsigned int s = 1; s < exp_size; s <<= 1)
        {
            if (tid < s)
            {
                tmp_1 = s_histogram[(tid + 1) * (1 << count) - 1 - (1 << (count - 1))];
                tmp_2 = s_histogram[(tid + 1) * (1 << count) - 1];
                s_histogram[(tid + 1) * (1 << count) - 1 - (1 << (count - 1))] = tmp_2;
                s_histogram[(tid + 1) * (1 << count) - 1] = tmp_1 + tmp_2;
                count--;
            }
            __syncthreads();
        }
        histogram[tid] = s_histogram[tid];
    }
}

__global__ void ScanNoShared(int *histogram, int exp_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < exp_size)
    {
        int count = 1;
        for (unsigned int s = exp_size / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                histogram[(tid + 1) * (1 << count) - 1] = histogram[(tid + 1) * (1 << count) - 1 - (1 << (count - 1))] + histogram[(tid + 1) * (1 << count) - 1];
                count++;
            }
            __syncthreads();
        }
        histogram[exp_size - 1] = 0;
        count--;
        int tmp_1;
        int tmp_2;
        for (unsigned int s = 1; s < exp_size; s <<= 1)
        {
            if (tid < s)
            {
                tmp_1 = histogram[(tid + 1) * (1 << count) - 1 - (1 << (count - 1))];
                tmp_2 = histogram[(tid + 1) * (1 << count) - 1];
                histogram[(tid + 1) * (1 << count) - 1 - (1 << (count - 1))] = tmp_2;
                histogram[(tid + 1) * (1 << count) - 1] = tmp_1 + tmp_2;
                count--;
            }
            __syncthreads();
        }
    }
}

unsigned int nextPowerOfTwo(unsigned int n)
{
    unsigned int count = 0;

    // if n == power of 2
    if (n && !(n & (n - 1)))
        return n;

    while (n != 0)
    {
        n >>= 1;
        count++;
    }

    return 1 << count;
}

int main()
{
    cudaEvent_t start, stop;
    cudaEvent_t start_1, stop_1;
    float elapsedTime;
    float elapsedTime1;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);

    int size = 256;
    int exp_size = nextPowerOfTwo(size);
    int *d_arr;
    int *d_arr_2;
    int *arr = (int *)malloc(sizeof(int) * size);
    int *arr_2 = (int *)malloc(sizeof(int) * exp_size);
    printf("%d\n", exp_size);

    int i;
    for (i = 0; i < size; i++)
    {
        arr[i] = i;
        arr_2[i] = i;
    }

    for (i; i < exp_size; i++)
    {
        arr_2[i] = 0;
    }

    cudaMalloc(&d_arr, sizeof(int) * size);
    cudaMemcpy(d_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_arr_2, sizeof(int) * size);
    cudaMemcpy(d_arr_2, arr_2, sizeof(int) * size, cudaMemcpyHostToDevice);
    int z;
     z = 0;
    cudaEventRecord(start, 0);
    while (z < 1000)
    {
        Scan<<<1, 1024, 4 * size>>>(d_arr, size, exp_size);
        z++;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time for shared memory: %f ms\n", elapsedTime);

    z = 0;
    cudaEventRecord(start_1, 0);
    while (z < 1000)
    {
        ScanNoShared<<<1, 1024>>>(d_arr_2, exp_size);
        z++;
    }
    cudaEventRecord(stop_1, 0);
    cudaEventSynchronize(stop_1);
    cudaEventElapsedTime(&elapsedTime1, start_1, stop_1);
    printf("Elapsed time for no shared memory: %f ms\n", elapsedTime1);

    cudaDeviceSynchronize();



    cudaMemcpy(arr, d_arr, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(arr_2, d_arr_2, sizeof(int) * size, cudaMemcpyDeviceToHost);

    printf("\n");
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);

    printf("\n");
    printf("\n");
    for (int i = 0; i < size; i++)
        printf("%d ", arr_2[i]);

    return 0;
}
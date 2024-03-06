// Using different memory spaces in CUDA
#include <stdio.h>
#include "gputimer.h"
#define SIZE 128

// a __global__ function runs on the GPU & can be called from host
__global__ void use_global_memory_GPU(float *array)
{
    // "array" is a pointer into global memory on the device
    array[threadIdx.x] = threadIdx.x * 2.0;
}

// (for clarity, hardcoding 128 threads/elements and omitting out-of-bounds checks)
// 用这么多线程去计算平均值
__global__ void use_shared_memory_GPU(float *array)
{
    // local variables, private to each thread
    int index = threadIdx.x;
    int sum = 0;
    float ave;

    // __shared__ variables are visible to all threads in the thread block
    // and have the same lifetime as the thread block
    __shared__ float sh_arr[128];

    // copy data from "array" in global memory to sh_arr in shared memory.
    // here, each thread is responsible for copying a single element.
    sh_arr[index] = array[index];

    // ensure all the writes to shared memory have completed
    __syncthreads();
    // now, sh_arr is fully populated. Let's find the average of all previous elements
    for(int i = 0; i < index; i++){
        sum += sh_arr[i];
    }
    ave = sum / (float) index;

    // if array[index] is greater than the average of array[0..index-1], replace with average.
    // since array[] is in global memory, this change will be seen by the host (and potentially 
    // other thread blocks, if any)

    // 从shared memory 写入 global memory
    // if(sh_arr[index] > ave)
    //     sh_arr[index] = ave;
    // __syncthreads();
    // array[index] = sh_arr[index];
    
    // 直接从local memory 写入 global memory
    if(array[index] > ave)
        array[index] = ave;
}

int main(int argc, char **argv)
{
    GpuTimer timer;
    float h_arr[SIZE] = {0.0};      // convention: h_ variables live on host
    float* d_arr;      // convention: d_ variables live on device (GPU global mem)
    cudaMalloc(&d_arr, sizeof(int) * SIZE);
    cudaMemcpy(d_arr,h_arr, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    // launch the kernel (1 block of 128 threads)
    use_global_memory_GPU<<<1,128>>>(d_arr);  // modifies the contents of array at d_arr
    // copy the modified array back to the host, overwriting contents of h_arr
    cudaMemcpy(h_arr, d_arr, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    for(int i = 0; i < SIZE; i++)
        printf("Array[%d] = %f\n",i,h_arr[i]);

    timer.Start();
    // as before, pass in a pointer to data in global memory
    use_shared_memory_GPU<<<1,128>>>(d_arr);
    timer.Stop();
    cudaMemcpy(h_arr, d_arr, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    // copy the modified array back to the host
    for(int i = 0; i < SIZE; i++)
        printf("Array[%d] = %f\n",i,h_arr[i]);
    printf("Time elapsed = %g ms\n",timer.Elapsed());
    cudaFree(d_arr);
    return 0;
}
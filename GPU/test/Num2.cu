#include <cstdint>
#include <iostream>
#include <vector>

#include <cooperative_groups.h>

__global__ void kernel(int *given_tuples, int *histogram, int tuple_num, int bucket_num)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // extern __shared__ int s_histogram[];
    if(tid == 0)
        printf("%d %d\n",tuple_num,bucket_num);
    if (tid < tuple_num)
    {
        cooperative_groups::grid_group g = cooperative_groups::this_grid();
        // need to modify
        // int pos = given_tuples[tid] % bucket_num;
        // maintain a histogram in shared memory for every block
        // atomicAdd(&s_histogram[pos], 1);
        g.sync();
        printf("111");
        // // output to global memory
        // if (tid % blockDim.x == 0)
        // {
        //     printf("tid : %d\n", tid);
        //     for (int i = 0; i < bucket_num; i++)
        //         atomicAdd(&histogram[i], s_histogram[i]);
        // }

        // g.sync();
        // if (tid == 0)
        // {
        //     for (int i = 0; i < bucket_num; i++)
        //     {
        //         printf("%d\n", histogram[i]);
        //         if (histogram[i] != 0)
        //             histogram[i] = ((histogram[i]) / 32 + 1) * 32;
        //     }
        // }
    }
}

int main()
{
    int size = 2000;
    int bucketnum = 32;
    int *arr = (int *)malloc(size * 4);
    for (int i = 0; i < size; i++)
        arr[i] = i;
    int *d_values;
    cudaMalloc(&d_values, sizeof(int) * size);
    cudaMemcpy(d_values, arr, sizeof(int) * size, cudaMemcpyHostToDevice);

    int *d_histogram;
    cudaMalloc(&d_histogram, 4 * size);
    cudaMemset(d_histogram, 0, 4 * size);

    void *params[] = {&d_values, &d_histogram, &size, &bucketnum};

    // cudaLaunchCooperativeKernel((void *)kernel, 3, 1024, params, 4 * bucketnum);
    cudaLaunchCooperativeKernel((void *)kernel, 2, 1024, params);

    cudaMemcpy(arr, d_values, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaDeviceSynchronize();
    return 0;
}
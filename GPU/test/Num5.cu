// Num 5 
// 一共 1024 * 1024 个数组 我们有1024 blocks * 1024 threads
// 最后再由一个数组处理这1024个元素

#include <stdio.h>
#include "gputimer.h"

using namespace std;

#define TOTALSIZE (1<<20)
#define BINSIZE (1<<10)

// 最后将结果写进全局空间的一个数组
__global__ void reduce(float* arr_in, float* arr_out){
    // 需要内存连读取
    // 一个block 共享内存
    // 先把整个数组取到shared memory中
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int t_index = threadIdx.x;

    // shared memory的大小可以在调用函数时设定，需要加关键词extern
    extern __shared__ float sdata[];
    // 先将数据存在shared_memory中
    sdata[t_index] = arr_in[index];
    __syncthreads();

    for(int i = blockDim.x / 2; i >= 1; i >>= 1){
        if(t_index < blockDim.x)
            sdata[t_index] += sdata[t_index + i];
        __syncthreads(); 
    }
    
    // 留一个元素写回全局空间
    if(t_index == 0)
        arr_out[blockDim.x] = sdata[0]; 
}

int main(){
    GpuTimer timer;

    int devCount;
    cudaGetDeviceCount(&devCount);
    
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if(cudaGetDeviceProperties(&devProps,dev) == 0){
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
    }

    float h_arr[TOTALSIZE];
    float sum;
    // 先初始化一下，然后计算一下正确的和
    for(int i = 0; i <TOTALSIZE; i++){
        h_arr[i] = -1.0f + random()/(RAND_MAX/2.0f);
        if(i < 10)
            printf("h_in[%d] = %f \n",i,h_arr[i]);
        sum += h_arr[i];
    }
    printf("true sum = %f\n",sum);

    // 用GPU计算
    // 先分配全局空间内存
    float * d_in;
    float * d_out;
    int totalbyte = TOTALSIZE * sizeof(float);
    int binbyte = BINSIZE * sizeof(float);
    cudaMalloc(&d_in, totalbyte);
    cudaMalloc(&d_out, binbyte);
    cudaMemcpy(d_in, h_arr, totalbyte, cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,binbyte);

    timer.Start();
    reduce<<<1024,1024,1024 * sizeof(float)>>>(d_in,d_out);
    reduce<<<1,1024,1024 * sizeof(float)>>>(d_in,d_out);
    timer.Stop();
    printf("Time elapse = %g ms\n",timer.Elapsed());

    cudaMemcpy(&sum, &d_out[0], sizeof(float), cudaMemcpyDeviceToHost);

    printf("reduce sum = %f\n",sum);
}
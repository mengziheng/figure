#include<iostream>

__global__ void kernel_1(int* a){
    int tid = threadIdx.x;
    int i = a[0];
    a[1] = i;
}

__global__ void kernel_2(int* a){
    int i0 = a[0], i1 = a[1];
    a[2] = i0;
    a[3] = i1;
}

float fun(int x){
    return x - std::pow(x - 1, 32) / std::pow(x, 32) * x;
}

int main(){
    int arr[10];
    for(int i = 0; i < 10; i++)
        arr[i] = i;
    int* d_arr;
    cudaMalloc(&d_arr, sizeof(int) * 10);
    cudaMemcpy(d_arr,arr,sizeof(int) * 10,cudaMemcpyHostToDevice);

    kernel_1<<<1,32>>>(d_arr);
    kernel_2<<<1,32>>>(d_arr);

    cudaDeviceSynchronize();

    std::cout << fun(2) << std::endl;
    std::cout << fun(4) << std::endl;
    std::cout << fun(8) << std::endl;
    std::cout << fun(32) << std::endl;
    std::cout << fun(64) << std::endl;
    std::cout << fun(128) << std::endl;
    std::cout << fun(256) << std::endl;
    std::cout << fun(512) << std::endl;
    std::cout << fun(1024) << std::endl;
    std::cout << fun(2048) << std::endl;
    std::cout << fun(4096) << std::endl;
}
#include <stdio.h>
#include <iostream>

__global__ void aadd(unsigned int *d){
    __shared__ unsigned int s;
    s = *d;
    int m = 1;
    if(threadIdx.x==1)
        m = 0;
    atomicAdd(&s,m);
    *d = s;
}

void print_binary(unsigned int number) {
    if (number >> 1) {
        print_binary(number >> 1);
    }
    putc((number & 1) ? '1' : '0', stdout);
}

int main(){
    unsigned int* dev_arr;
    unsigned int* host_arr = (unsigned int*)malloc(sizeof(unsigned int));
    cudaMalloc(&dev_arr,sizeof(int));
    cudaMemset(&dev_arr,0,sizeof(int));
    aadd<<<1,32>>>(dev_arr);
    cudaDeviceSynchronize();
    cudaMemcpy(host_arr, dev_arr, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%d\n",*host_arr);
    return 0;
}
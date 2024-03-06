#include<iostream>

using namespace std;

__global__ void kernel(){
    printf("1");
    int* ptr;
    if(threadIdx.x == 1){
        cudaMalloc(&ptr, 32);
        ptr[0] = 0;
        ptr[1] = 1;
        ptr[2] = 2;
        ptr[3] = 3;
    }
    if(threadIdx.x == 2){
        printf("%d",ptr[0]);
        printf("%d",ptr[1]);
        printf("%d",ptr[2]);
        printf("%d",ptr[3]);
    }
}

int main(){
    kernel<<<1,32>>>();
    cudaDeviceSynchronize();
    printf("\n");
}
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "random"
#include "algorithm"
#include "gputimer.h"

using namespace std;

const uint32_t RSize = 32;
const uint32_t SSize = 64;

//用来测试以warp为单位进行比较

const uint32_t kEmpty = 0xffffffff;
/* --- PRINTF_BYTE_TO_BINARY macro's --- */
#define PRINTF_BINARY_PATTERN_INT8 "%c%c%c%c%c%c%c%c"
#define PRINTF_BYTE_TO_BINARY_INT8(i)    \
    (((i) & 0x80ll) ? '1' : '0'), \
    (((i) & 0x40ll) ? '1' : '0'), \
    (((i) & 0x20ll) ? '1' : '0'), \
    (((i) & 0x10ll) ? '1' : '0'), \
    (((i) & 0x08ll) ? '1' : '0'), \
    (((i) & 0x04ll) ? '1' : '0'), \
    (((i) & 0x02ll) ? '1' : '0'), \
    (((i) & 0x01ll) ? '1' : '0')
 
#define PRINTF_BINARY_PATTERN_INT16 \
    PRINTF_BINARY_PATTERN_INT8              PRINTF_BINARY_PATTERN_INT8
#define PRINTF_BYTE_TO_BINARY_INT16(i) \
    PRINTF_BYTE_TO_BINARY_INT8((i) >> 8),   PRINTF_BYTE_TO_BINARY_INT8(i)
#define PRINTF_BINARY_PATTERN_INT32 \
    PRINTF_BINARY_PATTERN_INT16             PRINTF_BINARY_PATTERN_INT16
#define PRINTF_BYTE_TO_BINARY_INT32(i) \
    PRINTF_BYTE_TO_BINARY_INT16((i) >> 16), PRINTF_BYTE_TO_BINARY_INT16(i)

__global__ void ballot(unsigned int *R , unsigned int *S,unsigned int *result){
    unsigned int s = S[threadIdx.x]; //假设每个线程读取的都一样
    // 这些地方都可以改进完善
    unsigned int lane = threadIdx.x % 32;
    unsigned int r = R[lane]; // 线程读取的s和warp读取的32个r进行比较
    unsigned int mask = ~0;
    int i;
    for(i = 0 ; i < 32 ; i ++)
    {
        unsigned int bit = 1 << i ; 
        unsigned int vote = __ballot_sync(0xFFFFFFFF,r & bit); //一共32位，对应32个数的第i位，每位取决于当前读取的r的第i位是否为0
        mask = mask & (( s & bit) ? vote : ~vote); // 一共32个数，检查s的第i位与每个数的第i位是否与相匹配，只要匹配就是1，一旦有1位不匹配，则不管其他循环怎么样，都会为0

        // 这是一个展示结果的过程,会发现1在逐渐变少，因为（32个数中）越来越多的数字中的某一位发现与要比较的s不同，因此被置为0
        if(threadIdx.x == 47){
           printf("i : %2.d votes : ",i);
           printf(
           PRINTF_BINARY_PATTERN_INT32 ,
           PRINTF_BYTE_TO_BINARY_INT32(vote));
           printf(" mask : ");
           printf(
           PRINTF_BINARY_PATTERN_INT32 "\n",
           PRINTF_BYTE_TO_BINARY_INT32(mask));
        }
    }
    result[threadIdx.x] = mask;
}

// shuffle 
__global__ void shuffle(unsigned int *R , unsigned int*S,unsigned int *result){
    // 从global memory中读入数据，依此与shared memory中的数据进行比较
    // __shared__ unsigned int* sha_arr;
    // 每个thread先从global memory中读取一个
    unsigned int s = S[threadIdx.x];
    // 每个thread再从shared memory中读取一个用来比较
    unsigned int lane = threadIdx.x % 32;
    unsigned int r = R[lane]; // 线程读取的s和warp读取的32个r进行比较
    // 用来记录最终结果，是32bit的数字
    unsigned int mask = 0;
    // 用于记录shuffle的值
    unsigned int value;
    int i;
    // printf("thread %u read %u\n",threadIdx,s);
    for(i = 0 ; i < 32 ; i ++)
    {
        value = __shfl_sync(0xffffffff, r, i);   // Synchronize all threads in warp, and get "value" from lane 0
        if(value == s)
            mask = mask | (1 << i) ; 
        // if(threadIdx.x == 1){
        //     printf("%u %u ",s, value);
        //     printf(
        //     PRINTF_BINARY_PATTERN_INT32 "\n",
        //     PRINTF_BYTE_TO_BINARY_INT32(mask));
        // }
    }
    //result[threadIdx.x] = mask;
}

void print_binary(unsigned int number) {
    if (number >> 1) {
        print_binary(number >> 1);
    }
    putc((number & 1) ? '1' : '0', stdout);
}

// 生成随机的KV，范围是0 --- (KEmpty-1)
std::vector<int> generate_random_keyvalues(std::mt19937& rnd, uint32_t numkvs,string filename)
{   
    std::ofstream fout(filename,std::ios::binary);
    std::uniform_int_distribution<uint32_t> dis(0, kEmpty - 1);
    std::vector<int> kvs;  
    // reserve改变容量，resize改变大小
    kvs.reserve(numkvs);
    for (uint32_t i = 0; i < numkvs; i++)
    {
        uint32_t rand = dis(rnd);
        kvs.push_back(rand);
        fout.write((char*)&rand, sizeof(unsigned int));
    }
    fout.close();
    return kvs;
}

// return numshuffledkvs random items from kvs
std::vector<int> shuffle_keyvalues(std::mt19937& rnd, std::vector<int> kvs, uint32_t numshuffledkvs,string filename)
{
    std::shuffle(kvs.begin(), kvs.end(), rnd);
    std::vector<int> shuffled_kvs;
    shuffled_kvs.resize(numshuffledkvs);

    std::copy(kvs.begin(), kvs.begin() + numshuffledkvs, shuffled_kvs.begin());

    std::ofstream fout(filename,std::ios::binary);
    for (uint32_t i = 0; i < numshuffledkvs; i++)
    {
        fout.write((char*)&shuffled_kvs[i], sizeof(unsigned int));
    }
    fout.close();
    return shuffled_kvs;
}

int main(){

    // 用来创建数据
    std::random_device rd;
    uint32_t seed = 1;
    std::mt19937 rnd(seed);  // mersenne_twister_engine
    std::vector<int> S_vector = generate_random_keyvalues(rnd, SSize,"S");
    std::vector<int> R_vector = shuffle_keyvalues(rnd, S_vector, RSize,"R");
    
    // for (vector<int>::iterator it = R_vector.begin(); it != R_vector.end(); ++it) {
    //     cout << *it << " ";
    // }

    unsigned int* dev_S;
    unsigned int* dev_R;
    unsigned int* dev_Result_ballot;
    unsigned int* dev_Result_shuffle;
    unsigned int* host_Result_ballot = (unsigned int *)malloc(sizeof(unsigned int) * SSize);
    unsigned int* host_Result_shuffle = (unsigned int *)malloc(sizeof(unsigned int) * SSize);
    cudaMalloc(&dev_S,sizeof(unsigned int)*SSize);
    cudaMalloc(&dev_R,sizeof(unsigned int)*RSize);
    cudaMalloc(&dev_Result_ballot,sizeof(unsigned int)*SSize);
    cudaMalloc(&dev_Result_shuffle,sizeof(unsigned int)*SSize);
    cudaMemcpy(dev_S,S_vector.data(),sizeof(unsigned int)*SSize,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_R,R_vector.data(),sizeof(unsigned int)*RSize,cudaMemcpyHostToDevice);
    GpuTimer time;


    shuffle<<<1,SSize>>>(dev_R,dev_S,dev_Result_shuffle);
    shuffle<<<1,SSize>>>(dev_R,dev_S,dev_Result_shuffle);
    shuffle<<<1,SSize>>>(dev_R,dev_S,dev_Result_shuffle);
    shuffle<<<1,SSize>>>(dev_R,dev_S,dev_Result_shuffle);
    shuffle<<<1,SSize>>>(dev_R,dev_S,dev_Result_shuffle);
    shuffle<<<1,SSize>>>(dev_R,dev_S,dev_Result_shuffle);
    shuffle<<<1,SSize>>>(dev_R,dev_S,dev_Result_shuffle);
    cudaDeviceSynchronize(); // synchronzie
    time.Start();
    shuffle<<<1,SSize>>>(dev_R,dev_S,dev_Result_shuffle);
    time.Stop();
    cudaDeviceSynchronize(); // synchronzie
    printf("Time elapse = %g ms\n",time.Elapsed());

    
    ballot<<<1,SSize>>>(dev_R,dev_S,dev_Result_ballot);
    ballot<<<1,SSize>>>(dev_R,dev_S,dev_Result_ballot);
    ballot<<<1,SSize>>>(dev_R,dev_S,dev_Result_ballot);
    ballot<<<1,SSize>>>(dev_R,dev_S,dev_Result_ballot);
    ballot<<<1,SSize>>>(dev_R,dev_S,dev_Result_ballot);
    cudaDeviceSynchronize(); // synchronzie

    time.Start();
    ballot<<<1,SSize>>>(dev_R,dev_S,dev_Result_ballot);
    time.Stop();
    cudaDeviceSynchronize(); // synchronzie
    printf("Time elapse = %g ms\n",time.Elapsed());



    cudaMemcpy(host_Result_ballot,dev_Result_ballot,sizeof(unsigned int)*SSize,cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Result_shuffle,dev_Result_shuffle,sizeof(unsigned int)*SSize,cudaMemcpyDeviceToHost);

    // for(int i = 0 ; i < SSize; i++){
    //     printf("%u %u\n",host_Result_ballot[i],host_Result_shuffle[i]);
    // }

    // for(int i = 0 ; i < SSize; i++){
    //     print_binary(host_Result_shuffle[i]);
    // }

    // for(int i = 0 ; i < SSize; i++){
    //     printf("%u\n",S_vector[i]);
    // }

    // for(int i = 0 ; i < RSize; i++){
    //     printf("%u\n",R_vector[i]);
    // }

    // cudaDeviceSynchronize();
    // return 0;
}


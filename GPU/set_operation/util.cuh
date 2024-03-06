#include "cuda_runtime.h"

#define S 100
#define WARP_SIZE 32
#define CHUNK_SIZE 32
#define MIN_BUCKET_NUM 8
#define BUCKET_SIZE 4

using namespace std;

#define map_value(hash_value, bucket_num) (((hash_value) % MIN_BUCKET_NUM) * (bucket_num / MIN_BUCKET_NUM) + (hash_value) / MIN_BUCKET_NUM)

#define gpuTiming(...) cudaEvent_t start, end; \
        cudaEventCreate(&start); \
        cudaEventCreate(&end); \
        cudaEventRecord(start, 0); \
        __VA_ARGS__; \
        cudaEventRecord(end, 0); \
        cudaEventSynchronize(end); \
        float elapsed_time; \
        cudaEventElapsedTime(&elapsed_time, start, end); \
        printf("Time to generate:  %f ms\n", elapsed_time); \
        cudaEventDestroy(start); \
        cudaEventDestroy(end); \

void sort(int *array, int size);
__device__ void load_next_index(int &cur_index, int *latest_index);
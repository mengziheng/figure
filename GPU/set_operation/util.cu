#include "util.cuh"

// due to the overhead of kernel launch non-negligible, when array is small, we sort on CPU
void sort(int *array, int size)
{
    if (size < S)
        sort_on_CPU(array, size);
    else
    {
        sort_on_GPU<<<216, 1024>>>(array, size);
    }
    return;
}

void sort_on_CPU(int *array, int size)
{
}

__global__ void sort_on_GPU(int *array, int size)
{
}

__device__ void set_intersection()
{
}

__device__ void load_next_index(int &cur_index, int *latest_index)
{
    if (cur_index % CHUNK_SIZE != CHUNK_SIZE - 1)
    {
        cur_index++;
        return;
    }

    if (threadIdx.x == 0)
    {
        cur_index = atomicAdd(latest_index, CHUNK_SIZE);
    }
    cur_index = __shfl_sync(0xffffffff, cur_index, 0);
}

// hash table的结构是否可以优化？
__inline__ __device__ bool search_in_hashtable(int x, int *hash_table, int hash_table_len)
{
    int mapped_value = map_value(x % hash_table_len, hash_table_len);
    int *cmp = hash_table + mapped_value;
    int index = 0;
    while (*cmp != -1)
    {
        if (*cmp == x)
        {
            return true;
        }
        cmp = cmp + hash_table_len;
        index++;
        // 这个if语句注释了会快多少？
        if (index == BUCKET_SIZE)
        {
            mapped_value++;
            index = 0;
            if (mapped_value == hash_table_len)
                mapped_value = 0;
            cmp = &hash_table[mapped_value];
        }
    }
    return false;
}

// 1、一个warp并行处理32个查询
// 每个线程负责
// 可以优化一下负载不均衡，让thread做完自动下一个
// 输出暂时简化处理
__global__ void set_intersection(int *array, int array_size, int *hash_table, int hash_table_len, int latest_index, int *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int item, bucket;
    for (int cur_index = tid; cur_index < array_size; cur_index += blockDim.x * gridDim.x)
    {
        item = array[cur_index];
        if (search_in_hashtable(item, hash_table, hash_table_len))
            output[cur_index] = -1;
        else
            output[cur_index] = item;
    }
    return;
}

// 2、一个warp连续处理CHUNK_SIZE个查询
// 这个需要hash table的构建方式是按行存储的
// __global__ void set_intersection(int *array, int array_size, int *hash_table, int hash_table_len, int latest_index)
// {
//     int wid = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
//     int stride = blockDim.x * gridDim.x / WARP_SIZE;
//     int cur_index = wid * CHUNK_SIZE;
//     int item, bucket;
//     for (; cur_index < array_size; load_next_index(cur_index, latest_index))
//     {
//         item = array[cur_index];
//         bucket = map_value(item % hash_table_len, hash_table_len);
        
//         // search in hashtable parallel
//         int cmp = hash_table + mapped_value;
//     }
//     return;
// }

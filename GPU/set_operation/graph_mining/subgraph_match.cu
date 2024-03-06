#include <sstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cooperative_groups.h>
#include "hash_table.cuh"
#include "subgraph_match.cuh"

using namespace cooperative_groups;
using namespace std;

#define S 100
#define WARP_SIZE 32
#define CHUNK_SIZE 1
#define MIN_BUCKET_NUM 8
#define BUCKET_SIZE 4

#define array3D(devPtr, depth, height, width, pitch) devPtr[depth * BLOCK_NUM * BLOCK_SIZE / 32 * pitch / sizeof(int) + height * pitch / sizeof(int) + width]

// h : height of subtree; h = pattern vertex number
__inline__ __device__ bool checkDuplicate(int *mapping, int &level, int item)
{
    for (int i = 0; i < level; i++)
        if (mapping[i] == item)
            return true;
    return false;
}
__inline__ __device__ bool checkRestriction(int *mapping, int &level, int item, int *restriction)
{
#ifdef withDuplicate
    // if (restriction[level] == -1)
    // {
    for (int i = 0; i < level; i++)
        if (mapping[i] == item)
            return true;
            // }
#endif
#ifdef withRestriction
    if (restriction[level] == -1)
        return false;
    if (item < mapping[restriction[level]])
        return false;
    return true;
#else
    return false;
#endif
}

__device__ void load_next_index(int &cur_index, int *latest_index)
{
    if (cur_index % CHUNK_SIZE != CHUNK_SIZE - 1)
    {
        cur_index++;
        return;
    }

    if (threadIdx.x % 32 == 0)
    {
        cur_index = atomicAdd(latest_index, CHUNK_SIZE);
    }
    cur_index = __shfl_sync(0xffffffff, cur_index, 0);
}

// 尝试一下把vertex_degrees去掉试一试呢？
__device__ void find_min_degree(int lid, int *vertexs, int num, int *degree_offset, int &min_degree, int &min_degree_vertex)
{
    if (lid == 0)
    {
        int *vertex_degrees = new int[num];
        for (int i = 0; i < num; i++)
            vertex_degrees[i] = (degree_offset[vertexs[i] + 1] - degree_offset[vertexs[i]]);
        min_degree = INT_MAX;
        for (int i = 0; i < num; i++)
            if (vertex_degrees[i] < min_degree)
            {
                min_degree = vertex_degrees[i];
                min_degree_vertex = i;
            }
    }
    min_degree = __shfl_sync(0xffffffff, min_degree, 0, 32);
    min_degree_vertex = __shfl_sync(0xffffffff, min_degree_vertex, 0, 32);
}

// 加不加inline有没有区别
__device__ bool search_in_hashtable(int item, int *hash_table, int hash_table_len, int bucket_num)
{
    int mapped_value = map_value(item % hash_table_len, hash_table_len);
    int *cmp = hash_table + mapped_value;
    int index = 0;
    while (*cmp != -1)
    {
        if (*cmp == item)
        {
            return true;
        }
        cmp = cmp + bucket_num;
        index++;
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

// 初始为一个点
__global__ void DFSKernelForReuse(int index_length, int bucket_num, int *degree_offset, int *adjcant, long long *hash_tables_offset, int *hash_tables, int *vertex, unsigned long long *sum, int *latest_index, int *partial_result, size_t pitch)
{
    unsigned long long my_count = 0;
    __shared__ unsigned long long shared_count;
    if (threadIdx.x == 0)
        shared_count = 0;

    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lid = threadIdx.x % 32;
    int result[4];
    int partial_result_length[4];

    // each warp process a subtree, 以点开始还是以一条边开始？;
    for (result[0] = wid * CHUNK_SIZE; result[0] < index_length; load_next_index(result[0], latest_index))
    {
        // 若干个点、若干个邻居、邻居集合是数组、数组做交集
        // 这种方式不太好我们之后尝试 先正常的写入试试
        // partial_result[1][0] = -1;
        // partial_result[1][1] = result[0]; // 这里可以试一下这样和直接写入的性能区别
        partial_result_length[1] = degree_offset[result[0] + 1] - degree_offset[result[0]];
        // ！！！！测试一下把degree_offset提取出来
        for (int i = lid; i < partial_result_length[1]; i += 32)
        {
            array3D(partial_result, 1, wid, i, pitch) = adjcant[degree_offset[result[0]] + i];
        }
        // // 这里涉及到中间点的存储问题，现在用的是这种稀疏存储的方式，需要压缩存储一下。
        // for (result[1] = array2D(partial_result,1,0,pitch); result[1] < partial_result_length[1]; result[1] = result[1]++)
        for (int index_1 = 0; index_1 < partial_result_length[1]; index_1++)
        {
            result[1] = array3D(partial_result, 1, wid, index_1, pitch);
            int min_degree;
            int min_degree_vertex;
            int *min_array;
            find_min_degree(lid, result, 2, degree_offset, min_degree, min_degree_vertex);
            min_array = &adjcant[degree_offset[result[min_degree_vertex]]];
            if (lid == 0 && wid == 0)
                printf("%d %d %d %d %d %d\n", partial_result_length[1], index_1, min_degree, min_degree_vertex, result[1], result[0]);
            // 可不可以先用寄存器保存下来，然后再删掉这个寄存器，重新分配一个寄存器。

            // 寻找需要比较的对象

            int compared_vertex = 1;
            if (min_degree == 1)
                compared_vertex = 0;

            // 对array中的数组遍历
            for (int i = lid; i < min_degree; i = i + 32)
            {
                int item = min_array[lid];

                // 先进行剪枝，是否需要？后续修改一下
                // if (item.degree > 代码生成的degree)
                //     continue;

                // 代码生成多次hash_table的search
                // bucket_num这个变量能不能优化？
                if (search_in_hashtable(item, hash_tables + hash_tables_offset[result[compared_vertex]], hash_tables_offset[result[compared_vertex] + 1] - hash_tables_offset[result[compared_vertex]], bucket_num))
                    continue;
                // 有一个问题？item存在哪里？下次取第一个
                array3D(partial_result, 2, wid, i, pitch) = item;
                my_count++;
            }
            // 由于最后一个点和第三个点对称，因此只需要从第三个点中找不同即可。
        }
        // 这一部分是负责取新的点的部分
    }
    atomicAdd(&shared_count, my_count);
    if (wid == 0 && lid == 0)
    {
        printf("im out\n");
    }
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(sum, shared_count);
}

struct arguments SubgraphMatching(int process_id, int process_num, struct arguments args, char *argv[])
{
    int deviceCount;
    HRR(cudaGetDeviceCount(&deviceCount));
    printf("device id : %d\n", (process_id + 1) % deviceCount);
    HRR(cudaSetDevice((process_id) % deviceCount));
    string Infilename = argv[1];
    string pattern = argv[2];
    load_factor = atof(argv[4]);
    bucket_size = atoi(argv[5]);

    chunk_size = atoi(argv[8]);

    int *d_adjcant, *d_vertex, *d_degree_offset;
    int max_degree;
    tie(d_adjcant, d_vertex, d_degree_offset, max_degree) = loadGraphWithName(Infilename, pattern);
    // printGpuInfo();
    printf("max degree is : %d\n", max_degree);
    int *d_hash_tables;
    long long *d_hash_tables_offset;
    long long bucket_num;
    tie(d_hash_tables_offset, d_hash_tables, bucket_num) = buildHashTable(d_adjcant, d_vertex, d_degree_offset);

    int *d_ir; // intermediate result;
    int Width = max_degree;
    int Height = BLOCK_SIZE * BLOCK_NUM / 32 * H;
    size_t Pitch;
    HRR(cudaMallocPitch(&d_ir, &Pitch, Width, Height));

    // cout << "ir memory size is : " << 216 * 32 * max_degree * H * sizeof(int) / 1024 / 1024 << "MB" << endl;
    cout << "ir memory size is : " << 216 * 32 * max_degree * H * sizeof(int) << endl;

    int *G_INDEX;
    HRR(cudaMalloc(&G_INDEX, sizeof(int)));

    unsigned long long *d_sum;
    HRR(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    HRR(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    // double start_time = wtime();

    double cmp_time;
    double time_start;
    double max_time = 0;
    double min_time = 1000;
    double ave_time = 0;

    time_start = clock();

    HRR(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    for (; process_id < process_num; process_id += deviceCount)
    {
        HRR(cudaMemset(G_INDEX, 0, sizeof(int)));

        int length;
        length = vertex_count;

        time_start = clock();
        time_start = clock();
        DFSKernelForReuse<<<BLOCK_SIZE, BLOCK_NUM>>>(length, bucket_num, d_degree_offset, d_adjcant, d_hash_tables_offset, d_hash_tables, d_vertex, d_sum, G_INDEX, d_ir, Pitch);

        HRR(cudaDeviceSynchronize());
        cmp_time = clock() - time_start;
        cmp_time = cmp_time / CLOCKS_PER_SEC;
        if (cmp_time > max_time)
            max_time = cmp_time;
        if (cmp_time < min_time)
            min_time = cmp_time;
        ave_time += cmp_time;
        // cout << "this time" << cmp_time << ' ' << max_time << endl;
        HRR(cudaFree(d_ir));
        HRR(cudaMalloc(&d_ir, (long long)216 * 32 * max_degree * H * sizeof(int)));
    }

    HRR(cudaFree(d_hash_tables));
    HRR(cudaFree(d_hash_tables_offset));
    HRR(cudaFree(d_adjcant));
    HRR(cudaFree(d_vertex));
    HRR(cudaFree(d_degree_offset));
    HRR(cudaFree(d_ir));

    std::cout << "time: " << max_time * 1000 << " ms" << std::endl;

    long long sum;
    cudaMemcpy(&sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);
    cout << pattern << " count is " << sum << endl;

    args.time = max_time;
    args.count = sum;
    return args;
}
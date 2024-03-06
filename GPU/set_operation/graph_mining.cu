#include "util.cuh"

// 这里到时候修改一下看看会不会快一些
// __device__ void find_min_degree(int lid, int *vertexs, int num, int *degree_offset)
// {
//     int vertex_degree = (degree_offset[vertexs[lid] + 1] - degree_offset[vertexs[lid]]) * (lid <= num) + -1 * (lid > num);
//     int min_degree = vertex_degree;
//     for (int i = 16; i >= 1; i /= 2)
//     {
//         int temp = __shfl_xor_sync(0xffffffff, min_degree, i, 32);
//         if (temp < min_degree)
//         {
//             min_degree = temp;
//         }
//     }
//     if(min_degree == vertex_degree){

//     }
// }

// 这里到时候修改一下看看会不会快一些
__device__ void find_min_degree(int lid, int *vertexs, int num, int *degree_offset, int &min_degree, int &min_degree_vertex)
{
    if (lid == 0)
    {
        int vertex_degrees[num];
        for (int i = 0; i < num; i++)
            vertex_degrees[num] = (degree_offset[vertexs[lid] + 1] - degree_offset[vertexs[lid]]);
        min_degree_vertex = INT_MAX;
        for (int i = 0; i < num; i++)
            if (vertex_degrees[num] < min_degree)
            {
                min_degree = vertex_degrees[num];
                min_degree_vertex = i;
            }
    }
    min_degree = __shfl_sync(0xffffffff, min_degree, 0, 32);
    min_degree_vertex = __shfl_sync(0xffffffff, min_degree_vertex, 0, 32);
}

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
__global__ void DFSKernelForReuse(int chunk_size, int index_length, int bucket_size, int bucket_num, int max_degree, int *degree_offset, int *adjcant, long long *hash_tables_offset, int *hash_tables, int *vertex, int *candidates_of_all_warp, unsigned long long *sum, int *latest_index, int **partial_result)
{
    int sum 0;
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpid
    int lid = threadIdx.x % 32;
    int result[4];
    int partial_result_length[4];

    // each warp process a subtree
    // result[0]后续可以修改，不然感觉会负载不均衡;
    for (result[0] = wid * chunk_size; result[0] < index_length; load_next_index(result[0], latest_index))
    {
        // 若干个点、若干个邻居、邻居集合是数组、数组做交集
        partial_result[1][0] = -1;
        partial_result[1][1] = result[0]; // 这里可以试一下这样和直接写入的性能区别
        partial_result_length[1] = degree_offset[result[0] + 1] - degree_offset[result[0]];

        // 这里涉及到中间点的存储问题，现在用的是这种稀疏存储的方式，需要压缩存储一下。
        for (result[1] = partial_result[1][0]; result[1] < partial_result_length[1]; result[1] = result[1]++)
        {
            int min_degree;
            int min_degree_vertex;
            int *min_array;
            find_min_degree(lid, result, 2, degree_offset, min_degree, min_degree_vertex);
            min_array = &adjcant[degree_offset[result[min_degree_vertex]]];
            // 可不可以先用寄存器保存下来，然后再删掉这个寄存器，重新分配一个寄存器。
            // 对array中的数组遍历
            for (int i = lid; i < min_degree; i = i + 32)
            {
                int item = min_array[lid];

                // 先进行剪枝，是否需要？后续修改一下
                // if (item.degree > 代码生成的degree)
                //     continue;

                // 代码生成多次hash_table的search
                // bucket_num这个变量能不能优化？
                if (search_in_hashtable(item, hash_tables + hash_tables_offset[result[min_degree_vertex]], hash_tables_offset[result[min_degree_vertex] + 1] - hash_tables_offset[result[min_degree_vertex]], bucket_num))
                    continue;
                // 有一个问题？item存在哪里？下次取第一个
                partial_result[2][i] = item;
                sum++;
            }
            // 由于最后一个点和第三个点对称，因此只需要从第三个点中找不同即可。
        }
        // 这一部分是负责取新的点的部分
    }
    
}
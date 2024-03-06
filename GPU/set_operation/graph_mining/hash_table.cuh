#ifndef HT_HEADER
#define HT_HEADER

#include <iostream>
#include <string>
#include <cub/cub.cuh>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "common.cuh"

using namespace std;

__global__ void buildHashTableLensKernel(int *hash_tables_lens, int *degree_offset, int vertex_count, float load_factor, int bucket_size);

__global__ void buildHashTableKernel(long long *hash_tables_offset, int *hash_tables, int *adjcant, int *vertex, int edge_count, long long bucket_num, int bucket_size);

tuple<long long *, int *, long long> buildHashTable(int *d_adjcant, int *d_vertex, int *d_csr_row_value);

__inline__ __device__ void swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}

#endif

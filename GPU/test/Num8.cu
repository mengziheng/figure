#include <stdio.h>
#include <iostream>

/* --- PRINTF_BYTE_TO_BINARY macro's --- */
#define PRINTF_BINARY_PATTERN_INT8 "%c%c%c%c%c%c%c%c"
#define PRINTF_BYTE_TO_BINARY_INT8(i) \
    (((i)&0x80ll) ? '1' : '0'),       \
        (((i)&0x40ll) ? '1' : '0'),   \
        (((i)&0x20ll) ? '1' : '0'),   \
        (((i)&0x10ll) ? '1' : '0'),   \
        (((i)&0x08ll) ? '1' : '0'),   \
        (((i)&0x04ll) ? '1' : '0'),   \
        (((i)&0x02ll) ? '1' : '0'),   \
        (((i)&0x01ll) ? '1' : '0')

#define PRINTF_BINARY_PATTERN_INT16 \
    PRINTF_BINARY_PATTERN_INT8 PRINTF_BINARY_PATTERN_INT8
#define PRINTF_BYTE_TO_BINARY_INT16(i) \
    PRINTF_BYTE_TO_BINARY_INT8((i) >> 8), PRINTF_BYTE_TO_BINARY_INT8(i)
#define PRINTF_BINARY_PATTERN_INT32 \
    PRINTF_BINARY_PATTERN_INT16 PRINTF_BINARY_PATTERN_INT16
#define PRINTF_BYTE_TO_BINARY_INT32(i) \
    PRINTF_BYTE_TO_BINARY_INT16((i) >> 16), PRINTF_BYTE_TO_BINARY_INT16(i)

__global__ void vote_ballot(int *a, int *b, int n)
{
    int tid = threadIdx.x;
    if (tid >= n)
    {
        return;
    }
    int temp = a[tid];
    b[tid] = __ballot_sync(0xFFFFFFFF, temp);
    int t = 14;
    int d = __ffs(t);
    if (threadIdx.x == 1)
    {
        printf(
            PRINTF_BINARY_PATTERN_INT32,
            PRINTF_BYTE_TO_BINARY_INT32(b[tid]));
        printf("\n");
        printf("%d\n", b[tid]);
        b[tid]  ^= 1UL << 5;
        printf(
            PRINTF_BINARY_PATTERN_INT32,
            PRINTF_BYTE_TO_BINARY_INT32(b[tid]));
        printf("\n");
        printf("%d\n", b[tid]);
    }
}

int main()
{
    int *h_a, *h_b, *d_a, *d_b;
    int n = 32;
    int nsize = n * sizeof(int);
    h_a = (int *)malloc(nsize);
    h_b = (int *)malloc(nsize);
    for (int i = 0; i < n; ++i)
    {
        h_a[i] = i;
    }
    memset(h_b, 0, nsize);
    cudaMalloc(&d_a, nsize);
    cudaMalloc(&d_b, nsize);
    cudaMemcpy(d_a, h_a, nsize, cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, nsize);
    vote_ballot<<<1, 256>>>(d_a, d_b, n);
    cudaMemcpy(h_b, d_b, nsize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
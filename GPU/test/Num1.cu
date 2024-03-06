#include <cstdint>
#include <iostream>
#include <vector>

#include <cooperative_groups.h>

__global__ void kernel(uint32_t values[], int b)
{
  using namespace cooperative_groups;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 5)
  {
    grid_group g = this_grid();
    printf("num : %lld",g.num_threads());
    g.sync();
    printf("im %d %lld b : %d\n", threadIdx.x, g.thread_rank(), b);
    g.sync();
  }
}

int main()
{
  constexpr uint32_t kNum = 1;
  std::vector<uint32_t> h_values(kNum);
  uint32_t *d_values;
  int b = 2;

  cudaMalloc(&d_values, sizeof(uint32_t) * kNum);
  cudaMemcpy(d_values, h_values.data(), sizeof(uint32_t) * kNum, cudaMemcpyHostToDevice);

  void *params[] = {&d_values, &b};

  cudaLaunchCooperativeKernel((void *)kernel, 2, 4, params);

  cudaMemcpy(h_values.data(), d_values, sizeof(uint32_t) * kNum, cudaMemcpyDeviceToHost);

  cudaFree(d_values);
  cudaDeviceSynchronize();
  return 0;
}
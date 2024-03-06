#include <iostream>
#include <vector>
#include <ctime>
#include <tuple>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <climits>

#define LOAD_FACTOR 0.25
#define BUCKET_SIZE 4
#define MIN_BUCKET_NUM 32
#define map_value(hash_value, bucket_num) (((hash_value) % MIN_BUCKET_NUM) * (bucket_num / MIN_BUCKET_NUM) + (hash_value) / MIN_BUCKET_NUM)
#define BLOCK_NUM 216
#define BLOCK_SIZE 1024

struct HT
{
  int *hash_table;
  int hash_table_size;
};

// 输入一个pAddr的地址，在函数内部判断其的值是否与期望值nExpected相等
// 如果相等那么就将pAddr的值改为nNew并同时返回true；否则就返回false，什么都不做
bool compare_and_swap(int *pAddr, int nExpected, int nNew)
{
  if (*pAddr == nExpected)
  {
    *pAddr = nNew;
    return true;
  }
  else
    return false;
}

bool hash_compare(int a, int b)
{
  return (a % MIN_BUCKET_NUM) < (b % MIN_BUCKET_NUM); // 升序排列
}

// 按列存储
HT build_hash(int *array, int size, bool opt = false)
{
  int *hash_table;
  int hash_table_size, log, bucket_num;

  // 确定bucket num
  bucket_num = (int)(size / LOAD_FACTOR / BUCKET_SIZE);
  log = log2f(bucket_num);
  bucket_num = powf(2, log) == bucket_num ? bucket_num : powf(2, log + 1);
  if (bucket_num < MIN_BUCKET_NUM && bucket_num != 0)
    bucket_num = MIN_BUCKET_NUM;

  // printf("array size : %d, bucket number : %d\n", size, bucket_num);

  hash_table = (int *)malloc(sizeof(int) * BUCKET_SIZE * bucket_num);
  hash_table_size = bucket_num * BUCKET_SIZE;
  memset(hash_table, -1, sizeof(int) * BUCKET_SIZE * bucket_num);

  // 按列构建hash table
  for (int i = 0; i < size; i++)
  {
    int item = array[i];
    int bucket_id = item % bucket_num;
    if (opt)
      bucket_id = map_value(bucket_id, bucket_num);
    int index = 0;
    while (compare_and_swap(&hash_table[bucket_id + bucket_num * index], -1, item) == false)
    {
      index++;
      if (index == BUCKET_SIZE)
      {
        index = 0;
        bucket_id++;
        if (bucket_id == bucket_num)
          bucket_id = 0;
      }
    }
  }
  HT ht;
  ht.hash_table = hash_table;
  ht.hash_table_size = hash_table_size;
  return ht;
}

__device__ bool search_hash(int tid, int item, int *hash_table, int hash_table_size, bool opt, bool show)
{
  int bucket_num = hash_table_size / BUCKET_SIZE;
  int bucket_id = item % bucket_num;
  if (opt)
    bucket_id = map_value(bucket_id, bucket_num);
  int *cmp = hash_table + bucket_id;
  int index = 0;
  int time = 0;

  while (*cmp != -1)
  {
    if (*cmp == item)
    {
      return true;
    }

    time++;
    cmp = cmp + bucket_num;
    index++;
    if (index == BUCKET_SIZE)
    {
      bucket_id++;
      index = 0;
      if (bucket_id == bucket_num)
        bucket_id = 0;
      cmp = &hash_table[bucket_id];
    }
  }
  return false;
}

__global__ void intersection_kernel(int *probe_array, int probe_array_size, int *hash_table, int hash_table_size, int *output, bool show)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = tid; i < probe_array_size; i = i + blockDim.x * gridDim.x)
  {
    int item = probe_array[i];
    bool exist = true;
    exist = search_hash(tid, item, hash_table, hash_table_size, false, show);
    if (exist)
      output[i] = item;
    else
      output[i] = -1;
  }
}

__global__ void intersection_kernel_opt(int *probe_array, int probe_array_size, int *hash_table, int hash_table_size, int *output, bool show)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = tid; i < probe_array_size; i = i + blockDim.x * gridDim.x)
  {
    int item = probe_array[i];
    bool exist = true;
    exist = search_hash(tid, item, hash_table, hash_table_size, true, show);
    if (exist)
      output[i] = item;
    else
      output[i] = -1;
  }
}

std::tuple<int, int *> intersection(int *x, int *y, int size_x, int size_y, bool opt = false, bool show = false)
{
  int *probe_array = (size_x < size_y) ? x : y;
  int *build_array = (size_x >= size_y) ? x : y;
  int probe_size = (size_x < size_y) ? size_x : size_y;
  int build_size = (size_x >= size_y) ? size_x : size_y;

  if (opt)
    std::sort(probe_array, probe_array + probe_size, hash_compare);

  // printf("begin build hash table\n");
  HT ht = build_hash(build_array, build_size, opt);
  // printf("finish build hash table\n");
  int *hash_table = ht.hash_table;
  int hash_table_size = ht.hash_table_size;

  int *d_probe_array, *d_build_hash_table, *d_output;
  cudaMalloc(&d_probe_array, sizeof(int) * probe_size);
  cudaMalloc(&d_build_hash_table, sizeof(int) * hash_table_size);
  cudaMalloc(&d_output, sizeof(int) * probe_size);
  cudaMemcpy(d_probe_array, probe_array, sizeof(int) * probe_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_build_hash_table, hash_table, sizeof(int) * hash_table_size, cudaMemcpyHostToDevice);

  // printf("begin intersection\n");
  float kernel_time;
  if (opt)
  {
    float time_start = clock();
    intersection_kernel_opt<<<BLOCK_NUM, BLOCK_SIZE>>>(d_probe_array, probe_size, d_build_hash_table, hash_table_size, d_output, show);
    cudaDeviceSynchronize();
    kernel_time = (clock() - time_start) / CLOCKS_PER_SEC;
  }
  else
  {
    float time_start = clock();
    intersection_kernel<<<BLOCK_NUM, BLOCK_SIZE>>>(d_probe_array, probe_size, d_build_hash_table, hash_table_size, d_output, show);
    cudaDeviceSynchronize();
    kernel_time = (clock() - time_start) / CLOCKS_PER_SEC;
  }

  int *output = new int[probe_size];
  cudaMemcpy(output, d_output, sizeof(int) * probe_size, cudaMemcpyDeviceToHost);
  std::sort(output, output + probe_size);
  if (show)
  {
    if (opt)
      printf("\nintersection result after optimal GPU intersection : \n");
    else
      printf("\nintersection result after GPU intersection : \n");
    int i;
    for (i = 0; i < probe_size; i++)
    {
      if (output[i] != -1)
        break;
    }
    output = output + i;
    std::cout << "kernel time: " << kernel_time * 1000 << " ms" << std::endl;
    return std::make_tuple(probe_size - i, output);
  }
  return std::make_tuple((int)NULL, (int *)NULL);
}

int *generate_random_array(int size, int range)
{
  // 设置随机数种子，以确保每次生成的数组都不同
  std::random_device rd;                                     // 获取随机设备种子
  std::mt19937 generator(rd());                              // 使用 Mersenne Twister 伪随机数生成器
  std::uniform_int_distribution<int> distribution(0, range); // 生成在[min, max]范围内的均匀分布的整数

  int *randomArray = new int[size];

  for (int i = 0; i < size; ++i)
  {
    // 生成随机整数并将其添加到数组中
    randomArray[i] = distribution(generator);
  }
  return randomArray;
}

void show_array(int *arr, int size)
{
  for (int i = 0; i < size; i++)
    printf("%d ", arr[i]);
  printf("\n");
}

std::tuple<int, int *> intersection_loop(int *x, int *y, int size_x, int size_y)
{
  if (size_x >= size_y)
  {
    int *temp = y;
    y = x;
    x = temp;
    int t = size_x;
    size_x = size_y;
    size_y = t;
  }

  int *output = new int[size_x];
  for (int i = 0; i < size_x; i++)
    for (int j = 0; j < size_y; j++)
    {
      if (x[i] == y[j])
      {
        output[i] = x[i];
        break;
      }
      else
        output[i] = -1;
    }
  std::sort(output, output + size_x);
  printf("intersection result after loop intersection : \n");
  int i;
  for (i = 0; i < size_x; i++)
  {
    if (output[i] != -1)
      break;
  }
  output = output + i;
  return std::make_tuple(size_x - i, output);
}

bool array_are_equal(int arr1[], int arr2[], int size)
{
  for (int i = 0; i < size; i++)
  {
    if (arr1[i] != arr2[i])
    {
      return false; // 发现不相等的元素，返回false
    }
  }
  return true; // 所有元素都相等，返回true
}

__global__ void warm_up_gpu()
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " <num1> <num2>\n";
    return 1;
  }

  // 将输入的参数转换为整数
  int size_x = std::atoi(argv[1]);
  int size_y = std::atoi(argv[2]);
  int range_x = size_y * 2, range_y = size_y * 2;

  cudaSetDevice(0);
  // int size_x = 1024 * 256, range_x = 1024 * 1024 * 1024;
  // int size_y = 512 * 1024 * 1024, range_y = 1024 * 1024 * 1024;

  // int size_x = 256, range_x = 2048;
  // int size_y = 700, range_y = 2048;

  // int size_x = 10, range_x = 40;
  // int size_y = 20, range_y = 40;

  int *x = generate_random_array(size_x, range_x);
  int *y = generate_random_array(size_y, range_y);

  // show_array(x, size_x);
  // show_array(y, size_y);
  std::tuple<int, int *> result_loop = intersection_loop(x, y, size_x, size_y);
  printf("size is : %d\n", std::get<0>(result_loop));

  std::tuple<int, int *> result_gpu = intersection(x, y, size_x, size_y, false, true);
  printf("size is : %d\n", std::get<0>(result_gpu));
  if (array_are_equal(std::get<1>(result_loop), std::get<1>(result_gpu), std::get<0>(result_loop)))
    printf("intersection correct for GPU_version!\n");

  std::tuple<int, int *> result_opt = intersection(x, y, size_x, size_y, true, true);
  printf("size is : %d\n", std::get<0>(result_opt));
  if (array_are_equal(std::get<1>(result_loop), std::get<1>(result_opt), std::get<0>(result_loop)))
    printf("intersection correct for OPT_version!\n");

  cudaDeviceSynchronize();

  return 0;
}
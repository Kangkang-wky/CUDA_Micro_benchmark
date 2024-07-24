#include "util.h"
#include <cstdint>
#include <cstdio>

__global__ void smem_1(uint32_t *a)
{
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++)
  {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 15 || tid == 16)
  {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}

__global__ void smem_2(uint32_t *a)
{
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++)
  {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 0 || tid == 15)
  {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}

__global__ void smem_3(uint32_t *a)
{
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++)
  {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
      smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}

__global__ void smem_4(uint32_t *a)
{
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++)
  {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr;
  if (tid < 16)
  {
    addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
  }
  else
  {
    addr = (tid / 8) * 2 + ((tid % 8) % 2);
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

// 注意 smem_5
// thread 0 - 3 访问第 0 个 uint4， thread 4 - 7 访问第 8 个
// uint4（到了第二行）; thread 8 - 11 访问第 1 个 uint4， thread 12 - 15 访问第
// 9 个 uint4（到了第二行）;

// 这里符合合并条件 1, 所以前两个和后两个 quarter warp 分别合并。但是每个 half
// warp 内，产生了 2-way bank conflict，所以需要拆成 2 次 transaction。 即一共 2
// 个 bank conflict， 4 次 transaction.

__global__ void smem_5(uint32_t *a)
{
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++)
  {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
      smem)[(tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8];
}

// 前两个 quarter warp 与 后两个 quarter warp 的行为不一致
__global__ void smem_6(uint32_t *a)
{
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++)
  {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 16) * 4 + (tid % 16 / 8) * 8;
  if (tid < 16)
  {
    addr += (tid % 4 / 2) * 2;
  }
  else
  {
    addr += (tid % 4 % 2) * 2;
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

__global__ void smem_7(uint32_t *a)
{
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++)
  {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = tid;

  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

// V8 增加每个线程处理的元素个数，尝试对一个32*32的进行更多列的同时处理                         // 发现32*2会有bank conflict，32*4以上没有bank conflict
template <int NUM_PER_THREAD>
__global__ void mat_transpose_kernel_v8(const float *idata, float *odata, int M, int N)
{
  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;

  __shared__ float sdata[32][33];

  int x = bx * 32 + tx;
  int y = by * 32 + ty;

  constexpr int ROW_STRIDE = 32 / NUM_PER_THREAD;

  if (x < N)
  {
    // #pragma unroll // 就算不循环展开也没有bank conflict
    for (int y_off = 0; y_off < 32; y_off += ROW_STRIDE)
    { // 每个线程处理4个元素，间隔为8
      if (y + y_off < M)
      {
        sdata[ty + y_off][tx] = idata[(y + y_off) * N + x]; // sts
      }
    }
  }
  __syncthreads();

  x = by * 32 + tx;
  y = bx * 32 + ty;
  if (x < M)
  {
    for (int y_off = 0; y_off < 32; y_off += ROW_STRIDE)
    {
      if (y + y_off < N)
      {
        odata[(y + y_off) * M + x] = sdata[tx][ty + y_off]; // lds
      }
    }
  }
}

__global__ void global_test1(uint32_t *d_b)
{
  int a = d_b[blockIdx.x * blockDim.x + threadIdx.x];
  d_b[blockIdx.x * blockDim.x + threadIdx.x] = (a + 1) / 64 * 8 + (((a + 1) / 8) % 8) * 32 + (a + 1) % 8;
}

__global__ void global_test2(uint32_t *d_b)
{

  int index = blockIdx.x * blockDim.x + (threadIdx.x / 64) * 8 + ((threadIdx.x / 8) % 8) * 32 + threadIdx.x % 8;

  int a = d_b[index];
  d_b[index] = a + 1;
}

__global__ void global_test3(uint32_t *d_b)
{
  int a = d_b[blockIdx.x * blockDim.x + threadIdx.x];
  d_b[blockIdx.x * blockDim.x + threadIdx.x] = a + 1;
}

__global__ void global_test4(uint32_t *d_b)
{
  int index = blockIdx.x * blockDim.x + (threadIdx.x / 64) * 8 + ((threadIdx.x / 8) % 8) * 32 + threadIdx.x % 8;
  if (threadIdx.x == 4 || threadIdx.x == 5 || threadIdx.x == 6 || threadIdx.x == 7)
    index += 32;
  if (threadIdx.x == 12 || threadIdx.x == 13 || threadIdx.x == 14 || threadIdx.x == 15)
    index -= 32;
  int a = d_b[index];
  d_b[index] = a + 1;
}

int main()
{
  uint32_t *d_a;
  uint32_t *d_b;
  cudaMalloc(&d_a, sizeof(uint32_t) * 128);
  cudaMalloc(&d_b, sizeof(uint32_t) * 128 * 2 * 1024 * 1024);
  smem_1<<<1, 32>>>(d_a);
  smem_2<<<1, 32>>>(d_a);
  smem_3<<<1, 32>>>(d_a);
  smem_4<<<1, 32>>>(d_a);
  smem_5<<<1, 32>>>(d_a);
  smem_6<<<1, 32>>>(d_a);
  smem_7<<<1, 32>>>(d_a);

  global_test1<<<1024 * 1024, 256>>>(d_b);
  global_test2<<<1024 * 1024, 256>>>(d_b);
  global_test3<<<1024 * 1024, 256>>>(d_b);
  global_test4<<<1024 * 1024, 256>>>(d_b);

  cudaFree(d_a);
  cudaFree(d_b);

  float *idata,
      *odata;
  int M = 8192, N = 8192;
  cudaMalloc(&idata, sizeof(float) * M);
  cudaMalloc(&odata, sizeof(float) * N);

  constexpr int NUM_PER_THREAD = 8;
  // dim3 block(32, 32 / NUM_PER_THREAD); // 8 * 32 = 256
  // dim3 grid(256, 256);                 // 256 * 256 = 65536
  // mat_transpose_kernel_v8<NUM_PER_THREAD><<<grid, block>>>(idata, odata, M, N);

  cudaDeviceSynchronize();
  cudaFree(idata);
  cudaFree(odata);
  return 0;
}
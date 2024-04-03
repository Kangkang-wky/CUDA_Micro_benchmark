#include <cstdint>
#include <cstdio>

__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 15 || tid == 16) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}

__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 0 || tid == 15) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}

__global__ void smem_3(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
      smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}

__global__ void smem_4(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr;
  if (tid < 16) {
    addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
  } else {
    addr = (tid / 8) * 2 + ((tid % 8) % 2);
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

// 注意 smem_5
// thread 0 - 3 访问第 0 个 uint4， thread 4 - 7 访问第 8 个 uint4（到了第二行）;
// thread 8 - 11 访问第 1 个 uint4， thread 12 - 15 访问第 9 个 uint4（到了第二行）;

// 这里符合合并条件 1, 所以前两个和后两个 quarter warp 分别合并。但是每个 half warp 内，产生了 2-way bank conflict，所以需要拆成 2 次 transaction。
// 即一共 2 个 bank conflict， 4 次 transaction.

__global__ void smem_5(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[(tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8];
}

// 前两个 quarter warp 与 后两个 quarter warp 的行为不一致
__global__ void smem_6(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 16) * 4 + (tid % 16 / 8) * 8;
  if (tid < 16) {
    addr += (tid % 4 / 2) * 2;
  } else {
    addr += (tid % 4 % 2) * 2;
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

int main() {
  uint32_t *d_a;
  cudaMalloc(&d_a, sizeof(uint32_t) * 128);
  smem_1<<<1, 32>>>(d_a);
  smem_2<<<1, 32>>>(d_a);
  smem_3<<<1, 32>>>(d_a);
  smem_4<<<1, 32>>>(d_a);
  smem_5<<<1, 32>>>(d_a);
  smem_6<<<1, 32>>>(d_a);
  cudaFree(d_a);
  cudaDeviceSynchronize();
  return 0;
}
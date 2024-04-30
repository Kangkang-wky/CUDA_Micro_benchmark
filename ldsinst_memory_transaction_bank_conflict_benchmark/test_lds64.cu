#include <cstdint>

// lds64 指令测试 benchmark, 以下均测试对齐访问
// half-warp 先访问一半试试, 一半线程活跃, 16线程 x lds64(8 bytes) = 128 bytes =
// 1 个 memory transaction
__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid < 16) {
    reinterpret_cast<uint2 *>(a)[tid] =
        reinterpret_cast<const uint2 *>(smem)[tid];
  }
}

// 两个 half-warp, 访问的是不同的 cacheline, memory transaction 内是没有 bank
// conflict 的 两次 memory transaction, 并且是没有 bank conflict
__global__ void smem_1_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid];
}

// 两个 half-warp, 需要两个 wavefront
__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid < 15 || tid == 16) {
    reinterpret_cast<uint2 *>(a)[tid] =
        reinterpret_cast<const uint2 *>(smem)[tid == 16 ? 15 : tid];
  }
}

// 两个 half-warp, 触发第一个广播机制, 只有一个 memory transaction 发生,
// 不会触发 bank conflict
__global__ void smem_3(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid / 2];
}

// 两个 half-warp, 第一个 half-warp 触发第一条, 第二个 half-warp 触发第二条
// 需要两个 memory transaction
__global__ void smem_4(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr;
  if (tid < 16) {
    addr = tid / 2;
  } else {
    addr = (tid / 4) * 4 + (tid % 4) % 2;
  }
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[addr];
}

// 触发两次 memory transaction 两个 wavefronts
__global__ void smem_5(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid % 16];
}

// lds64 指令
__global__ void smem_6(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid];
}

int main() {
  uint32_t *d_a;
  cudaMalloc(&d_a, sizeof(uint32_t) * 128);
  // micro benchmark
  smem_1<<<1, 32>>>(d_a);
  smem_1_2<<<1, 32>>>(d_a);
  smem_2<<<1, 32>>>(d_a);
  smem_3<<<1, 32>>>(d_a);
  smem_4<<<1, 32>>>(d_a);
  smem_5<<<1, 32>>>(d_a);
  smem_6<<<1, 32>>>(d_a);
  cudaFree(d_a);
  cudaDeviceSynchronize();
  return 0;
}
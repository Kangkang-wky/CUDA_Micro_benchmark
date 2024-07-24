#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
// #include <cublas_v2.h>

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

void mat_transpose_v8(const float *idata, float *odata, int M, int N)
{ // M = N = 8192
    // constexpr int BLOCK_SZ = 32;
    constexpr int NUM_PER_THREAD = 8;
    dim3 block(32, 32 / NUM_PER_THREAD); // 8 * 32 = 256
    dim3 grid(256, 256);                 // 256 * 256 = 65536
    mat_transpose_kernel_v8<NUM_PER_THREAD><<<grid, block>>>(idata, odata, M, N);
}

int main()
{
    int M = 8192;
    int N = 8192;

    size_t bytes = M * N * sizeof(float);
    float *h_idata = (float *)malloc(bytes); // Host input matrix
    float *h_odata = (float *)malloc(bytes); // Host output matrix

    // Initialize the input matrix
    srand(time(0)); // 初始化随机数生成器
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            h_idata[i * N + j] = static_cast<float>(rand()) / RAND_MAX; // Fill with some values
        }
    }

    // Print part of the original matrix to verify initial state
    printf("Part of the original matrix:\n");
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            printf("%f ", h_idata[i * N + j]);
        }
        printf("\n");
    }

    float *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, bytes);

    // Copy data from host to device
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    // Setup timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    mat_transpose_v8(d_idata, d_odata, M, N);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copy result back to host
    cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Transposed Matrix (Part):\n");
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            printf("%f ", h_odata[i * N + j]);
        }
        printf("\n");
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for matrix transpose: %f milliseconds\n", milliseconds);
    double bandwidth = pow(2, 29) / (milliseconds * 1e6); // 8192*8192*4B*2
    printf("Memory bandwidth: %f GB/s\n", bandwidth);
    printf("Memory bandwidth usage rate: %.2f%% \n", bandwidth * 100 / 256);

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

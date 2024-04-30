#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include "cuda_runtime.h"
#include "cudnn.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>

// runtime error 检查
// cuda runtime error
#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t cuda_error = (func);                                           \
    if (cuda_error != cudaSuccess)                                             \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__,                           \
             cudaGetErrorString(cuda_error));                                  \
  }

// cudnn runtime error
#define CHECK_CUDNN(func)                                                      \
  {                                                                            \
    cudnnStatus_t cudnn_status = (func);                                       \
    if (cudnn_status != CUDNN_STATUS_SUCCESS)                                  \
      printf("%s %d CUDNN: %s\n", __FILE__, __LINE__,                          \
             cudnnGetErrorString(cudnn_status));                               \
  }

// cublas runtime error
#define CHECK_CUBLAS(func)                                                     \
  {                                                                            \
    cublasStatus_t cublas_status = (func);                                     \
    if (cublas_status != CUBLAS_STATUS_SUCCESS)                                \
      printf("%s %d CUBLAS: %s\n", __FILE__, __LINE__,                         \
             cublasGetErrorString(cublas_status));                             \
  }

#endif // _CUDA_UTILS_H_
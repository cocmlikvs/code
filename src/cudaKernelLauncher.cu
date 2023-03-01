#include <stdio.h>
#include <cuda_runtime.h>

void cudaCheckError(cudaError_t error, const char* file, const int line) {
  if (error != cudaSuccess) {
    printf("CUDA error (%s:%d): %s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK_ERROR(error) cudaCheckError(error, __FILE__, __LINE__)

void cudaLaunchKernel(void (*func)(void*), const void* args, const int blockSize, const int numBlocks) {
  func<<<numBlocks, blockSize>>>(args);
  CUDA_CHECK_ERROR(cudaPeekAtLastError());
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

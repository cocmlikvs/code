#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_runtime.h>
#include <murmur3_hash.h>

constexpr int kNumBits = 1024 * 1024 * 8;  // 8 MB
constexpr int kNumHashes = 4;
constexpr int kBlockSize = 256;

__device__ uint32_t MurmurHash3(const void* key, const int len, const uint32_t seed) {
  const uint8_t* data = static_cast<const uint8_t*>(key);
  const int nblocks = len / 4;

  uint32_t h1 = seed;

  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  const uint32_t* blocks = reinterpret_cast<const uint32_t*>(data + nblocks * 4);
  for (int i = -nblocks; i; i++) {
    uint32_t k1 = blocks[i];

    k1 *= c1;
    k1 = (k1 << 15) | (k1 >> (32 - 15));
    k1 *= c2;

    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >> (32 - 13));
    h1 = h1 * 5 + 0xe6546b64;
  }

  const uint8_t* tail = reinterpret_cast<const uint8_t*>(data + nblocks * 4);
  uint32_t k1 = 0;
  switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
      [[fallthrough]];
    case 2:
      k1 ^= tail[1] << 8;
      [[fallthrough]];
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = (k1 << 15) | (k1 >> (32 - 15));
      k1 *= c2;
      h1 ^= k1;
      break;
  }

  h1 ^= len;
  h1 ^= h1 >> 16;
  h1 *= 0x85ebca6b;
  h1 ^= h1 >> 13;
  h1 *= 0xc2b2ae35;
  h1 ^= h1 >> 16;

  return h1;
}

__global__ void HashCounterKernel(const uint8_t* data, const int size, const int num_hashes,
                                   const uint32_t* seeds, uint8_t* filter) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = index; i < size; i += stride) {
    const uint32_t hash = MurmurHash3(data + i, size - i, seeds[0]);
    for (int j = 1; j < num_hashes; j++) {
      const uint32_t newHash = MurmurHash3(data + i, size - i, seeds[j]);
      if (newHash < hash) {
        hash = newHash;
      }
    }

    const uint32_t bitIndex = hash % (kNumBits * 8);
    const uint32_t byteIndex = bitIndex / 8;
    const uint32_t bitOffset = bitIndex % 8;

    atomicAdd(&filter[byteIndex], 1 << bitOffset);
  }
}

int main() {
  // Allocate memory on host and device
  constexpr int kInputSize = 1024;
  uint8_t* data = new uint8_t[kInputSize];
  for (int i = 0; i < kInputSize; i++) {
    data[i] = static_cast<uint8_t>(i % 256);
  }

  uint8_t* d_data;
  cudaMalloc(&d_data, kInputSize * sizeof(uint8_t));
  cudaMemcpy(d_data, data, kInputSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Generate seeds for hash functions
  uint32_t seeds[kNumHashes];
  for (int i = 0; i < kNumHashes; i++) {
    seeds[i] = static_cast<uint32_t>(i + 1);
  }

  // Allocate memory for hash counter on host and device
  const int filterSize = (kNumBits + 7) / 8;
  uint8_t* filter = new uint8_t[filterSize];
  memset(filter, 0, filterSize);
  uint8_t* d_filter;
  cudaMalloc(&d_filter, filterSize * sizeof(uint8_t));
  cudaMemcpy(d_filter, filter, filterSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Compute hash values and update hash counter
  const int numBlocks = (kInputSize + kBlockSize - 1) / kBlockSize;
  HashCounterKernel<<<numBlocks, kBlockSize>>>(d_data, kInputSize, kNumHashes, seeds, d_filter);

  // Copy hash counter from device to host
  cudaMemcpy(filter, d_filter, filterSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  // Verify hash counter by checking if it contains all the elements
  bool allPresent = true;
  for (int i = 0; i < kInputSize; i++) {
    const uint32_t hash = MurmurHash3(data + i, kInputSize - i, seeds[0]);
    bool present = true;
    for (int j = 1; j < kNumHashes; j++) {
      const uint32_t newHash = MurmurHash3(data + i, kInputSize - i, seeds[j]);
      if (newHash < hash) {
        present = false;
        break;
      }
    }

    const uint32_t bitIndex = hash % (kNumBits * 8);
    const uint32_t byteIndex = bitIndex / 8;
    const uint32_t bitOffset = bitIndex % 8;

    if ((filter[byteIndex] & (1 << bitOffset)) == 0) {
      allPresent = false;
      break;
    }
  }

  if (allPresent) {
    std::cout << "All elements are present in hash counter\n";
  } else {
    std::cout << "Not all elements are present in hash counter\n";
  }

  // Free memory
  delete[] data;
  delete[] filter;
  cudaFree(d_data);
  cudaFree(d_filter);

  return 0;
}


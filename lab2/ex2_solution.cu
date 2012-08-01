#include <iostream>

// Everything done by Rolf Andreassen!

using namespace std;

__device__ bool* syncArray = 0;

__device__ void device_vector_reduce_blocks_recursive (double* toBeReduced, int workingLength) {

  syncArray[blockDim.x] = false;

  // First reduce this block

  // Copy from global to shared memory for local reduction

  __shared__ double localResults[1024];

  int localIndex = blockDim.x * blockIdx.x + threadIdx.x;

  localResults[threadIdx.x] = toBeReduced[localIndex];

  // Reduce local block of (at most) 1024 entries

  int len = blockDim.x;

  if (len > workingLength) len = workingLength;

  while (len > 1) {

    if ((localIndex < workingLength) && (threadIdx.x < (len - (len % 2)) / 2)) {

      localResults[threadIdx.x] += localResults[threadIdx.x + (len + (len % 2)) / 2];

    }

    len = (len + (len % 2)) / 2;

    __syncthreads();

  }

 

  // Need to synchronise over blocks! Otherwise we may overwrite

  // data another block is still working on.

  syncArray[blockDim.x] = true;

  bool everyoneDone = false;

  while (!everyoneDone) {

    everyoneDone = true;

    for (int i = 0; i < blockDim.x; ++i) {

      if (syncArray[i]) continue;

      everyoneDone = false;

      break;

    }

  }

 

  if ((0 == threadIdx.x) && (localIndex < workingLength)) toBeReduced[blockIdx.x] = localResults[threadIdx.x];

  // First blockDim.x entries of toBeReduced are now sums for individual blocks

 

  // Now repeat reduction just on the first part of the vector.

  int newWorkingLength = (workingLength + blockDim.x - 1) / blockDim.x;

  if (newWorkingLength > 1) device_vector_reduce_blocks_recursive(toBeReduced, newWorkingLength);

}

__global__ void device_vector_reduce_blocks (double* toBeReduced, int workingLength) {
  // Global function cannot recurse, outsource actual algorithm to device function
  device_vector_reduce_blocks_recursive(toBeReduced, workingLength);
}

int main (int argc, char** argv) {
  int sizeOfVector = 100;
  if (argc > 1) sizeOfVector = atoi(argv[1]);

  double* host_numbers = new double[sizeOfVector];
  double checkSum = 0;
  srand(42);
  for (int i = 0; i < sizeOfVector; ++i) {
    host_numbers[i] = rand() % 10;
    checkSum += host_numbers[i];
  }

  std::cout << "CPU result: " << checkSum << std::endl;
  double* dev_numbers;
  cudaMalloc((void**) &dev_numbers, sizeOfVector*sizeof(double));
  cudaMemcpy(dev_numbers, host_numbers, sizeOfVector*sizeof(double), cudaMemcpyHostToDevice);

  int numThreads = min(1024, sizeOfVector);
  int numBlocks  = (1023 + sizeOfVector) / 1024;
  std::cout << "Blocks and threads: " << numBlocks << " " << numThreads << std::endl;
  cudaMalloc((void**) syncArray, numBlocks * sizeof(bool));

  device_vector_reduce_blocks<<<numBlocks, numThreads>>>(dev_numbers, sizeOfVector);
  cudaDeviceSynchronize(); // Ensure that kernel is done before copying result
  cudaMemcpy(&checkSum, dev_numbers, sizeof(double), cudaMemcpyDeviceToHost);

  std::cout << "GPU result: " << checkSum << std::endl;

  return 0;
}
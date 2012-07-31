#include <iostream>
#include "stdio.h"
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda.h>
#include <cassert>
// #include "/private/herten/NVIDIA_GPU_Computing_SDK/C/src/simplePrintf/cuPrintf.cuh"
#include "../cuPrintf.cuh"


__global__ void dotProduct (int* vec1, int* vec2, int* vec3, int streamId) {
	vec3[threadIdx.x] = vec1[threadIdx.x] * vec2[threadIdx.x];
	printf("[%i]-[%i] vec1 * vec2 = vec3 == %i * %i = %i\n", streamId, threadIdx.x, vec1[threadIdx.x], vec2[threadIdx.x], vec3[threadIdx.x]);
}


int main (int argc, char** argv) {
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	assert(properties.deviceOverlap);
	
	cudaStream_t stream0;
	cudaStream_t stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	
	int sizeOfVector = 100;
	int nOfChunks = 4;
	if (argc > 1) nOfChunks = atoi(argv[1]);
	int chunkSize = sizeOfVector/nOfChunks;
	std::cout << "Using " << nOfChunks << " chunks, leading to a chunk size of " << chunkSize << "." << std::endl;
	
	int *host_a, *host_b, *host_result;
	int *dev0_a, *dev0_b, *dev0_result;
	int *dev1_a, *dev1_b, *dev1_result;
	
	cudaHostAlloc((void**) &host_a, sizeOfVector*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_b, sizeOfVector*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**) &host_result, sizeOfVector*sizeof(int), cudaHostAllocDefault);
	
	cudaMalloc((void**) &dev0_a, chunkSize*sizeof(int));
	cudaMalloc((void**) &dev0_b, chunkSize*sizeof(int));
	cudaMalloc((void**) &dev0_result, chunkSize*sizeof(int));
	
	cudaMalloc((void**) &dev1_a, chunkSize*sizeof(int));
	cudaMalloc((void**) &dev1_b, chunkSize*sizeof(int));
	cudaMalloc((void**) &dev1_result, chunkSize*sizeof(int));
	
	srand(23);
	for (unsigned int i = 0; i < sizeOfVector; i++) {
		host_a[i] = rand() % 100;
		host_b[i] = rand() % 100;
	}
	for (unsigned int i = 0; i < sizeOfVector; i++) {
		std::cout << "host_a[" << i << "] = " << host_a[i] << ", host_b[" << i << "] = " << host_b[i] << std::endl;
	}
	
	int numBlocks = 1;
	int numThreads = chunkSize;
	
	for (int i = 0; i < sizeOfVector; i += 2*chunkSize) {
		cudaMemcpyAsync(dev0_a, host_a + i,
				chunkSize*sizeof(int),
				cudaMemcpyHostToDevice,
				stream0);
		cudaMemcpyAsync(dev1_a, host_a + chunkSize + i,
				chunkSize*sizeof(int),
				cudaMemcpyHostToDevice,
				stream1);
		
		cudaMemcpyAsync(dev0_b, host_b + i,
				chunkSize*sizeof(int),
				cudaMemcpyHostToDevice,
				stream0);
		cudaMemcpyAsync(dev1_b, host_b + chunkSize + i,
				chunkSize*sizeof(int),
				cudaMemcpyHostToDevice,
				stream1);
		
		dotProduct<<<numBlocks, numThreads, 0, stream0>>>(dev0_a, dev0_b, dev0_result, i);
		dotProduct<<<numBlocks, numThreads, 0, stream1>>>(dev1_a, dev1_b, dev1_result, i+1);
		
		cudaMemcpyAsync(host_result + i, dev0_result, chunkSize*sizeof(int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_result + chunkSize + i, dev1_result, chunkSize*sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}
	
	cudaThreadSynchronize(); // wait for all threads to finish
	
	for (int i = 0; i < sizeOfVector; i++) {
		std::cout << "host[" << i << "] = " << host_result[i] << std::endl;
	}
	
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFree(dev0_a);
	cudaFree(dev0_b);
	cudaFree(dev0_result);
	cudaFree(dev1_a);
	cudaFree(dev1_b);
	cudaFree(dev1_result);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
}

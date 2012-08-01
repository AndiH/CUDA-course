#include <iostream>
#include "stdio.h"
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda.h>
#include <cassert>
#include "book.h"
// #include "/private/herten/NVIDIA_GPU_Computing_SDK/C/src/simplePrintf/cuPrintf.cuh"
//#include "../cuPrintf.cuh"

struct ThreadInfo {
	int deviceToRunOn;
	thrust::host_vector<int> host_data_a;
	thrust::host_vector<int> host_data_b;
	int numData;
	int returnValue;
};

void * threadFunction(void* tinfo) {
	ThreadInfo * currData = (ThreadInfo*) tinfo;
	int nDev = currData->deviceToRunOn;
	cudaSetDevice(nDev);
	
	thrust::device_vector<int> localA = currData->host_data_a;
	thrust::device_vector<int> localB = currData->host_data_b;
// 	std::cout << "[" << nDev << "]: localA[0] = " << localA[0] << std::endl;
// 	std::cout << "[" << nDev << "]: localB[0] = " << localB[0] << std::endl;
	
	int dotProductResult = thrust::inner_product(localA.begin(), localA.end(), localB.begin(), 0);
// 	std::cout << "[" << nDev << "]: dot product = " << dotProductResult << std::endl;
	currData->returnValue = dotProductResult;
	return 0;
};

int main (int argc, char** argv) {
	int sizeOfVector = 100;
	int elementOne = 1;
	int elementTwo = 2;
	if (argc > 1) elementOne = atoi(argv[1]);
	if (argc > 1) elementTwo = atoi(argv[2]);
	
	thrust::host_vector<int> firstVectorA(sizeOfVector/2, elementOne);
	thrust::host_vector<int> firstVectorB(sizeOfVector/2, elementOne);
	thrust::host_vector<int> secondVectorA(sizeOfVector/2, elementTwo);
	thrust::host_vector<int> secondVectorB(sizeOfVector/2, elementTwo);
	
	
	ThreadInfo oneThread;
	oneThread.deviceToRunOn = 0;
	oneThread.host_data_a = firstVectorA;
	oneThread.host_data_b = secondVectorA;
	oneThread.numData = sizeOfVector / 2;
	
	ThreadInfo twoThread;
	twoThread.deviceToRunOn = 1;
	twoThread.host_data_a = firstVectorB;
	twoThread.host_data_b = secondVectorB;
	twoThread.numData = (1+sizeOfVector) / 2;
	
	CUTThread threadId = start_thread(&threadFunction, &oneThread);
	threadFunction(&twoThread);
	
	end_thread(threadId);
	int finalDotProduct = oneThread.returnValue + twoThread.returnValue;
	std::cout << "Dot product is: " << oneThread.returnValue << " + " << twoThread.returnValue << " = " << finalDotProduct << std::endl;
	
}

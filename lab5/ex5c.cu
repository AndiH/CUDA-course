#include <iostream>
#include "stdio.h"
#include <vector>
#include <cuda.h>
#include "math.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "../cuPrintf.cu"

struct PrintStruct {
	__device__ int operator() (int x) {
		printf("x = %i\n", x);
		return 0;
	}
	
	__device__ int operator() (thrust::tuple<int, int *> x) const {
		int x_1 =  thrust::get<0>(x);
		int * x_2 = thrust::get<1>(x);
		
		printf("x_1 = %i, x_2[%i] = %i\n", x_1, x_1, x_2[x_1]);
		return 0;
	}
	
	__device__ int operator() (thrust::tuple<int *, int> x) const {
		int position = thrust::get<1>(x);
		int * x_1 = thrust::get<0>(x) + 2 * position;
		int * x_2 = thrust::get<0>(x) + 2 * position + 1;
		
		printf("x_2a = %i \nx_2b = %i\n", x_1[0], x_2[0]);
		
		return 0;
	}
};

int main (int argc, char** argv) {
	// part 1
	int result = thrust::transform_reduce(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(10), PrintStruct(), 0, thrust::plus<int>());
	
	if (result != 0) std::cout << result << std::endl;
	
	
	// part 2
	int numberOfData = 10;
	srand(23);
	
	thrust::host_vector<int> h_vec(numberOfData);
	int h_array[numberOfData];
	for (int i = 0; i < numberOfData; i++) {
		h_vec[i] = rand() % 100;
		h_array[i] = rand() % 100;
	}
	int * d_array;
	cudaMalloc((void**) &d_array, numberOfData*sizeof(int*));
	cudaMemcpy(d_array, &h_array, numberOfData*sizeof(int*), cudaMemcpyHostToDevice);
	thrust::constant_iterator<int *> constIt(d_array);
	
	result = thrust::transform_reduce(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::make_counting_iterator<int>(0), 
				constIt
			)
		), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				thrust::make_counting_iterator<int>(numberOfData), 
				constIt
			)
		), 
		PrintStruct(), 
		0, 
		thrust::plus<int>()
	);
		
	if (result != 0) std::cout << result << std::endl;
	
	// part 3
	result = thrust::transform_reduce(
		thrust::make_zip_iterator(
			thrust::make_tuple(
				constIt, 
				thrust::make_counting_iterator<int>(0)
			)
		), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				constIt + numberOfData/2,
				thrust::make_counting_iterator<int>(numberOfData/2)
			)
		), 
		PrintStruct(), 
		0, 
		thrust::plus<int>()
	);
		
	if (result != 0) std::cout << result << std::endl;
	
	
	return 1;
}
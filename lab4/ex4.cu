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
// #include "/private/herten/NVIDIA_GPU_Computing_SDK/C/src/simplePrintf/cuPrintf.cuh"

__global__ void buggyAdd (int * numbers) {
	numbers[0]++;
}

__global__ void betterAdd(int * numbers) {
	atomicAdd(&numbers[0], 1);
}


// typedef thrust::tuple<thrust::constant_iterator<thrust::device_vector<int>* >, thrust::counting_iterator<int> > myTuple;

struct coarsePlus {
	int n;
	coarsePlus(int _n) : n(_n) {}
	__host__ __device__ double operator() (const thrust::tuple<thrust::device_vector<int>*, int> &thatTuple) const {
		double sum = 0;
		int factor = *thrust::get<1>(thatTuple);
		for (int i = n*factor; i < (2*n*factor - 1); i++) {
			sum += (**(thrust::get<0>(thatTuple)))[i];
		}
		return sum;
	} 
};

__global__ void potential_scatter (float* charge, float* grid) {
	float currentCharge = charge[threadIdx.x];
	int gridDimension = blockDim.x;
	
	for (int i = 0; i < 100; i++) { // 100 should actually be gridDimension
		float denominator = threadIdx.x - (i + 1); // thats the distance
		//FIX BY ROLF (HAS TO BE TESTED:)
		// float denominator = 1.0 / (1.0 + fabs(threadIdx.x - i));
		
		float _denominator = denominator;
		if (denominator < 0) _denominator = -denominator; // there is no abs() on cuda devices or something

		currentCharge /= _denominator;
// 		cuPrintf("Position %i, value1 = %f", i, grid[i]);
// 		atomicAdd(&grid[i],2);
// 		cuPrintf(", value2 = %f", grid[i]);
		atomicAdd(&grid[i], 2*currentCharge);
	}
}

int main (int argc, char** argv) {
	int* host_number = new int;
	host_number[0] = 10;
	
	std::cout << "source number = " << host_number[0] << std::endl;
	
	int* dev_number;
	cudaMalloc((void**) &dev_number, sizeof(int));
	
	std::vector<int> blocks, threads;
	blocks.push_back(1); blocks.push_back(1); blocks.push_back(2); blocks.push_back(10); blocks.push_back(100); blocks.push_back(65535);
	threads.push_back(1); threads.push_back(10); threads.push_back(450); threads.push_back(100); threads.push_back(1000); threads.push_back(1024);
	
	std::cout << "## Part 1: Simple (buggy) add: " << std::endl;
	for (unsigned int i = 0; i < blocks.size(); i++) {
		cudaMemcpy(dev_number, host_number, sizeof(int), cudaMemcpyHostToDevice);
		buggyAdd<<<blocks[i],threads[i]>>> (dev_number);
		cudaMemcpy(host_number, dev_number, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "  <<<" << blocks[i] << "," << threads[i] << ">>> = " << host_number[0] << std::endl;
		host_number[0] = 10;
	}
	
	std::cout << "## Part 2: Using atomicAdd()" << std::endl;
	for (unsigned int i = 0; i < blocks.size(); i++) {
		cudaMemcpy(dev_number, host_number, sizeof(int), cudaMemcpyHostToDevice);
		betterAdd<<<blocks[i],threads[i]>>> (dev_number);
		cudaMemcpy(host_number, dev_number, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "  <<<" << blocks[i] << "," << threads[i] << ">>> = " << host_number[0] << std::endl;
		host_number[0] = 10;
	}
	
	std::cout << "## Part 3: coarsePlus" << std::endl;
	
	int sizeOfVector = 100;
	
	thrust::host_vector<int> h_vec(sizeOfVector);
	srand(23);
	for (unsigned int i = 0; i < sizeOfVector; i++) {
		h_vec[i] = rand() % 100;
	}
	
	thrust::device_vector<int> d_vec = h_vec;
	
	
	
	thrust::constant_iterator<thrust::device_vector<int>* > constIt = thrust::make_constant_iterator(&d_vec);
	
	
	int result = thrust::transform_reduce(
		thrust::make_zip_iterator(
			thrust::make_tuple(
// 				thrust::make_constant_iterator(d_vec*), 
				constIt, // no constant iterator as a first element! because: start of a zip iterator is first element; end is, when first element again is reached (or something) - but with constant iterator it's always the same - so start = end - so invoked just once
					   // solution: switch them around: first part counting it, second part const it
				thrust::make_counting_iterator(0)
			)
		),
		thrust::make_zip_iterator(
			thrust::make_tuple(
// 				thrust::make_constant_iterator(d_vec*), 
				constIt,
				thrust::make_counting_iterator(0)+sizeOfVector
			)
		),
		coarsePlus(8), 
		0, 
		thrust::plus<int>());
	
	std::cout << "## Part 4: Coulomb potential" << std::endl;
	
	// Generating potential
	int nOfCharges = 100;
	float* charge = new float[nOfCharges];
	srand(23); 
	for (unsigned int i = 0; i < nOfCharges; i++) {
		charge[i] = rand() % 100;
		std::cout << "Charge at position " << i << " is equal to " << charge[i] << std::endl;
	}
	
	float* grid = new float[nOfCharges];
	for (unsigned int i = 0; i < nOfCharges; i++) {
		grid[i] = 5; // for debugging, real value is 0
	}

	float* dev_charge = 0;
	float* dev_grid = 0;
	
	cudaMalloc((void**) &dev_charge, nOfCharges * sizeof(float));
	cudaMalloc((void**) &dev_grid, nOfCharges * sizeof(float));
	
	cudaMemcpy(dev_charge, charge, nOfCharges * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_grid, grid, nOfCharges * sizeof(float), cudaMemcpyHostToDevice);

// 	cudaPrintfInit();
	potential_scatter<<<1,nOfCharges>>>(dev_charge, dev_grid);
// 	cudaPrintfDisplay(std::stdout, true);
// 	cudaPrintfEnd;
	
	cudaMemcpy(grid, dev_grid, nOfCharges * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (unsigned int i = 0; i < nOfCharges; i++) {
		std::cout << "Potential at position " << i << " is equal to " << grid[i] << std::endl;
	}
	

	// Thrust
// 	thrust::host_vector<float> tCharge(nOfCharges);
// 	thrust::host_vector<float> tGrid(nOfCharges);
// 	for (unsigned int i = 0; i < nOfCharges; i++) {
// 		tCharge[i] = charge[i];
// 		tGrid[i] = 0;
// 	}
// 	
// 	thrust::device_vector<float> dev_tCharge = tCharge;
// 	thrust::device_vector<float> dev_tGrid = tGrid;
// 	
// 	float* rawTCharge = thrust::raw_pointer_cast( &tCharge[0] );
// 	float* rawTGrid = thrust::raw_pointer_cast( &tGrid[0] );
// 	
// 	potential_scatter<<<1, nOfCharges>>>(rawTCharge ,rawTGrid);

}

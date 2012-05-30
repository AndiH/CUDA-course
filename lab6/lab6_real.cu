#include <iostream>
#include "stdio.h"
#include <vector>
#include <cuda.h>
// #include "cuPrintf.cu"

const int TILEWIDTH = 2;

__global__ void simpleMatrixStuff (double** firstMatrix, double** resultMatrix) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			resultMatrix[i][j] = 2*firstMatrix[i][j];
		}
	}
}

__global__ void matrixMultTiled (double** firstMatrix, double** secondMatrix, double** resultMatrix, int width = 4) {
	
	__shared__ double localFirstMatrix[TILEWIDTH][TILEWIDTH];
	__shared__ double localSecondMatrix[TILEWIDTH][TILEWIDTH];
	
	double result = 0;
	
	// Define short values for index variables
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int basecol = blockIdx.x * blockDim.x;
	int baserow = blockIdx.y * blockDim.y;
	
	// Tile Loop
	for (int i = 0; i < width / TILEWIDTH; i++) {
		// Fill local matrices
		localFirstMatrix[tx][ty] = firstMatrix[i*TILEWIDTH+tx][baserow+ty];
		localSecondMatrix[tx][ty] = secondMatrix[basecol+tx][i*TILEWIDTH+ty];
// 		localSecondMatrix[ty][tx] = secondMatrix[basecol+tx][i*TILEWIDTH+ty]; // transposing matrix
		__syncthreads(); // (Globally) Wait for all local matrizes to be filled
		
		// Loop for calculating result matrix's element
		for (int j = 0; j < TILEWIDTH; j++) {
			result += localFirstMatrix[j][tx] * localSecondMatrix[ty][j];
// 			result += localFirstMatrix[j][ty] * localSecondMatrix[j][tx]; // transposed matrix
		}
		__syncthreads(); // (Globally) Wait for every res_matrix element to be calculated
	}
	resultMatrix[basecol+tx][baserow+ty] = result; // Fill calculated res_matrix element in actual final res_matrix
}
// Output stuff
void mOut_singleLine (double* matrixLine, int sizeOfMatrix) {
	for (unsigned int i = 0; i < sizeOfMatrix; i++) {
		if (matrixLine[i] < 10) std::cout << " ";
		std::cout << matrixLine[i] << " ";
	}
}
void mOut_singleMatrix (double ** singleMatrix, int sizeOfMatrix) {
	for (unsigned int i = 0; i < sizeOfMatrix; i++) {
		std::cout << "( ";
		mOut_singleLine(singleMatrix[i], sizeOfMatrix);
		std::cout << ")" << std::endl; 
	}
}
void mOut_product (double** matrixOne, double** matrixTwo, int sizeOfMatrix) {
	for (int i = 0; i < sizeOfMatrix; i++) {
		std::cout << "( ";
		mOut_singleLine(matrixOne[i], sizeOfMatrix);
		std::cout << ")";
		if (i == 1 || i == 2) {
			std::cout << " * ";
		} else {
			std::cout << "   ";
		}
		std::cout << "( ";
		mOut_singleLine(matrixTwo[i], sizeOfMatrix);
		std::cout << ")" << std::endl;
	}
}


int main (int argc, char** argv) {
// 	cudaPrintfInit(); 
	int sizeOfMatrix = 4;
	double** matrixOne = new double *[sizeOfMatrix];
	double** matrixTwo = new double *[sizeOfMatrix];
	double** matrixRes = new double *[sizeOfMatrix];
	
	srand(23);
	
	for (int i = 0; i < sizeOfMatrix; i++) {
		matrixOne[i] = new double[sizeOfMatrix];
		matrixTwo[i] = new double[sizeOfMatrix];
		matrixRes[i] = new double[sizeOfMatrix];
		
		for (int j = 0; j < sizeOfMatrix; j++) {
			matrixOne[i][j] = rand() % 100;
			matrixTwo[i][j] = rand() % 100;
			matrixRes[i][j] = -1;
		}
	}
	std::cout << "Original matrices:" << std::endl;
// 	mOut_singleMatrix(matrixOne, sizeOfMatrix);
	mOut_product(matrixOne, matrixTwo, sizeOfMatrix);
	
	
	double** dev_matrixOne;
	double** dev_matrixTwo;
	double** dev_matrixRes;
	
	cudaMalloc((void**)&dev_matrixOne, sizeOfMatrix*sizeof(double*));
	cudaMalloc((void**)&dev_matrixTwo, sizeOfMatrix*sizeof(double*));
	cudaMalloc((void**)&dev_matrixRes, sizeOfMatrix*sizeof(double*));
	for (int i = 0; i < sizeOfMatrix; i++) {
		double* dev_tempOne = 0;
		cudaMalloc((void**) &dev_tempOne, sizeOfMatrix*sizeof(double));
		cudaMemcpy(dev_matrixOne + i, &dev_tempOne, sizeof(double*), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_tempOne, matrixOne[i], sizeOfMatrix*sizeof(double), cudaMemcpyHostToDevice);
		double* dev_tempTwo = 0;
		cudaMalloc((void**) &dev_tempTwo, sizeOfMatrix*sizeof(double));
		cudaMemcpy(dev_matrixTwo + i, & dev_tempTwo, sizeof(double*), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_tempTwo, matrixTwo[i], sizeOfMatrix*sizeof(double), cudaMemcpyHostToDevice);
		
		double* dev_tempRes = 0;
		cudaMalloc((void**) &dev_tempRes, sizeOfMatrix*sizeof(double));
		cudaMemcpy(dev_matrixRes + i, &dev_tempRes, sizeof(double*), cudaMemcpyHostToDevice);
	}
	
	simpleMatrixStuff<<<1,1>>>(dev_matrixOne, dev_matrixRes);
	
	for (int i = 0; i < sizeOfMatrix; i++) {
		double* tempDouble = 0;
		cudaMemcpy(&tempDouble, dev_matrixRes+i, sizeof(double*), cudaMemcpyDeviceToHost);
		cudaMemcpy(matrixRes[i], tempDouble, sizeOfMatrix*sizeof(double), cudaMemcpyDeviceToHost);
	}
	
	std::cout << "Check for matrix1:" << std::endl;
	mOut_singleMatrix(matrixRes, sizeOfMatrix);
	
	simpleMatrixStuff<<<1,1>>>(dev_matrixTwo, dev_matrixRes);
	
	for (int i = 0; i < sizeOfMatrix; i++) {
		double* tempDouble = 0;
		cudaMemcpy(&tempDouble, dev_matrixRes+i, sizeof(double*), cudaMemcpyDeviceToHost);
		cudaMemcpy(matrixRes[i], tempDouble, sizeOfMatrix*sizeof(double), cudaMemcpyDeviceToHost);
	}
	
	std::cout << "Check for matrix2:" << std::endl;
	mOut_singleMatrix(matrixRes, sizeOfMatrix);
	
	matrixMultTiled<<<dim3(2,2), dim3(2,2)>>>(dev_matrixOne, dev_matrixTwo, dev_matrixRes);
	
	for (int i = 0; i < sizeOfMatrix; i++) {
		double* tempDouble = 0;
		cudaMemcpy(&tempDouble, dev_matrixRes+i, sizeof(double*), cudaMemcpyDeviceToHost);
		cudaMemcpy(matrixRes[i], tempDouble, sizeOfMatrix*sizeof(double), cudaMemcpyDeviceToHost);
	}
	
	std::cout << "Multiplied Matrix: " << std::endl;
	mOut_singleMatrix(matrixRes, sizeOfMatrix);
	std::cout << "Which is, sadly, wrong -- if you use B*A instead of A*B you get the right result: " << std::endl;
	
	matrixMultTiled<<<dim3(2,2), dim3(2,2)>>>(dev_matrixTwo, dev_matrixOne, dev_matrixRes);
	
	for (int i = 0; i < sizeOfMatrix; i++) {
		double* tempDouble = 0;
		cudaMemcpy(&tempDouble, dev_matrixRes+i, sizeof(double*), cudaMemcpyDeviceToHost);
		cudaMemcpy(matrixRes[i], tempDouble, sizeOfMatrix*sizeof(double), cudaMemcpyDeviceToHost);
	}
	
	mOut_singleMatrix(matrixRes, sizeOfMatrix);
	
// 	cudaPrintfDisplay(stdout, true);
// 	cudaPrintfEnd();
	
}

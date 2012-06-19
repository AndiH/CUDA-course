#include <iostream>
#include "stdio.h"
#include <vector>
#include <cuda.h>
#include "TROOT.h"
#include "TApplication.h"
#include "TSystem.h"
#include "TMinuit.h"
#include "TRandom3.h"
#include "TVectorT.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TF1.h"
#include "math.h"
#include "../cuPrintf.cu"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

/*
 * ####################
 * ###    SYNTAX    ###
 * ####################
 * 
 * Trying out the following syntax for index_array
 * 
 * nOfFunctions, position_of_weight_of_func1_in_PARAM_array, position_of_rest_parameters_of_func1_in_THIS_array, position_of_weight_of_func2_in_PARAM_array, position_of_rest_parameters_of_func2_in_THIS_array, ..., position_of_1st_param_of_func1_in_PARAM_array, position_of_2nd_param_of_func1_in_PARAM_array, ..., position_of_1st_param_of_func2_in_PARAM_array, position_of_2nd_param_of_func2_in_PARAM_array, ...
 * 
 * NOTE: If a a function's mean should be 1 - all_other_functions_means, provide a -1 as position_of_weight_of_func1_in_PARAM_array (see indices array further down)
 * 
 */


thrust::device_vector<double>* dev_theEvents;

__device__ double dev_gaussian (double x, double* params, int* indices) {
	/* ###    SYNTAX    ###
	 * The Gaussian needs a local version of the index array. This local version is
	 *   # [0] = position of mean in PARAM array
	 *   # [1] = position of sigma PARAM array
	 */

	double mean = params[indices[0]];
	double sigma = params[indices[1]];
// 	printf("indices[0] = %i, params[0] = mean = %f\n", indices[0], mean); // ### DEBUG
	
	return exp(-0.5*pow((x - mean)/sigma, 2)) / (sigma * sqrt(2 * M_PI));
}

typedef double(*dev_function_pointer)(double, double*, int*);
__device__ dev_function_pointer pointer_to_gaussian = dev_gaussian; // can't be converted to be a parameter, so it still needs to be a global variable


__device__ void* dev_function_table[200]; // = dev_function_table
void* host_function_table[200]; // used in an alternative implementation, see below

struct sumOfFunctions {
	double* params;
	int* indices;
	
	__host__ __device__ double operator() (double x) {
		int nOfFunctions = indices[0];
		double insideOfLog = 0;
		for (int i = 0; i < nOfFunctions; i++) {
			dev_function_pointer f = reinterpret_cast<dev_function_pointer>(dev_function_table[i]);
			int position_of_weight_of_f = indices[1+i*2];
			double weight_of_f = params[position_of_weight_of_f]; 
			
			// ### BEGIN part for common mean
			// if weight_of_f is -1 it should actually be 1-sumofothermeans
			if (position_of_weight_of_f <= -1) {
				double sumOfOtherWeights = 0;
				for (int j = 0; j < nOfFunctions; j++) {
					int position_of_weight_of_other_f = indices[1+j*2];
					if (j != i) sumOfOtherWeights += params[position_of_weight_of_other_f]; // sum up all weights but the current one
				}
				printf(""); // ### FIXME Why is this line needed?!
// 				printf(" sumOfOtherWeights = %f\n", sumOfOtherWeights); /// ### DEBUG
				weight_of_f = 1 - sumOfOtherWeights;
			}
			// ### END part for common mean
			
			int position_of_rest_params_of_f = indices[2+i*2];
			int * localIndices = indices + position_of_rest_params_of_f; // set pointer to new position
			insideOfLog += weight_of_f * (*f)(x, params, localIndices);
		}
		return -2 * log(insideOfLog);
	}
};

sumOfFunctions * fStruct = 0; // Declination of a pointer to SumFunctor, so that dev_FitFcn will work

void dev_FitFcn (int& npar, double* deriv, double& fun, double* param, int flg) {
	cudaMemcpy(fStruct->params, param, npar*sizeof(double), cudaMemcpyHostToDevice); // Allocated in main(), copied here (see comment in main())
	fun = thrust::transform_reduce(dev_theEvents->begin(), dev_theEvents->end(), (*fStruct), 0.0, thrust::plus<double>());
// 	printf("fun = %f\n", fun); // ### DEBUG
}
	
int main (int argc, char** argv) {
	int sizeOfVector = 10000;
	if (argc > 1) sizeOfVector = atoi(argv[1]);
	
	TRandom3 myRandom(23);
	double myMean = 2.7;
	double mySigma = 0.2;
	double myWeight = 0.6;
	
	double myMean2 = 4.1;
	double mySigma2 = 0.8;
	
	std::cout << "Original Parameters: " << std::endl << "  Function 1: Mean = " << myMean << ", Sigma = " << mySigma << ", weight = " << myWeight << std::endl << "  Function 2: Mean = " << myMean2 << ", Sigma = " << mySigma2 << std::endl;
	
	// Filling vector with data
	std::vector<double> theEvents;
	for (int i = 0; i < sizeOfVector; i++) {
		if (myRandom.Uniform() <= myWeight) {
			theEvents.push_back(myRandom.Gaus(myMean, mySigma));
		} else {
			theEvents.push_back(myRandom.Gaus(myMean2, mySigma2));
		}
	}
	thrust::device_vector<double> dev_local_theEvents(theEvents);
	dev_theEvents = &dev_local_theEvents;
	
	TMinuit myMinuit(2);
	// In the following DefineParameters I always added some uncertainties by involving TRandom3.Uniform
	myMinuit.DefineParameter(0, "weight1", myRandom.Uniform(myWeight+myWeight*0.1, myWeight-myWeight*0.1), 0.1, myWeight*0.5, myWeight*1.5);
	myMinuit.DefineParameter(1, "mean1", myRandom.Uniform(myMean+myMean*0.1, myMean-myMean*0.1), 0.1, myMean*0.5, myMean*1.5);
	myMinuit.DefineParameter(2, "sigma1", myRandom.Uniform(mySigma+mySigma*0.1, mySigma-mySigma*0.1), 0.1, mySigma*0.5, mySigma*1.5);
	myMinuit.DefineParameter(3, "mean2", myRandom.Uniform(myMean2+myMean2*0.1, myMean2-myMean2*0.1), 0.1, myMean2*0.5, myMean2*1.5);
	myMinuit.DefineParameter(4, "sigma2", myRandom.Uniform(mySigma2+mySigma2*0.1, mySigma-mySigma2*0.1), 0.1, mySigma*0.5, mySigma*1.5);
	
	int * indices = new int[9];
	indices[0] = 2; // 2 functions
	indices[1] = 0; // func1: position of weight in PARAM array
	indices[2] = 5; // func1: position of rest info in this array
	indices[3] = -1;// func2: position of weight in PARAM array
	indices[4] = 7; // func2: position of rest info in this array
	indices[5] = 1; // func1: position of mean in PARAM array
	indices[6] = 2; // func1: position of sigma in PARAM array
	indices[7] = 3; // func2: position of mean in PARAM array
	indices[8] = 4; // func2: position of sigma in PARAM array

	fStruct = new sumOfFunctions(); // Definition of SumFunctor; 
	
	// Allocate memory for local indices array and copy it to the GPU
	cudaMalloc((void**) &(fStruct->indices), 5*sizeof(int));
	cudaMemcpy(fStruct->indices, indices, 5*sizeof(int), cudaMemcpyHostToDevice);
	
	// Point to the functions which actually should be fitted
	/* #######################
	 * ###      NOTE       ### 
	 * It is either possible to work with an host function table and copy it to the device function table, or to extend the host function table by each device function addresses by pointer arithmetic.
	 * The first version follows as a out commented and un-test example, the second version is acutally used.
	 * #######################
	 */
// 	void * dummy[1];
// 	cudaMemcpyFromSymbol(dummy, "dev_gaussian", sizeof(void*));
// 	host_function_table[0] = dummy[0];
// 	host_function_table[1] = dummy[0];
// 	cudaMemcpyToSymbol(dev_function_table, host_function_table, sizeof(void*));
	
	void * functionAddress[1];
	cudaMemcpyFromSymbol(functionAddress, "pointer_to_gaussian", sizeof(void*));
	cudaMemcpyToSymbol(dev_function_table, functionAddress, sizeof(void*));
	cudaMemcpyToSymbol(dev_function_table, functionAddress, sizeof(void*), 1*sizeof(void*)); // also the second function is pointer to the gaussian, but it should not be placed at the 0th position of dev_function_table - it should be placed there with an offset of 1*sizeof(void*) -- meaning it essentially extends the dev_function_table by one void* element

	// Allocate memory for local params array - it's copied in dev_FitFcn, because this is done serveral times
	cudaMalloc((void**) &(fStruct->params),  5*sizeof(double));
	
	myMinuit.SetFCN(&dev_FitFcn);
	myMinuit.Migrad();
	
		
	/* ################################
	 * ### VISUALIZATION and OUTPUT ###
	 * ################################
	 */
	
	TH1D * histo = new TH1D("histo", "Hello", 100, myMean-6*mySigma, myMean2+6*mySigma2);
	for (int i = 0; i < theEvents.size(); i++) histo->Fill(theEvents[i]);
	
	TF1 * g1 = new TF1("g1", "gaus(0)+gaus(3)", myMean-6*mySigma, myMean2+6*mySigma2);
	
	for (int i = 0; i < 5; i++) {
		double tempVal1, tempVal2;
		myMinuit.GetParameter(i, tempVal1, tempVal2);
		int k = i;
		if (i > 2) k = i+1; // skip for second weight
		g1->SetParameter(k, tempVal1);
	}
	g1->SetParameter(3, 1 - g1->GetParameter(0)); // second weight
	
	g1->SetLineColor(kBlue);
	TApplication *theApp = new TApplication("app", &argc, argv, 0, -1);
	TCanvas * c1 = new TCanvas("c1", "default", 100, 10, 800, 600);
	histo->Scale(1/histo->Integral()*pow(2*M_PI,1.5)); // Normalization is probably wrong
	histo->Draw("hist");
	g1->Draw("same");
	c1->Update();
	theApp->Run();
	
}
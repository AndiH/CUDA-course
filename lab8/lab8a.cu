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
 * nOfFunctions, position_of_weight_of_func1_in_PARAM_array, position_of_rest_parameters_of_func1_in_THIS_array, weight_of_func2_in_PARAM_array, position_of_rest_parameters_of_func2_in_THIS_array, ..., position_of_1st_param_of_func1_in_PARAM_array, position_of_2nd_param_of_func1_in_PARAM_array, ..., position_of_1st_param_of_func2_in_PARAM_array, position_of_2nd_param_of_func2_in_PARAM_array, ...
 * 
 */

// __constant__ __device__ double dev_params[5]; // not need, as the params array is a parameter now
// __constant__ __device__ int dev_indices[100]; // not needed, as the indices array is a parameter now
thrust::device_vector<double>* dev_theEvents;

__device__ double dev_gaussian (double x, double* params, int* indices) {
	/* ###    SYNTAX    ###
	 * The Gaussian needs a local version of the index array. This local version is the global syntax array minus all indices which are not for this one gaussian
	 * So:
	 * nOfFunctions, position_of_weight_of_func1_in_PARAM_array, position_of_rest_parameters_of_func1_in_THIS_array, position_of_1st_param_of_func1_in_PARAM_array, position_of_2nd_param_of_func1_in_PARAM_array
	 * 
	 */

	int position_of_mean_in_index_array = indices[2]; // [0] = nOfFunctions, [1] = weight position, [0] = position of mean in index array
	int positoin_of_sigma_in_index_array = position_of_mean_in_index_array+1; // sigma is one element down the road starting from the position of the mean
	double mean = params[indices[position_of_mean_in_index_array]];
	double sigma = params[indices[positoin_of_sigma_in_index_array]];
// 	printf("indices[0] = %i, params[0] = mean = %f\n", indices[0], mean); // ### DEBUG
	
	return exp(-0.5*pow((x - mean)/sigma, 2)) / (sigma * sqrt(2 * M_PI));
}

typedef double(*dev_function_pointer)(double, double*, int*);
__device__ dev_function_pointer pointer_to_gaussian = dev_gaussian; // can't be converted to be a parameter, so it still needs to be a global variable

struct SumFunctor {
	double* params;
	int* indices;
	void* func;
	
	__host__ __device__ double operator() (double x) {
		double weight = 1; // No weight today
// 		printf("Index 0 = %i\n", indices[0]); // ### DEBUG
		dev_function_pointer f1 = reinterpret_cast<dev_function_pointer>(func);
		double gauss = weight * (*f1)(x, params, indices);
		return -2 * log(gauss);
	}
};

SumFunctor * fStruct = 0; // Declination of a pointer to SumFunctor, so that dev_FitFcn will work

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
	std::cout << "Set variables to mean = " << myMean << ", sigma = " << mySigma << std::endl;
	
	// Filling vector with data
	std::vector<double> theEvents;
	for (int i = 0; i < sizeOfVector; i++) {
		theEvents.push_back(myRandom.Gaus(myMean, mySigma));
	}
	thrust::device_vector<double> dev_local_theEvents(theEvents);
	dev_theEvents = &dev_local_theEvents;
	
	TMinuit myMinuit(2);
	myMinuit.DefineParameter(0, "mean", myRandom.Uniform(myMean+myMean*0.1, myMean-myMean*0.1), 0.1, myMean*0.5, myMean*1.5);
	myMinuit.DefineParameter(1, "sigma", myRandom.Uniform(mySigma+mySigma*0.1, mySigma-mySigma*0.1), 0.1, mySigma*0.5, mySigma*1.5);
	
	int * indices = new int[5];
	indices[0] = 1; // 1 function
	indices[1] = -1; // position of weight // ### there's no weight
	indices[2] = 3; // position of rest info in this array
	indices[3] = 0; // position of mean in param array
	indices[4] = 1; // position of sigma in param array

	
	fStruct = new SumFunctor(); // Definition of SumFunctor; 
	
	// Allocate memory for local indices array and copy it to the GPU
	cudaMalloc((void**) &(fStruct->indices), 5*sizeof(int));
	cudaMemcpy(fStruct->indices, indices, 5*sizeof(int), cudaMemcpyHostToDevice);
	
	// Point the pointer_to_gaussian function pointer on the GPU to fStruct's version
	cudaMemcpyFromSymbol((void**) &(fStruct->func), "pointer_to_gaussian", sizeof(void*), 0, cudaMemcpyDeviceToHost);
	
	// Allocate memory for local params array, it's copied in dev_FitFcn, because this is done serveral times
	cudaMalloc((void**) &(fStruct->params),  5*sizeof(double));
	
	myMinuit.SetFCN(&dev_FitFcn);
	myMinuit.Migrad();
	
		
	/* ################################
	 * ### VISUALIZATION and OUTPUT ###
	 * ################################
	 */
	
	TH1D * histo = new TH1D("histo", "Hello", 100, myMean-6*mySigma, myMean+6*mySigma);
	for (int i = 0; i < theEvents.size(); i++) histo->Fill(theEvents[i]);
	
	TF1 * g1 = new TF1("g1", "gaus(0)", myMean-6*mySigma, myMean+6*mySigma);
	
	double tempVal1, tempVal2;
	myMinuit.GetParameter(0, tempVal1, tempVal2);
	g1->SetParameter(1, tempVal1);
	myMinuit.GetParameter(1, tempVal1, tempVal2);
	g1->SetParameter(2, tempVal1);
	g1->SetParameter(0, 1);
	
	g1->SetLineColor(kBlue);
	TApplication *theApp = new TApplication("app", &argc, argv, 0, -1);
	TCanvas * c1 = new TCanvas("c1", "default", 100, 10, 800, 600);
	histo->Scale(1/histo->Integral()*4*M_PI*M_PI); // Normalization is probably wrong
	histo->Draw("hist");
	g1->Draw("same");
	c1->Update();
	theApp->Run();
	
}
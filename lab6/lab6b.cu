#include <iostream>
#include "stdio.h"
#include <vector>
#include <cuda.h>
#include "TSystem.h"
#include "TMinuit.h"
#include "TRandom3.h"
#include "math.h"
// #include "cuPrintf.cu"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

std::vector<double> theEvents;

__constant__ __device__ double dev_params[2];
thrust::device_vector<double>* d_theEvents;

// *** Testing purpose implementation
// *** Following works but probably slows down the process ***
// __device__ __host__ double gauss (double x, double mean, double sigma) {
// 	return pow((x-mean)/sigma, 2);
// }

struct GaussianFunctor {
// 	GaussianFunctor(double _mean, double _sigma) : mean(_mean), sigma(_sigma) {}
	__device__ double operator() (double x) {
		double mean = dev_params[0];
		double sigma = dev_params[1];
		
// 		return -2*log(exp(-0.5*pow((x-mean)/sigma, 2)));
		return pow((x-mean)/sigma,2);
		
		// *** Start test implementatoin ***
		// *** Following three commented lines will make use  of global gauss function in order to use the same one function from a GPU struct and a CPU method ***
// 		double returnvalue = gauss(x, mean, sigma);
// 		printf("Gauss: x = %f, mean = %f, sigma = %f --> return = %f\n", x, mean, sigma, returnvalue);
// 		return returnvalue;
		// *** End test implementation ***
	}
// private:
// 	double mean, sigma;
};

// FOR TESTING PURPOSES
template <typename T> struct square {
	__host__ __device__ T operator() (const T& x) const {
		return x * x;
	}
};

void FitFcn (int& npar, double* deriv, double& fun, double* param, int flg) {
	double mean = param[0];
	double sigma = param[1];
	
	double nll = 0;
	for (unsigned int i = 0; i < theEvents.size(); i++) {
		double x = theEvents[i];
		double thisEventProb = exp(-0.5*pow((x-mean)/sigma, 2));
		nll -= 2*log(thisEventProb);
		
		// *** Test implementation
// 		nll -= gauss(x, mean, sigma);
	}
	fun = nll;
}

void dev_FitFcn (int& npar, double* deriv, double& fun, double* param, int flg) {
	cudaMemcpyToSymbol("dev_params", param, 2*sizeof(double), 0, cudaMemcpyHostToDevice);
	fun = thrust::transform_reduce(d_theEvents->begin(), d_theEvents->end(), GaussianFunctor(), 0., thrust::plus<double>());
}

int main(int argc, char** argv) {
// 	gSystem->Load("libMinuit");
	std::cout << "############################" << std::endl << "## You're lucky! Because of the default TMinuit output into the shell, I implemented a bunch of line separators!" << std::endl << "############################" << std::endl << std::endl;
	int sizeOfVector = 10000;
	if (argc > 1) sizeOfVector = atoi(argv[1]);
	
	TRandom3 myRandom(23);
	for (int i = 0; i < sizeOfVector; i++) {
		theEvents.push_back(myRandom.Gaus(0,1));
// 		if (i % 100 == 0) std::cout << "## Just pushed " << theEvents[i] << " into number array" << std::endl;
	}

	TMinuit minuit(2);
	std::cout << "## TMINUIT:: Defining parameters ##" << std::endl;
	minuit.DefineParameter(0, "mean", 0, 0.1, -1, 1);
	minuit.DefineParameter(1, "sigma", 1, 0.1, 0.5, 1.5);
	
	
	std::cout << "## TMINUIT:: Setting Functoin ##" << std::endl;
	minuit.SetFCN(&FitFcn);
	std::cout << "## TMINUIT:: Calling Migrad() ##" << std::endl;
	minuit.Migrad();
	
	std::cout << "############################" << std::endl;
	std::cout << "## Now on with the parallized version" << std::endl;
	std::cout << "############################" << std::endl;
	
	
	thrust::device_vector<double> d_localEvents(theEvents);
	d_theEvents = &d_localEvents;

	TMinuit dev_minuit(2);
	dev_minuit.DefineParameter(0, "dmean", 0, 0.1, -1, 1);
	dev_minuit.DefineParameter(1, "dsigma", 1, 0.1, 0.5, 1.5);
	dev_minuit.SetFCN(&dev_FitFcn);
	dev_minuit.Migrad();
	
}
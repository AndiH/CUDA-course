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
#include "math.h"
//#include "cuPrintf.cu"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

// helper function
TVectorD stdVectorToRootVector (std::vector<double> vector) {
	TVectorD tempVector(vector.size());
	for (unsigned int i = 0; i < vector.size(); i++) tempVector[i] = vector[i];
	return tempVector;
}


std::vector<double> theEvents;

__constant__ __device__ double dev_params[5];
thrust::device_vector<double>* d_theEvents;

__device__ double dev_gaussian (double x, double mean, double sigma) {
	return exp(-0.5*pow((x - mean)/sigma, 2)) / (sigma * sqrt(2 * M_PI));
}

struct GaussianFunctor {
	__device__ double operator() (double x) {
		double mean1 = dev_params[0];
		double sigma1 = dev_params[1];
		double mean2 = dev_params[2];
		double sigma2 = dev_params[3];
		double weight1 = dev_params[4];
		
		return -2 * log(weight1 * dev_gaussian(x, mean1, sigma1)
				+ (1 - weight1) * dev_gaussian(x, mean2, sigma2)
			       );
	}
};

void dev_FitFcn (int& npar, double* deriv, double& fun, double* param, int flg) {
	cudaMemcpyToSymbol("dev_params", param, 5*sizeof(double), 0, cudaMemcpyHostToDevice);
	fun = thrust::transform_reduce(d_theEvents->begin(), d_theEvents->end(), GaussianFunctor(), 0., thrust::plus<double>());
}

int main(int argc, char** argv) {
// 	gSystem->Load("libMinuit");
	std::cout << "############################" << std::endl << "## You're lucky! Because of the default TMinuit output into the shell, I implemented a bunch of line separators!" << std::endl << "############################" << std::endl << std::endl;
	int sizeOfVector = 10000;
	if (argc > 1) sizeOfVector = atoi(argv[1]);
	
	TRandom3 myRandom(23);
	double myMean1 = 3;
	double mySigma1 = 0.6;
	double myMean2 = 2;
	double mySigma2 = 0.2;
	double myG1DrawProbability = 0.42; // (0,1]
	std::cout << "Mean1 = " << myMean1 << ", mySigma1 = " << mySigma1 << ", myMean2 = " << myMean2 << ", myMean2 = " << myMean2 << ",  weight1 = " << myG1DrawProbability << std::endl;
	for (int i = 0; i < sizeOfVector; i++) {
		if (myRandom.Uniform() <= myG1DrawProbability) {
			theEvents.push_back(myRandom.Gaus(myMean1, mySigma1));
		} else {
			theEvents.push_back(myRandom.Gaus(myMean2, mySigma2));
		}
	}
	
	thrust::device_vector<double> d_localEvents(theEvents);
	d_theEvents = &d_localEvents;
	
	TMinuit minuit(5);
	std::cout << "## TMINUIT:: Defining parameters ##" << std::endl;
	// DefineParameter syntax is: 
	// 	int paramter number, 
	//	char parmeter name,
	//	double initial value,
	//	double initial error,
	//	double lower limit,
	//	double upper limit
	minuit.DefineParameter(0, "mean1", myMean1, 0.1, myMean1-1, myMean2+1); // add +-2 for uncertainties
	minuit.DefineParameter(1, "sigma1", mySigma1, 0.1, mySigma1-1, mySigma1+1);
	minuit.DefineParameter(2, "mean2", myMean2, 0.1, myMean2-1, myMean2+1);
	minuit.DefineParameter(3, "sigma3", mySigma2, 0.1, mySigma2-1, mySigma2+1);
	minuit.DefineParameter(4, "weight1", 0.5, 0.01, 0., 1.);
	
	
	std::cout << "## TMINUIT:: Setting Function ##" << std::endl;
	minuit.SetFCN(&dev_FitFcn);
	std::cout << "## TMINUIT:: Calling Migrad() ##" << std::endl;
	minuit.Migrad();
	
// 	TVectorD root_theEvents = stdVectorToRootVector(theEvents);
	TH1D * histVis = new TH1D(stdVectorToRootVector(theEvents));
	TH1D * secondHist = new TH1D("secondHist", "Titel", 100, 1, 4);
	for (int i = 0; i < theEvents.size(); i++) secondHist->Fill(theEvents[i]);
	
	TApplication *theApp = new TApplication("app", &argc, argv, 0, -1);
	TCanvas * c1 = new TCanvas("c1", "default", 100, 10, 800, 600);
	secondHist->Draw("hist");
	c1->Update();
// 	c1->Print("c1.pdf");
	theApp->Run();

	
}
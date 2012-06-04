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
//#include "cuPrintf.cu"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

/*
 * ### Description for myself ###
 * 
 * ### ATTENTION: Following description is for an old version of this file. It incorpoorated a general unary_function thingy. Since it didn't work (workaround involved static_cast andor templates!) I skipped this part. ###
 * 
 * I want to generalize the function, which is used for fitting. At least a bit.
 * Since Minuit->SetFCN is called with &FitFcn I have to change FitFcn:
 * 	# Instead of my old GaussianFunctor() I use a general, dereferenced function pointer. This one is set in the main method to my new GaussionFunctor substitute SumFunctor() (fStruct = new SumFunctor()).
 * 	# SumFunctor() ...
 * 		+ inherits from thrust::unary_function and has to, in order to declare it generally in the top of this file, use it as an abstract function in FitFcn() and define it specifically in the main method. 
 * 		+ is the central thingy of this fit.
 * 		+ it does compiler workaround stuff to retrieve the functions I'd like to use for fitting (which acutally is the old gaussion, just wrapped in a confusing present box)
 * 		+ it retrieves the important parameter values from the global dev_params[] array
 * 		+ it does the calculation invoking the workaround-retrieved gauss_functions
 * 		+ it returns the value and
 * 		+ has two public member void pointers which are needed for compiler-workaround stuff
 * 	# Compiler workaround stuff ... 
 * 		+ the old dev_gaussian can't be used in this new method, because IDONTKNOW - anyway a pointer to the old dev_gaussian has to be used (a function pointer to be precise)
 * 		+ workaround is,
 * 			- to make a function pointer to it (declare this in head, fill it in main to skip compiler errors)
 * 			- pass dev_gaussion as a void pointer to the device (since function pointer won't work, yay another workaround) by setting SumFunctor()s func1 and func2 to the pointers to dev_gaussian
 * 			- reinterpret the void pointers to dev_gaussians as function pointers using reinterpret_cast template
 * 			- dereference them and invoke them with the three, good old variables x, mean and sigma; this is so easy because our function pointer typedef is just declared in the manner that the dev_gaussian method call (double, double, double) will be of the same structur
 * 		+ sadly, a simple SumFunctor()->func1 = &dev_gauss, as I would think of, wouldn't work, so there's another strange workaround involving cudaMemcpyFROMSymbol (!)
 * 
 * 
 * ### End of Description ###
 */

// helper function
TVectorD stdVectorToRootVector (std::vector<double> vector) {
	TVectorD tempVector(vector.size());
	for (unsigned int i = 0; i < vector.size(); i++) tempVector[i] = vector[i];
	return tempVector;
}

__constant__ __device__ double dev_params[5];
thrust::device_vector<double>* d_theEvents;

__device__ double dev_gaussian (double x, double mean, double sigma) {
	return exp(-0.5*pow((x - mean)/sigma, 2)) / (sigma * sqrt(2 * M_PI));
}
typedef double(*dev_function_pointer)(double, double, double);
__device__ dev_function_pointer pointer_to_gaussian = dev_gaussian;


struct SumFunctor : public thrust::unary_function<double, double> {
	double operator() (double x) {
		dev_function_pointer f1 = reinterpret_cast<dev_function_pointer>(func1);
		dev_function_pointer f2 = reinterpret_cast<dev_function_pointer>(func2);
		
		double weight1 = dev_params[0];
		double mean1 = dev_params[1];
		double sigma1 = dev_params[2];
		double mean2 = dev_params[3];
		double sigma2 = dev_params[4];
		
		double first_gauss = weight1 * (*f1)(x, mean1, sigma1);
		double second_gauss = (1-weight1) * (*f2)(x, mean2, sigma2);
		
		return -2 * log(first_gauss + second_gauss);
	}
	
	void* func1;
	void* func2;
};

SumFunctor* fStruct = 0;

void dev_FitFcn (int& npar, double* deriv, double& fun, double* param, int flg) {
	cudaMemcpyToSymbol("dev_params", param, npar*sizeof(double), 0, cudaMemcpyHostToDevice);
	fun = thrust::transform_reduce(d_theEvents->begin(), d_theEvents->end(), (*fStruct), 0.0, thrust::plus<double>());
// 	std::cout << fun << std::endl; // DEBUG
}

int main(int argc, char** argv) {
// 	gSystem->Load("libMinuit");
	std::cout << "############################" << std::endl << "## You're lucky! Because of the default TMinuit output into the shell, I implemented a bunch of line separators!" << std::endl << "############################" << std::endl << std::endl;
	int sizeOfVector = 10000;
	if (argc > 1) sizeOfVector = atoi(argv[1]);
	
	TRandom3 myRandom(23);
	double myMean1 = 1.7;
	double mySigma1 = 0.2;
	double myMean2 = 3;
	double mySigma2 = 0.6;
	double myG1DrawProbability = 0.58; // (0,1]
	std::cout << "Mean1 = " << myMean1 << ", mySigma1 = " << mySigma1 << ", myMean2 = " << myMean2 << ", myMean2 = " << myMean2 << ",  weight1 = " << myG1DrawProbability << std::endl;
	std::vector<double> theEvents;
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
	minuit.DefineParameter(0, "weight1", 0.5, 0.01, 0., 1.);
	minuit.DefineParameter(1, "mean1", myMean1, 0.1, myMean1-1, myMean1+1); // add +-1 for uncertainties
	minuit.DefineParameter(2, "sigma1", mySigma1, 0.1, mySigma1-1, mySigma1+1);
	minuit.DefineParameter(3, "mean2", myMean2, 0.1, myMean2-1, myMean2+1);
	minuit.DefineParameter(4, "sigma3", mySigma2, 0.1, mySigma2-1, mySigma2+1);
	
	fStruct = new SumFunctor(); // trick
	cudaMemcpyFromSymbol((void**) &(fStruct->func1), "pointer_to_gaussian", sizeof(void*), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol((void**) &(fStruct->func2), "pointer_to_gaussian", sizeof(void*), 0, cudaMemcpyDeviceToHost);
	std::cout << "## TMINUIT:: Setting Function ##" << std::endl;
	minuit.SetFCN(&dev_FitFcn);
	std::cout << "## TMINUIT:: Calling Migrad() ##" << std::endl;
	minuit.Migrad();
	
	
	/* ############
	 * ### VISUALIZATION and OUTPUT ###
	 */
	
	TH1D * secondHist = new TH1D("secondHist", "Titel", 100, 1, 4);
	for (int i = 0; i < theEvents.size(); i++) secondHist->Fill(theEvents[i]);
	
	TF1 * g1 = new TF1("g1", "gaus(0)+gaus(3)", 1, 4);
	std::vector<double> theReturnedParameters; // Errors are not saved
	
	for (int i = 0; i < 5; i++) {
		double tempVal1, tempVal2;
		minuit.GetParameter(i, tempVal1, tempVal2);
		theReturnedParameters.push_back(tempVal1);
// 		std::cout << "### Pushed back parameter " << i << " = " << tempVal1 << std::endl; // DEBUG
	}
	
	for (int i = 0; i < 5; i++) {

		int k = i;
		if (i > 2) k = i + 1;
// 		std::cout << "### k = " << k << " - param[k] = " << theReturnedParameters[i] << std::endl; // DEBUG
		g1->SetParameter(k, theReturnedParameters[i]);
	}
	g1->SetParameter(3, 1 - theReturnedParameters[0]); // second weight
	
	std::cout << "### Deviation of fitted parameters to original values:" << std::endl;
	std::cout << "### Delta_Weight1 = " << theReturnedParameters[0] << " - " << myG1DrawProbability << " = " << theReturnedParameters[0] - myG1DrawProbability << std::endl;
	std::cout << "### Delta_Mean1 = " << theReturnedParameters[1] << " - " << myMean1 << " = " << theReturnedParameters[1] - myMean1 << std::endl;
	std::cout << "### Delta Sigma1 = " << theReturnedParameters[2] << " - " << mySigma1 << " = " << theReturnedParameters[2] - mySigma1 << std::endl;
	std::cout << "### Delta_Mean2 = " << theReturnedParameters[3] << " - " << myMean2 << " = " << theReturnedParameters[3] - myMean2 << std::endl;
	std::cout << "### Delta Sigma2 = " << theReturnedParameters[4] << " - " << mySigma2 << " = " << theReturnedParameters[4] - mySigma2 << std::endl;
	
	
	g1->SetParameter(0, 0.6);
	g1->SetLineColor(kBlue);
	
	TApplication *theApp = new TApplication("app", &argc, argv, 0, -1);
	TCanvas * c1 = new TCanvas("c1", "default", 100, 10, 800, 600);
	secondHist->Scale(1/secondHist->Integral()*4*M_PI*M_PI);
	secondHist->Draw("hist");
	g1->Draw("SAME");
// 	g1->Print(); // DEBUG
	c1->Update();
	theApp->Run();

	
}
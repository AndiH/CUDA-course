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
__device__ double dev_breitwigner (double x, double x0, double gamma) {
	return (gamma / ((x - x0)*(x - x0) + gamma*gamma)) / M_PI;
}

typedef double(*dev_function_pointer)(double, double, double);
__device__ dev_function_pointer pointer_to_gaussian = dev_gaussian;
__device__ dev_function_pointer pointer_to_breitwigner = dev_breitwigner;


struct SumFunctor : public thrust::unary_function<double, double> {
	double operator() (double x) {
		dev_function_pointer f1 = reinterpret_cast<dev_function_pointer>(func1);
		dev_function_pointer f2 = reinterpret_cast<dev_function_pointer>(func2);
		
		double weight1 = dev_params[0];
		double mean1 = dev_params[1];
		double sigma1 = dev_params[2];
		double locationParam = dev_params[3]; // breitwigner x0 = location of peak
		double scaleParam = dev_params[4]; // breitwigner gamma = HW-at-HM
		
		double gauss = weight1 * (*f1)(x, mean1, sigma1);
		double breitwigner = (1-weight1) * (*f2)(x, locationParam, scaleParam);
		
		return -2 * log(gauss + breitwigner);
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
	double myLocPar = 3;
	double myScalePar = 0.6;
	double myGDrawProbability = 0.58; // (0,1]
	std::cout << "Mean1 = " << myMean1 << ", mySigma1 = " << mySigma1 << ", myLocPar = " << myLocPar << ", myLocPar = " << myLocPar << ",  weight1 = " << myGDrawProbability << std::endl;
	std::vector<double> theEvents;
	for (int i = 0; i < sizeOfVector; i++) {
		if (myRandom.Uniform() <= myGDrawProbability) {
			theEvents.push_back(myRandom.Gaus(myMean1, mySigma1));
		} else {
			theEvents.push_back(myRandom.BreitWigner(myLocPar, myScalePar));
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
	minuit.DefineParameter(1, "mean1", myMean1+myRandom.Uniform(-0.5,+0.5), 0.1, myMean1-1, myMean1+1); // add +-1 for uncertainties
	minuit.DefineParameter(2, "sigma1", mySigma1+myRandom.Uniform(-0.05,+0.05), 0.1, mySigma1-1, mySigma1+1);
	minuit.DefineParameter(3, "myLocPar", myLocPar+myRandom.Uniform(-0.5,+0.5), 0.1, myLocPar-1, myLocPar+1);
	minuit.DefineParameter(4, "myScalePar", myScalePar+myRandom.Uniform(-0.05,+0.05), 0.1, myScalePar-1, myScalePar+1);
	
	fStruct = new SumFunctor(); // trick
	cudaMemcpyFromSymbol((void**) &(fStruct->func1), "pointer_to_gaussian", sizeof(void*), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol((void**) &(fStruct->func2), "pointer_to_breitwigner", sizeof(void*), 0, cudaMemcpyDeviceToHost);
	std::cout << "## TMINUIT:: Setting Function ##" << std::endl;
	minuit.SetFCN(&dev_FitFcn);
	std::cout << "## TMINUIT:: Calling Migrad() ##" << std::endl;
	minuit.Migrad();
	
	
	/* ############
	 * ### VISUALIZATION and OUTPUT ###
	 */
	
	TH1D * secondHist = new TH1D("secondHist", "Titel", 100, 1, 4);
	for (int i = 0; i < theEvents.size(); i++) secondHist->Fill(theEvents[i]);
	
	TF1 * g1 = new TF1("g1", "gaus(0)+[3]*[5]/((x-[4])*(x-[4])+[5]*[5])", 1, 4);
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
	std::cout << "### Delta_Weight1 = " << theReturnedParameters[0] << " - " << myGDrawProbability << " = " << theReturnedParameters[0] - myGDrawProbability << std::endl;
	std::cout << "### Delta_Mean1 = " << theReturnedParameters[1] << " - " << myMean1 << " = " << theReturnedParameters[1] - myMean1 << std::endl;
	std::cout << "### Delta Sigma1 = " << theReturnedParameters[2] << " - " << mySigma1 << " = " << theReturnedParameters[2] - mySigma1 << std::endl;
	std::cout << "### Delta_Mean2 = " << theReturnedParameters[3] << " - " << myLocPar << " = " << theReturnedParameters[3] - myLocPar << std::endl;
	std::cout << "### Delta Sigma2 = " << theReturnedParameters[4] << " - " << myScalePar << " = " << theReturnedParameters[4] - myScalePar << std::endl;
	
	
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
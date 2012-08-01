#include <iostream>
#include "sys/time.h"
using namespace std;

double timeInSeconds (timeval& starttime, timeval& stopstime) {
  return 1e-6*(1e6*(stopstime.tv_sec - starttime.tv_sec) + (stopstime.tv_usec - starttime.tv_usec)); 
}

//__device__ double* dev_vector1 = 0;
//__device__ double* dev_vector2 = 0;
//__device__ double* dev_results = 0;
 
__global__ void device_vector_mult (double* v1, double* v2, double* res) {
  // IMPLEMENT ME 6: Multiply the threadIdx.x element of dev_vector1 by the 
  // corresponding element of dev_vector2, and store in dev_results. 
  res[threadIdx.x] = 2 * v1[threadIdx.x] * v2[threadIdx.x];
}

__global__ void device_vector_reduce (double* results, int* length) {
        int len = *length;
        
        while (len > 1) {
                if (threadIdx.x < (len - (len % 2)) / 2) {
                        results[threadIdx.x] += results[threadIdx.x + (len + (len % 2)) / 2];
                }
                len = (len + (len % 2)) / 2;
                __syncthreads();
        }
}



__global__ void device_vector_simpleAdd (double* vec, double* res) {
  if (threadIdx.x == 0) {
    res[0] = 2;
  }
}


int main (int argc, char** argv) {
  int sizeOfVector = 100;
  if (argc > 1) sizeOfVector = atoi(argv[1]); 

  // Declare and fill host-side arrays of doubles. 
  double* vector1 = new double[sizeOfVector];
  double* vector2 = new double[sizeOfVector];
  double* results = new double[sizeOfVector];
  double* gpuresults = new double[sizeOfVector];
  double* gpuAddresults = new double[sizeOfVector];

  srand(42); 
  for (int i = 0; i < sizeOfVector; ++i) {
    vector1[i] = rand() % 100; 
    vector2[i] = rand() % 100; 
    results[i] = 0; 
    gpuresults[i] = 0;
    gpuAddresults[i] = 0;
  }
  timeval startTime;
  timeval interTime; 
  timeval stopsTime; 

  gettimeofday(&startTime, NULL);

  // Use the CPU for this part. 
  // IMPLEMENT ME 1: Multiply each element of vector1 by the corresponding
  // element in vector2 and store in results. 
  for (int i = 0; i < sizeOfVector; ++i) {
        results[i] = vector1[i] * vector2[i];  
  }
  gettimeofday(&interTime, NULL);

  double total = 0;
  // IMPLEMENT ME 2: Sum the results array and store the sum in total.
  for (int i = 0; i < sizeOfVector; ++i) {
        total += results[i];
  }

  gettimeofday(&stopsTime, NULL);

  cout << "Dot product is: " << total << endl;

  // IMPLEMENT ME 3: Time the above operations together and separately
  // using 'gettimeofday'.

  cout << "Time for multiplication (seconds): " << timeInSeconds(startTime, interTime) << endl;
  cout << "Time for addition       (seconds): " << timeInSeconds(interTime, stopsTime) << endl;
  cout << "Overall time            (seconds): " << timeInSeconds(startTime, stopsTime) << endl;

  double* dev_vector1 = 0;
  double* dev_vector2 = 0;
  double* dev_results = 0;

  int sizeInBytes = sizeOfVector * sizeof(double);


  cudaMalloc((void**) &dev_vector1, sizeInBytes);
  cudaMalloc((void**) &dev_vector2, sizeInBytes);
  cudaMalloc((void**) &dev_results, sizeInBytes);

  cudaMemcpy(dev_vector1, vector1, sizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_vector2, vector2, sizeInBytes, cudaMemcpyHostToDevice);

  gettimeofday(&startTime, NULL);
  device_vector_mult<<<1, sizeOfVector>>>(dev_vector1, dev_vector2, dev_results);

  double gputotal = 0;

  cudaMemcpy(gpuresults, dev_results, sizeInBytes, cudaMemcpyDeviceToHost);

  gettimeofday(&interTime, NULL);
  for (int i = 0; i < sizeOfVector; ++i) {
    gputotal += gpuresults[i];
  }

  gettimeofday(&stopsTime, NULL);

  cout << "GPU-mult Dot product is: " << gputotal << endl;

  cout << "GPU-mult Time for multiplication (seconds): " << timeInSeconds(startTime, interTime) << endl;
  cout << "GPU-mult Time for addition       (seconds): " << timeInSeconds(interTime, stopsTime) << endl;
  cout << "GPU-mult Overall time            (seconds): " << timeInSeconds(startTime, stopsTime) << endl;

  double * dev_added = 0;
  cudaMalloc((void**) &dev_added, sizeof(double));


  //device_vector_simpleAdd<<<1, sizeOfVector>>>(dev_results, dev_added);
  device_vector_reduce<<<1, sizeOfVector>>>(dev_results, &sizeOfVector);


  double host_added = 2;
  //cudaMemcpy(&host_added, &dev_added[0], sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&host_added, &dev_results[0], sizeof(double), cudaMemcpyDeviceToHost);

  cout <<"GPU-full Dot product is: " << host_added << endl;
  cout << "Size of Vectors is: " << sizeOfVector << endl;

  return 0;
}


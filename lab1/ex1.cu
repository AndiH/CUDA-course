#include <iostream>
#include "sys/time.h"
using namespace std;


double timeInSeconds (timeval& starttime, timeval& stopstime) {
  return 1e-6*(1e6*(stopstime.tv_sec - starttime.tv_sec) + (stopstime.tv_usec - starttime.tv_usec)); 
}

__device__ double* dev_vector1 = 0;
__device__ double* dev_vector2 = 0;
__device__ double* dev_results = 0;
 
__global__ void device_vector_mult () {
  // IMPLEMENT ME 6: Multiply the threadIdx.x element of dev_vector1 by the 
  // corresponding element of dev_vector2, and store in dev_results. 
}

int main (int argc, char** argv) {
  int sizeOfVector = 100;
  if (argc > 1) sizeOfVector = atoi(argv[1]); 

  // Declare and fill host-side arrays of doubles. 
  double* vector1 = new double[sizeOfVector];
  double* vector2 = new double[sizeOfVector];
  double* results = new double[sizeOfVector];

  srand(42); 
  for (int i = 0; i < sizeOfVector; ++i) {
    vector1[i] = rand() % 100; 
    vector2[i] = rand() % 100; 
    results[i] = 0; 
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
  for (int i = 0; i < sizeOfVector; +i) {
        total += results[i];
  }

  gettimeofday(&stopsTime, NULL);

  cout << "Dot product is : " << total << endl; 

  // IMPLEMENT ME 3: Time the above operations together and separately
  // using 'gettimeofday'. 

  cout << "Time for multiplication (seconds): " << timeInSeconds(startTime, interTime) << endl; 
  cout << "Time for addition       (seconds): " << timeInSeconds(interTime, stopsTime) << endl; 
  cout << "Overall time            (seconds): " << timeInSeconds(startTime, stopsTime) << endl; 


  // Now on to the GPU! 
  
  // IMPLEMENT ME 4: Use cudaMalloc to allocate space for the three device vectors. 
  // IMPLEMENT ME 5: Use cudaMemcpy to initialise dev_vector1 and dev_vector2 to have
  // the same content as the host-side arrays. 

  // IMPLEMENT ME 6: Put in the function body for device_vector_mult, above. 

  // IMPLEMENT ME 7: Launch a kernel that runs device_vector_mult. 

  // IMPLEMENT ME 8: Use cudaMemcpy to copy back dev_results into results. 

  // IMPLEMENT ME 9: Calculate the dot product by summing over results, same
  // as above. 

  // IMPLEMENT ME 10: Take the time for the kernel launch and the addition,
  // and print out the results (including the dot product) as you did for the CPU. 

  // IMPLEMENT ME 11: Write a reduction kernel that sums over dev_results, and launch it.
  // Time this operation and compare with the code that first moves the transformed data
  // to the host, then sums over it. 

  return 0; 
}

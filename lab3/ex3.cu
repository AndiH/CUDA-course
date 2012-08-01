#include <iostream>
// #include <tuple>

// thrust includes
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>



using namespace std;

// ## Part 1: product template
template <class T> T productCalculation (T a, T b) {
	return a*b;
}


// ## Part 6
struct simpleProd { // first a non-tumple version to start up with structs
	int operator() (int a, int b) {
		return a*b;
	}
};
	
struct tupleProd : public unary_function<int,int> {
	__host__ __device__ int operator() (const thrust::tuple<int,int> &ab) {
		return thrust::get<0>(ab) * thrust::get<1>(ab);
	}
};
	

// ## Part 9
template <typename T> struct templateMod : public unary_function<T,T> { // templated version, built this for bugtracking
	int wrt;
	templateMod(int _wrt) : wrt(_wrt) {}
		
	__host__ __device__ T operator() (const T &x) const {
		return x % wrt;
	}
};

struct sMod {
	int wrt;
	sMod(int _wrt) : wrt(_wrt) {}
		
	__host__ __device__ int operator() (int x) {
		return x % wrt;
	}
};

// ## Part 10

struct tupleMod {
	__host__ __device__ int operator() (const thrust::tuple<int, int> numberAndModuland) { // "moduland" I invented to give a name to the x in a mod x
		return thrust::get<0>(numberAndModuland) % thrust::get<1>(numberAndModuland);
	}
};

int main (int argc, char** argv)  {
	/* ####################
	 * TEMPLATE STUFF
	 * ##################*/
	
	int a_int = 2;
	int b_int = 4;
	
	float a_float = 2.3;
	float b_float = 4.2;
	
	double a_double = 2.323;
	double b_double = 4.242;
	
	cout << "## Part 1: Multiplying by template" << endl;
	
	cout << "# 1.1 - int: ";
	cout << a_int << " * " << b_int << " = " << productCalculation(a_int, b_int) << endl;

	cout << "# 1.2 - float: ";
	cout << a_float << " * " << b_float << " = " << productCalculation(a_float, b_float) << endl;
	
	cout << "# 1.3 - double: ";
	cout << a_double << " * " << b_double << " = " << productCalculation(a_double, b_double) << endl;
	
	/* #####################
	 * THRUST
	 * ####################*/
	
	cout << "## Part 2: First Usage of Thrust -- make two device vectors with randomized entries" << endl;
	
	int sizeOfVector = 100; // use array with 100 entries or ...
	if (argc > 1) sizeOfVector = atoi(argv[1]); // use what has been specified by cmd line
	
	
	thrust::host_vector<int> h_vec1(sizeOfVector); // initialize host vectors
	thrust::host_vector<int> h_vec2(sizeOfVector);
	
	srand(23); 
	for (int i = 0; i < sizeOfVector; ++i) { // fill host vectors randomized
		h_vec1[i] = rand() % 100;
		h_vec2[i] = rand() % 100;
		cout << "# Filled random numbers, h_vec1[" << i << "] = " << h_vec1[i] << ", h_vec2[" << i << "] = " << h_vec2[i] << endl;
	}
	
	thrust::device_vector<int> d_vec1 = h_vec1; // copy host vectors to device vectors
	thrust::device_vector<int> d_vec2 = h_vec2;
	
	
	cout << "## Part 3: Use transform() on third device_vector" << endl;
	
	thrust::device_vector<int> d_res(sizeOfVector);
	thrust::transform(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), d_res.begin(), thrust::multiplies<int>());
	
	
	cout << "## Part 4: Use reduce() on third device_vector" << endl;
	
	int total = reduce(d_res.begin(), d_res.end()); // ", (int) 0, thrust::plus<int>()" has been reduced ;)
	
	cout << "# Reduction is = " << total << endl;
	
	
	cout << "## Part 5: Use inner_product() for single-kernel dot product" << endl;
	
	int innertotal = thrust::inner_product(d_vec1.begin(), d_vec1.end(), d_vec2.begin(), 0);
	
	cout << "# Inner Product is = " << innertotal << endl;
	
	
	cout << "## Part 6: Multiply by struct" << endl;
	
	simpleProd mult;
	cout << "# Simple struct product of " << a_int << " and " << b_int << " is " << mult(a_int, b_int) << endl;
	
	tupleProd fancyMult;
	cout << "# Fancy struct product of " << a_int << " and " << b_int << " is " << fancyMult(thrust::make_tuple(a_int,b_int)) << endl;
	
	
	cout << "## Part 7: zip_iterator() " << endl;
	
	thrust::tuple<int, int> zero = thrust::make_tuple(0, 0);
		
	int zippedProd = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(d_vec1.begin(), d_vec2.begin())), thrust::make_zip_iterator(thrust::make_tuple(d_vec1.end(), d_vec2.end())), 
						tupleProd(), 
						0, 
						thrust::plus<int>());
	
	cout << "# Transformed reduction using zip_it() " << zippedProd << endl;
	
	
	cout << "## Part 8: Modulus struct" << endl;
	
	// see above

	
	cout << "## Part 9: transform_reduce() with modulus struct" << endl;
	
	int modWhat = 2; // mod what?!
		
	cout << "# Number of odd entries " << thrust::transform_reduce(d_vec1.begin(), d_vec1.end(), sMod(modWhat), 0, thrust::plus<int>()) << endl;
	
	
	cout << "## Part 10: Modulus by tuple" << endl;
	
	thrust::constant_iterator<int> constIt(2);

	int tempNumber = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(d_vec1.begin(),constIt)),thrust::make_zip_iterator(thrust::make_tuple(d_vec1.end(),constIt)), 
						tupleMod(), 
						0, 
						thrust::plus<int>());
	
	cout << "# Number of odd entries " << tempNumber << endl;
	
	return 0;
}
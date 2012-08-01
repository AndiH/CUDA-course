## LAB 5 ##
URL: http://www.slac.stanford.edu/~rolfa/cudacourse/lab5.pdf (PDF also backupped here)

1. Using two streams for calculating the dot product. Several chunk sizes should be implemented. Use command line arguments to do so.

2. Using two GPUs for calculating the dot product. I chose Thrust to do it. Uses command line parameters, "./ex5b 1 2" is the default call and equal to "./ex5b".

3. Thrust practice
	1. struct `PrintStruct` with `operator() (thrust::counting_iterator)` which just prints stuff
	2. struct `PrintStruct` with second `operator() (thrust::tuple<int, int*>)` which prints stuff
	3. struct `PrintStruct` with third `operator() (thrust::tuple<int *, int>)` which prints stuff
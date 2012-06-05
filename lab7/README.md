## LAB 7 ##
All about involving the GPU in a TMinuit fit of Gaussians. See here: http://www.slac.stanford.edu/~rolfa/cudacourse/lab7.pdf.
1. Just like last week, but this time a fit with *two* Guassians, weightened 
2. Generalize it a bit more by using function pointers - this time with pointers two two Gaussians.  
It's like the generalized case of 1., specifically for the same thing as 1.
3. Now instead of two function pointers to Guassians, use a Guassian and a Breit-Wigner function.

Additionally to the fit I used ROOT to display the generated (and then fitted-on) events with an overlay of the GPU-fitted function.
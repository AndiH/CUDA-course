## LAB 8 ##
Making the TMinuit-GPU-fit more general and arbritary by working more with pointers and parameter arrays and index arrays.
See http://www.slac.stanford.edu/~rolfa/cudacourse/lab8.pdf and http://www.slac.stanford.edu/~rolfa/cudacourse/lab8Clarify.pdf

1. Gauss struct now uses parameters: param_array and index_array

2. A function table is implemented, the indices are read out and used. Additional feature: A position of weight of -1 means that it takes 1-SumOfOtherMeans as the mean (Attention: There is no check if there is more then one function with a -1 position_of_mean!).


Again I used the visual aid from last week.
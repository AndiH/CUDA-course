## LAB 4 ##
URL: http://www.slac.stanford.edu/~rolfa/cudacourse/lab4.pdf (PDF also backupped here)

1. Buggy kernel with race condition (simply add up something, so that it depends on which process finishes first)

2. NVIDIA has a thing for that in CUDA - atomicAdd

3. Use Thrust's transform_reduce to reduce a vector (but do it fancy with zip_iterators, and constant_iterators, and stuff)

4. Colomb potential calculation using scattering and atomicAdd


Lab never finished 100%, but I put into the code the tutors notes as comments.
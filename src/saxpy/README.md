[SAXPY](https://developer.nvidia.com/blog/six-ways-saxpy/) stands for “Single-Precision A·X Plus Y”. 
It is a function in the standard Basic Linear Algebra Subroutines (BLAS)library.
SAXPY is a combination of scalar multiplication and vector addition,
and it’s very simple: it takes as input two vectors of 32-bit floats X and Y with N elements each, and a scalar value A.
It multiplies each element X[i] by A and adds the result to Y[i]. 

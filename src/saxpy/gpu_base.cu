
__global__ void saxpy_kernel(const int n, const float a, const float *x, float *y) {
  // Get our global thread ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Make sure we do not go out of bounds
  if (i < n)
    y[i] = a * x[i] + y[i];
}

extern "C"
void saxpy(const int n, const float a, const float *x, float *y) {
  // Size, in bytes, of each vector
  int64_t bytes = int64_t(n) * sizeof(float);

  // Allocate memory for each vector on GPU
  float *d_x, *d_y;
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);
  cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)n/blockSize);

  // Execute the kernel
  saxpy_kernel<<<gridSize, blockSize>>>(n, a, d_x, d_y);

  // Copy array back to host
  cudaMemcpy( y, d_y, bytes, cudaMemcpyDeviceToHost );

  // Release device memory
  cudaFree(d_x);
  cudaFree(d_y);
}

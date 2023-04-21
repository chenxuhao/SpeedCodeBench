
#define ROWS_BLOCKDIM_X       16
#define COLUMNS_BLOCKDIM_X    16
#define ROWS_BLOCKDIM_Y       4
#define COLUMNS_BLOCKDIM_Y    8
#define ROWS_RESULT_STEPS     8
#define COLUMNS_RESULT_STEPS  8
#define ROWS_HALO_STEPS       1
#define COLUMNS_HALO_STEPS    1

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__global__ void conv_rows(float *__restrict__ dst,
                          const float *__restrict__ src,
                          const float *__restrict__ kernel,
                          const int imageW, const int imageH, const int pitch);
 
__global__ void conv_cols(float *__restrict__ dst,
                          const float *__restrict__ src,
                          const float *__restrict__ kernel,
                          const int imageW, const int imageH, const int pitch);

extern "C"
void convolution(float *Dst, float *Src, float *Kernel, int imageW, int imageH, int kernelR) {
  float* d_Kernel;
  cudaMalloc((void**)&d_Kernel, sizeof(float)*KERNEL_LENGTH);
  cudaMemcpy(d_Kernel, Kernel, sizeof(float)*KERNEL_LENGTH, cudaMemcpyHostToDevice);

  float* d_Input;
  cudaMalloc((void**)&d_Input, sizeof(float)*imageW*imageH);
  cudaMemcpy(d_Input, Src, sizeof(float)*imageW*imageH, cudaMemcpyHostToDevice);

  float* d_Buffer;
  cudaMalloc((void**)&d_Buffer, sizeof(float)*imageW*imageH);

  float* d_Output;
  cudaMalloc((void**)&d_Output, sizeof(float)*imageW*imageH);

  dim3 block (ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
  dim3 grid (imageW / ROWS_RESULT_STEPS / ROWS_BLOCKDIM_X, imageH/ROWS_BLOCKDIM_Y );
  conv_rows<<<grid, block>>>(d_Buffer, d_Input, d_Kernel, imageW, imageH, kernelR);

  dim3 block_col (COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
  dim3 grid_col (imageW / COLUMNS_BLOCKDIM_X, imageH / COLUMNS_RESULT_STEPS / COLUMNS_BLOCKDIM_Y);
  conv_cols<<<grid_col, block_col>>>(d_Output, d_Buffer, d_Kernel, imageW, imageH, kernelR);

  cudaMemcpy(Dst, d_Output, sizeof(float)*imageW*imageH, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_Kernel);
  cudaFree(d_Input);
  cudaFree(d_Buffer);
  cudaFree(d_Output);
}

__global__ void conv_rows(
    float *__restrict__ dst,
    const float *__restrict__ src,
    const float *__restrict__ kernel,
    const int imageW,
    const int imageH,
    const int pitch) {
  __shared__ float l_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

  int gidX = blockIdx.x;
  int gidY = blockIdx.y;
  int lidX = threadIdx.x;
  int lidY = threadIdx.y;
  //Offset to the left halo edge
  const int baseX = (gidX * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + lidX;
  const int baseY = gidY * ROWS_BLOCKDIM_Y + lidY;

  src += baseY * pitch + baseX;
  dst += baseY * pitch + baseX;

  //Load main data
#pragma unroll
  for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = src[i * ROWS_BLOCKDIM_X];

  //Load left halo
#pragma unroll
  for(int i = 0; i < ROWS_HALO_STEPS; i++)
    l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? src[i * ROWS_BLOCKDIM_X] : 0;

  //Load right halo
#pragma unroll
  for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? src[i * ROWS_BLOCKDIM_X] : 0;

  //Compute and store results
  __syncthreads();

#pragma unroll
  for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    float sum = 0;

#pragma unroll
    for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
      sum += kernel[KERNEL_RADIUS - j] * l_Data[lidY][lidX + i * ROWS_BLOCKDIM_X + j];

    dst[i * ROWS_BLOCKDIM_X] = sum;
  }
}

__global__ void conv_cols(
    float *__restrict__ dst,
    const float *__restrict__ src,
    const float *__restrict__ kernel,
    const int imageW,
    const int imageH,
    const int pitch)
{
  __shared__ float l_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

  int gidX = blockIdx.x;
  int gidY = blockIdx.y;
  int lidX = threadIdx.x;
  int lidY = threadIdx.y;

  //Offset to the upper halo edge
  const int baseX = gidX * COLUMNS_BLOCKDIM_X + lidX;
  const int baseY = (gidY * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + lidY;
  src += baseY * pitch + baseX;
  dst += baseY * pitch + baseX;

  //Load main data
#pragma unroll
  for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = src[i * COLUMNS_BLOCKDIM_Y * pitch];

  //Load upper halo
#pragma unroll
  for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
    l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

  //Load lower halo
#pragma unroll
  for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

  //Compute and store results
  __syncthreads();

#pragma unroll
  for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    float sum = 0;

#pragma unroll
    for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
      sum += kernel[KERNEL_RADIUS - j] * l_Data[lidX][lidY + i * COLUMNS_BLOCKDIM_Y + j];

    dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
  }
}


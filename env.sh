#export LD_LIBRARY_PATH=/usr/local/OpenBLAS/build/lib:$LD_LIBRARY_PATH
export KMP_AFFINITY=scatter
export KMP_LIBRARY=turnaround
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=20

export MPI_HOME=/usr
export CUDA_HOME=/usr/local/cuda
#PAPI_HOME = /usr/local/papi-6.0.0
#ICC_HOME = /opt/intel/compilers_and_libraries/linux/bin/intel64
export OPENBLAS_DIR=/usr/local/openblas
export MKL_DIR=/opt/apps/sysnet/intel/20.0/mkl
#export MKL_DIR = /opt/intel/mkl

#export GCC_HOME=/usr/lib/gcc/x86_64-linux-gnu/8
#export CILK_HOME=/home/cxh/OpenCilk/build
export CILK_HOME=/opt/opencilk/
#export CILK_CLANG=/home/cxh/OpenCilk/build/lib/clang/14.0.6
export CILK_CLANG=${CILK_HOME}/bin
export THRUST_HOME=/usr/local/cuda/include/thrust


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <assert.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM
#define ASSUMED_NR_CLUSTERS 32
#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

#define FLT_MAX 3.40282347e+38

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

/* to turn on the GPU delta and center reduction */
//#define GPU_DELTA_REDUCTION
//#define GPU_NEW_CENTER_REDUCTION

// t_features has the layout dim0[points 0-m-1]dim1[ points 0-m-1]...
texture<float, 1, cudaReadModeElementType> t_features;
// t_features_flipped has the layout point0[dim 0-n-1]point1[dim 0-n-1]
texture<float, 1, cudaReadModeElementType> t_features_flipped;
texture<float, 1, cudaReadModeElementType> t_clusters;

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */

/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */

float	min_rmse_ref = FLT_MAX;			/* reference min_rmse value */
void allocateDeviceMemory(int npoints, int nfeatures, int nclusters, float **features);
void deallocateMemory();
int kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
               int      nfeatures,				/* number of attributes for each point */
               int      npoints,				/* number of data points */
               int      nclusters,				/* number of clusters */
               int     *membership,				/* which cluster the point belongs to */
               float  **clusters,				/* coordinates of cluster centers */
               int     *new_centers_len,		/* number of elements in each cluster */
               float  **new_centers);			/* sum of elements in each cluster */
 
extern "C"
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) {
  allocateDeviceMemory(npoints, nfeatures, nclusters, feature);
  int i, j, n = 0;     // counters
  int loop=0, temp;
  int *new_centers_len;// [nclusters]: no. of points in each cluster 
  float delta;         // if the point moved */
  float **clusters;    // out: [nclusters][nfeatures] */
  float **new_centers; // [nclusters][nfeatures] */
  int *initial;	       /* used to hold the index of points not yet selected
                               prevents the "birthday problem" of dual selection (?)
                               considered holding initial cluster indices, but changed due to
                               possible, though unlikely, infinite loops */
  int initial_points;
  int c = 0;
  /* nclusters should never be > npoints
     that would guarantee a cluster without points */
  if (nclusters > npoints)
    nclusters = npoints;
  /* allocate space for and initialize returning variable clusters[] */
  clusters    = (float**) malloc(nclusters *             sizeof(float*));
  clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
  for (i=1; i<nclusters; i++)
    clusters[i] = clusters[i-1] + nfeatures;
  /* initialize the random clusters */
  initial = (int *) malloc (npoints * sizeof(int));
  for (i = 0; i < npoints; i++) {
    initial[i] = i;
  }
  initial_points = npoints;
  /* randomly pick cluster centers */
  for (i=0; i<nclusters && initial_points >= 0; i++) {
    for (j=0; j<nfeatures; j++)
      clusters[i][j] = feature[initial[n]][j];	// remapped
    /* swap the selected index to the end (not really necessary,
       could just move the end up) */
    temp = initial[n];
    initial[n] = initial[initial_points-1];
    initial[initial_points-1] = temp;
    initial_points--;
    n++;
  }
  /* initialize the membership to -1 for all */
  for (i=0; i < npoints; i++)
    membership[i] = -1;
  /* allocate space for and initialize new_centers_len and new_centers */
  new_centers_len = (int*) calloc(nclusters, sizeof(int));
  new_centers    = (float**) malloc(nclusters *            sizeof(float*));
  new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
  for (i=1; i<nclusters; i++)
    new_centers[i] = new_centers[i-1] + nfeatures;
  /* iterate until convergence */
  do {
    delta = 0.0;
    delta = (float) kmeansCuda(feature,			/* in: [npoints][nfeatures] */
        nfeatures,		/* number of attributes for each point */
        npoints,			/* number of data points */
        nclusters,		/* number of clusters */
        membership,		/* which cluster the point belongs to */
        clusters,		/* out: [nclusters][nfeatures] */
        new_centers_len,	/* out: number of points in each cluster */
        new_centers		/* sum of points in each cluster */
        );
      /* replace old cluster centers with new_centers */
      /* CPU side of reduction */
      for (i=0; i<nclusters; i++) {
        for (j=0; j<nfeatures; j++) {
          if (new_centers_len[i] > 0)
            clusters[i][j] = new_centers[i][j] / new_centers_len[i];	/* take average i.e. sum/n */
              new_centers[i][j] = 0.0;	/* set back to 0 */
        }
        new_centers_len[i] = 0;			/* set back to 0 */
      }
    c++;
  } while ((delta > threshold) && (loop++ < 500));	/* makes sure loop terminates */
  printf("iterated %d times\n", c);
  free(new_centers[0]);
  free(new_centers);
  free(new_centers_len);
  deallocateMemory();
  return clusters;
}

/*
int cluster(int      npoints,				// number of data points 
            int      nfeatures,				// number of attributes for each point 
            float  **features,	            // array: [npoints][nfeatures] 
            int      min_nclusters,			// range of min to max number of clusters
			int		 max_nclusters,
            float    threshold,				// loop terminating factor
            int     *best_nclusters,		// out: number between min and max with lowest RMSE 
            float ***cluster_centres,		// out: [best_nclusters][nfeatures] 
			float	*min_rmse,				// out: minimum RMSE 
			int		 isRMSE,				// calculate RMSE 
			int		 nloops					// number of iteration for each number of clusters 
			) {
  int		nclusters;						// number of clusters k 
  int		index =0;						// number of iteration to reach the best RMSE 
  int		rmse;							// RMSE for each clustering 
  int    *membership;						// which cluster a data point belongs to 
  float **tmp_cluster_centres;i             // hold coordinates of cluster centers 
  int		i;
  // sweep k from min to max_nclusters to find the best number of clusters 
  for (nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++) {
    if (nclusters > npoints) break;	// cannot have more clusters than points 
    // allocate device memory, invert data array (@ kmeans_cuda.cu) 
    allocateDeviceMemory(npoints, nfeatures, nclusters, features);
    // iterate nloops times for each number of clusters 
    for (i = 0; i < nloops; i++) {
      // initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) 
      tmp_cluster_centres = kmeans_clustering(features,
                                              nfeatures,
                                              npoints,
                                              nclusters,
                                              threshold,
                                              membership);
      if (*cluster_centres) {
        free((*cluster_centres)[0]);
        free(*cluster_centres);
      }
      *cluster_centres = tmp_cluster_centres;
      // find the number of clusters with the best RMSE 
      if (isRMSE) {
        rmse = rms_err(features, nfeatures, npoints, tmp_cluster_centres, nclusters);
        if(rmse < min_rmse_ref){
          min_rmse_ref = rmse;			//update reference min RMSE
          *min_rmse = min_rmse_ref;		//update return min RMSE
          *best_nclusters = nclusters;	//update optimum number of clusters
          index = i;						//update number of iteration to reach best RMSE
        }
      }
    }
    deallocateMemory();							// free device memory (@ kmeans_cuda.cu)
  }
  return index;
}
*/

__constant__ float c_clusters[ASSUMED_NR_CLUSTERS*34];		/* constant memory for cluster centers */

/* inverts data array from row-major to column-major.
   [p0,dim0][p0,dim1][p0,dim2] ... 
   [p1,dim0][p1,dim1][p1,dim2] ... 
   [p2,dim0][p2,dim1][p2,dim2] ... 
										to
   [dim0,p0][dim0,p1][dim0,p2] ...
   [dim1,p0][dim1,p1][dim1,p2] ...
   [dim2,p0][dim2,p1][dim2,p2] ...
*/
__global__ void invert_mapping(float *input,  /* original */
							   float *output, /* inverted */
							   int npoints,
							   int nfeatures) {
  int point_id = threadIdx.x + blockDim.x*blockIdx.x;	/* id of thread */
  int i;
  if (point_id < npoints) {
    for (i=0;i<nfeatures;i++)
      output[point_id + npoints*i] = input[point_id*nfeatures + i];
  }
  return;
}

/* find the index of nearest cluster centers and change membership*/
__global__ void
kmeansPoint(float  *features,			/* in: [npoints*nfeatures] */
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			float  *clusters,
			float  *block_clusters,
			int    *block_deltas) {
  // block ID
  const unsigned int block_id = gridDim.x*blockIdx.y+blockIdx.x;
  // point/thread ID  
  const unsigned int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;
  int  index = -1;
  if (point_id < npoints) {
    int i, j;
    float min_dist = FLT_MAX;
    float dist;													/* distance square between a point to cluster center */
    /* find the cluster center id with min distance to pt */
    for (i=0; i<nclusters; i++) {
      int cluster_base_index = i*nfeatures;					/* base index of cluster centers for inverted array */			
      float ans=0.0;												/* Euclidean distance sqaure */
      for (j=0; j < nfeatures; j++) {					
        int addr = point_id + j*npoints;					/* appropriate index of data point */
        float diff = (tex1Dfetch(t_features,addr) -
            c_clusters[cluster_base_index + j]);	/* distance between a data point to cluster centers */
        ans += diff*diff;									/* sum of squares */
      }
      dist = ans;		
      /* see if distance is smaller than previous ones:
         if so, change minimum distance and save index of cluster center */
      if (dist < min_dist) {
        min_dist = dist;
        index    = i;
      }
    }
  }
#ifdef GPU_DELTA_REDUCTION
  // count how many points are now closer to a different cluster center	
  __shared__ int deltas[THREADS_PER_BLOCK];
  if(threadIdx.x < THREADS_PER_BLOCK) {
    deltas[threadIdx.x] = 0;
  }
#endif
  if (point_id < npoints) {
#ifdef GPU_DELTA_REDUCTION
    /* if membership changes, increase delta by 1 */
    if (membership[point_id] != index) {
      deltas[threadIdx.x] = 1;
    }
#endif
    /* assign the membership to object point_id */
    membership[point_id] = index;
  }

#ifdef GPU_DELTA_REDUCTION
  // make sure all the deltas have finished writing to shared memory
  __syncthreads();

  // now let's count them
  // primitve reduction follows
  unsigned int threadids_participating = THREADS_PER_BLOCK / 2;
  for(;threadids_participating > 1; threadids_participating /= 2) {
    if(threadIdx.x < threadids_participating) {
      deltas[threadIdx.x] += deltas[threadIdx.x + threadids_participating];
    }
    __syncthreads();
  }
  if(threadIdx.x < 1)	{deltas[threadIdx.x] += deltas[threadIdx.x + 1];}
  __syncthreads();
  // propagate number of changes to global counter
  if(threadIdx.x == 0) {
    block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = deltas[0];
    //printf("original id: %d, modified: %d\n", blockIdx.y*gridDim.x+blockIdx.x, blockIdx.x);
  }
#endif

#ifdef GPU_NEW_CENTER_REDUCTION
  int center_id = threadIdx.x / nfeatures;    
  int dim_id = threadIdx.x - nfeatures*center_id;
  __shared__ int new_center_ids[THREADS_PER_BLOCK];
  new_center_ids[threadIdx.x] = index;
  __syncthreads();
  /***
    determine which dimension calculte the sum for
    mapping of threads is
    center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...
   ***/ 	
  int new_base_index = (point_id - threadIdx.x)*nfeatures + dim_id;
  float accumulator = 0.f;
  if(threadIdx.x < nfeatures * nclusters) {
    // accumulate over all the elements of this threadblock 
    for(int i = 0; i< (THREADS_PER_BLOCK); i++) {
      float val = tex1Dfetch(t_features_flipped,new_base_index+i*nfeatures);
      if(new_center_ids[i] == center_id) 
        accumulator += val;
    }
    // now store the sum for this threadblock
    /***
      mapping to global array is
      block0[center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...]block1[...]...
     ***/
    block_clusters[(blockIdx.y*gridDim.x + blockIdx.x) * nclusters * nfeatures + threadIdx.x] = accumulator;
  }
#endif
}

// return an integer delta -- had problems when return value was of float type
int kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
               int      nfeatures,				/* number of attributes for each point */
               int      npoints,				/* number of data points */
               int      nclusters,				/* number of clusters */
               int     *membership,				/* which cluster the point belongs to */
               float  **clusters,				/* coordinates of cluster centers */
               int     *new_centers_len,		/* number of elements in each cluster */
               float  **new_centers) {			/* sum of elements in each cluster */
  int delta = 0;			/* if point has moved */
  int i,j;				/* counters */
  cudaSetDevice(1);
  /* copy membership (host to device) */
  cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice);
  /* copy clusters (host to device) */
  cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
  /* set up texture */
  cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
  t_features.filterMode = cudaFilterModePoint;   
  t_features.normalized = false;
  t_features.channelDesc = chDesc0;
  if(cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
    printf("Couldn't bind features array to texture!\n");
  cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
  t_features_flipped.filterMode = cudaFilterModePoint;   
  t_features_flipped.normalized = false;
  t_features_flipped.channelDesc = chDesc1;
  if(cudaBindTexture(NULL, &t_features_flipped, feature_flipped_d, &chDesc1, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
    printf("Couldn't bind features_flipped array to texture!\n");
  cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
  t_clusters.filterMode = cudaFilterModePoint;   
  t_clusters.normalized = false;
  t_clusters.channelDesc = chDesc2;
  if(cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2, nclusters*nfeatures*sizeof(float)) != CUDA_SUCCESS)
    printf("Couldn't bind clusters array to texture!\n");
  /* copy clusters to constant memory */
  cudaMemcpyToSymbol("c_clusters",clusters[0],nclusters*nfeatures*sizeof(float),0,cudaMemcpyHostToDevice);
  /* setup execution parameters. changed to 2d (source code on NVIDIA CUDA Programming Guide) */
  dim3  grid( num_blocks_perdim, num_blocks_perdim );
  dim3  threads( num_threads_perdim*num_threads_perdim );
  /* execute the kernel */
  kmeansPoint<<< grid, threads >>>( feature_d,
      nfeatures,
      npoints,
      nclusters,
      membership_d,
      clusters_d,
      block_clusters_d,
      block_deltas_d);
  cudaThreadSynchronize();
  /* copy back membership (device to host) */
  cudaMemcpy(membership_new, membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost);	
#ifdef BLOCK_CENTER_REDUCE
  /*** Copy back arrays of per block sums ***/
  float * block_clusters_h = (float *) malloc(
      num_blocks_perdim * num_blocks_perdim * 
      nclusters * nfeatures * sizeof(float));

  cudaMemcpy(block_clusters_h, block_clusters_d, 
      num_blocks_perdim * num_blocks_perdim * 
      nclusters * nfeatures * sizeof(float), 
      cudaMemcpyDeviceToHost);
#endif
#ifdef BLOCK_DELTA_REDUCE
  int * block_deltas_h = (int *) malloc(
      num_blocks_perdim * num_blocks_perdim * sizeof(int));

  cudaMemcpy(block_deltas_h, block_deltas_d, 
      num_blocks_perdim * num_blocks_perdim * sizeof(int), 
      cudaMemcpyDeviceToHost);
#endif

  /* for each point, sum data points in each cluster
     and see if membership has changed:
     if so, increase delta and change old membership, and update new_centers;
     otherwise, update new_centers */
  delta = 0;
  for (i = 0; i < npoints; i++) {
    int cluster_id = membership_new[i];
    new_centers_len[cluster_id]++;
    if (membership_new[i] != membership[i]) {
#ifdef CPU_DELTA_REDUCE
      delta++;
#endif
      membership[i] = membership_new[i];
    }
#ifdef CPU_CENTER_REDUCE
    for (j = 0; j < nfeatures; j++) {			
      new_centers[cluster_id][j] += feature[i][j];
    }
#endif
  }


#ifdef BLOCK_DELTA_REDUCE	
  /*** calculate global sums from per block sums for delta and the new centers ***/    

  //debug
  //printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
  for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
    //printf("block %d delta is %d \n",i,block_deltas_h[i]);
    delta += block_deltas_h[i];
  }

#endif
#ifdef BLOCK_CENTER_REDUCE	

  for(int j = 0; j < nclusters;j++) {
    for(int k = 0; k < nfeatures;k++) {
      block_new_centers[j*nfeatures + k] = 0.f;
    }
  }
  for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
    for(int j = 0; j < nclusters;j++) {
      for(int k = 0; k < nfeatures;k++) {
        block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
      }
    }
  }
#ifdef BLOCK_CENTER_REDUCE
  for(int j = 0; j < nclusters;j++) {
    for(int k = 0; k < nfeatures;k++)
      new_centers[j][k]= block_new_centers[j*nfeatures + k];		
  }
#endif
#endif
  return delta;
}

/* free host and device memory */
void deallocateMemory() {
  free(membership_new);
  free(block_new_centers);
  cudaFree(feature_d);
  cudaFree(feature_flipped_d);
  cudaFree(membership_d);
  cudaFree(clusters_d);
#ifdef BLOCK_CENTER_REDUCE
  cudaFree(block_clusters_d);
#endif
#ifdef BLOCK_DELTA_REDUCE
  cudaFree(block_deltas_d);
#endif
}

/* allocate device memory, calculate number of blocks and threads, and invert the data array */
void allocateDeviceMemory(int npoints, int nfeatures, int nclusters, float **features) {	
  num_blocks = npoints / num_threads;
  if (npoints % num_threads > 0)		/* defeat truncation */
    num_blocks++;
  num_blocks_perdim = sqrt((double) num_blocks);
  while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
    num_blocks_perdim++;
  num_blocks = num_blocks_perdim*num_blocks_perdim;
  /* allocate memory for memory_new[] and initialize to -1 (host) */
  membership_new = (int*) malloc(npoints * sizeof(int));
  for(int i=0;i<npoints;i++) {
    membership_new[i] = -1;
  }
  /* allocate memory for block_new_centers[] (host) */
  block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));
  /* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
  cudaMalloc((void**) &feature_flipped_d, npoints*nfeatures*sizeof(float));
  cudaMemcpy(feature_flipped_d, features[0], npoints*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void**) &feature_d, npoints*nfeatures*sizeof(float));
  /* invert the data array (kernel execution) */	
  invert_mapping<<<num_blocks,num_threads>>>(feature_flipped_d,feature_d,npoints,nfeatures);
  /* allocate memory for membership_d[] and clusters_d[][] (device) */
  cudaMalloc((void**) &membership_d, npoints*sizeof(int));
  cudaMalloc((void**) &clusters_d, nclusters*nfeatures*sizeof(float));
#ifdef BLOCK_DELTA_REDUCE
  // allocate array to hold the per block deltas on the gpu side
  cudaMalloc((void**) &block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int));
  //cudaMemcpy(block_delta_d, &delta_h, sizeof(int), cudaMemcpyHostToDevice);
#endif

#ifdef BLOCK_CENTER_REDUCE
  // allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
  cudaMalloc((void**) &block_clusters_d, num_blocks_perdim * num_blocks_perdim * nclusters * nfeatures * sizeof(float));
  //cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
#endif
}


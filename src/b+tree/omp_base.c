#include <omp.h>									// (in directory known to compiler)			needed by openmp
#include <stdlib.h>									// (in directory known to compiler)			needed by malloc
#include <stdio.h>									// (in directory known to compiler)			needed by printf, stderr
#include "ops.h"								// (in directory provided here)

void kernel_cpu(int cores_arg, record *records,	knode *knodes, long knodes_elem, int korder,
                long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans) {
  double time0 = omp_get_wtime();
  int max_nthreads;
  max_nthreads = omp_get_max_threads();
  omp_set_num_threads(cores_arg);
  int threadsPerBlock;
  threadsPerBlock = korder < 1024 ? korder : 1024;
  double time1 = omp_get_wtime();
  int thid;
  int bid;
  int i;
  // process number of querries
  #pragma omp parallel for private (i, thid)
  for(bid = 0; bid < count; bid++){
    // process levels of the tree
    for(i = 0; i < maxheight; i++){
      // process all leaves at each level
      for(thid = 0; thid < threadsPerBlock; thid++){
        // if value is between the two keys
        if((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid+1] > keys[bid])){
          // this conditional statement is inserted to avoid crush due to but in original code
          // "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
          if(knodes[offset[bid]].indices[thid] < knodes_elem){
            offset[bid] = knodes[offset[bid]].indices[thid];
          }
        }
      }
      // set for next tree level
      currKnode[bid] = offset[bid];
    }
    //At this point, we have a candidate leaf node which may contain
    //the target record.  Check each key to hopefully find the record
    // process all leaves at each level
    for(thid = 0; thid < threadsPerBlock; thid++){
      if(knodes[currKnode[bid]].keys[thid] == keys[bid]){
        ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
      }
    }
  }
  auto time2 = omp_get_wtime();
  printf("Time spent in different stages of CPU/MCPU KERNEL:\n");
  printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n", (float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time2-time0) * 100);
  printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n", (float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time2-time0) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", (float) (time2-time0) / 1000000);
}

void kernel_cpu_2(int cores_arg,knode *knodes,long knodes_elem,int korder,long maxheight,int count,
                  long *currKnode,long *offset,long *lastKnode,long *offset_2,
                  int *start, int *end,int *recstart,int *reclength) {
  int i;
  double time0 = omp_get_wtime();
  int max_nthreads;
  max_nthreads = omp_get_max_threads();
  omp_set_num_threads(cores_arg);
  int threadsPerBlock;
  threadsPerBlock = korder < 1024 ? korder : 1024;
  double time1 = omp_get_wtime();
  // private thread IDs
  int thid;
  int bid;
  // process number of querries
  #pragma omp parallel for private (i, thid)
  for(bid = 0; bid < count; bid++){
    // process levels of the tree
    for(i = 0; i < maxheight; i++){
      // process all leaves at each level
      for(thid = 0; thid < threadsPerBlock; thid++){
        if((knodes[currKnode[bid]].keys[thid] <= start[bid]) && (knodes[currKnode[bid]].keys[thid+1] > start[bid])){
          // this conditional statement is inserted to avoid crush due to but in original code
          // "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
          if(knodes[currKnode[bid]].indices[thid] < knodes_elem){
            offset[bid] = knodes[currKnode[bid]].indices[thid];
          }
        }
        if((knodes[lastKnode[bid]].keys[thid] <= end[bid]) && (knodes[lastKnode[bid]].keys[thid+1] > end[bid])){
          // this conditional statement is inserted to avoid crush due to but in original code
          // "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
          if(knodes[lastKnode[bid]].indices[thid] < knodes_elem){
            offset_2[bid] = knodes[lastKnode[bid]].indices[thid];
          }
        }

      }
      // set for next tree level
      currKnode[bid] = offset[bid];
      lastKnode[bid] = offset_2[bid];
    }
    // process leaves
    for(thid = 0; thid < threadsPerBlock; thid++){
      // Find the index of the starting record
      if(knodes[currKnode[bid]].keys[thid] == start[bid]){
        recstart[bid] = knodes[currKnode[bid]].indices[thid];
      }
    }
    // process leaves
    for(thid = 0; thid < threadsPerBlock; thid++){
      // Find the index of the ending record
      if(knodes[lastKnode[bid]].keys[thid] == end[bid]){
        reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid]+1;
      }
    }
  }
  double time2 = omp_get_wtime();
  printf("Time spent in different stages of CPU/MCPU KERNEL:\n");
  printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n", (float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time2-time0) * 100);
  printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n", (float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time2-time0) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", (float) (time2-time0) / 1000000);
}

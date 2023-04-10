#include "stream.h"


/* For a given point x, find the cost of the following operation:
 * -- open a facility at x if there isn't already one there,
 * -- for points y such that the assignment distance of y exceeds dist(y, x),
 *    make y a member of x,
 * -- for facilities y such that reassigning y and all its members to x 
 *    would save cost, realize this closing and reassignment.
 * 
 * If the cost of this operation is negative (i.e., if this entire operation
 * saves cost), perform this operation and return the amount of cost saved;
 * otherwise, do nothing.
 */

/* numcenters will be updated to reflect the new number of centers */
/* z is the facility cost, x is the number of this point in the array points */
double pgain(long x, Points *points, double z, long int *numcenters, int kmax,
             bool *is_center, int *center_table, bool *switch_membership, bool isCoordChanged) {
#ifdef PROFILE
  double t0 = gettime();
#endif	

  //my block
  long bsize = points->num;
  long k1 = 0;
  long k2 = k1 + bsize;
  k2 = points->num;
  int i;
  int number_of_centers_to_close = 0;

  static double *work_mem;
  static double gl_cost_of_opening_x;
  static int gl_number_of_centers_to_close;

  //each thread takes a block of working_mem.
  int stride = *numcenters+2;
  //make stride a multiple of CACHE_LINE
  int cl = CACHE_LINE/sizeof(double);
  if( stride % cl != 0 ) { 
    stride = cl * ( stride / cl + 1);
  }
  int K = stride -2 ; // K==*numcenters

  //my own cost of opening x
  double cost_of_opening_x = 0;

  work_mem = (double*) malloc(stride*(2)*sizeof(double));
  gl_cost_of_opening_x = 0;
  gl_number_of_centers_to_close = 0;
  /*For each center, we have a *lower* field that indicates 
    how much we will save by closing the center. 
    Each thread has its own copy of the *lower* fields as an array.
    We first build a table to index the positions of the *lower* fields. 
    */

  int count = 0;
  for ( int i = k1; i < k2; i++ ) {
    if ( is_center[i] ) {
      center_table[i] = count++;
    }
  }
  work_mem[0] = count;

  int accum = 0;
  for( int p = 0; p < 1; p++ ) {
    int tmp = (int)work_mem[p*stride];
    work_mem[p*stride] = accum;
    accum += tmp;
  }

  for( int i = k1; i < k2; i++ ) {		
    if( is_center[i] ) {
      center_table[i] += (int)work_mem[0];
    }		
  }	

  //now we finish building the table. clear the working memory.
  memset(switch_membership + k1, 0, (k2-k1)*sizeof(bool));
  memset(work_mem, 0, stride*sizeof(double));
  memset(work_mem+stride,0,stride*sizeof(double));

#ifdef PROFILE
  double t1 = gettime();
  time_gain_init += t1-t0;
#endif
  //my *lower* fields
  double* lower = &work_mem[0];
  //global *lower* fields
  double* gl_lower = &work_mem[stride];

  #pragma omp parallel for reduction(+: cost_of_opening_x)
  for ( i = k1; i < k2; i++ ) {
    float x_cost = dist(points->p[i], points->p[x], points->dim) 
      * points->p[i].weight;
    float current_cost = points->p[i].cost;
    if ( x_cost < current_cost ) {
      // point i would save cost just by switching to x
      // (note that i cannot be a median, 
      // or else dist(p[i], p[x]) would be 0)			
      switch_membership[i] = 1;
      cost_of_opening_x += x_cost - current_cost;			
    } else {
      // cost of assigning i to x is at least current assignment cost of i

      // consider the savings that i's **current** median would realize
      // if we reassigned that median and all its members to x;
      // note we've already accounted for the fact that the median
      // would save z by closing; now we have to subtract from the savings
      // the extra cost of reassigning that median and its members 
      int assign = points->p[i].assign;
      lower[center_table[assign]] += current_cost - x_cost;			
    }
  }

#ifdef PROFILE
  double t2 = gettime();
  time_gain_dist += t2 - t1;
#endif	
  // at this time, we can calculate the cost of opening a center
  // at x; if it is negative, we'll go through with opening it


  for ( int i = k1; i < k2; i++ ) {
    if( is_center[i] ) {
      auto low = z;
      //aggregate from all threads
      for( int p = 0; p < 1; p++ ) {
        low += work_mem[center_table[i]+p*stride];
      }
      gl_lower[center_table[i]] = low;
      //printf("%d : %f %f\n", i, low, work_mem[center_table[i]+stride]);
      if ( low > 0 ) {
        // i is a median, and
        // if we were to open x (which we still may not) we'd close i

        // note, we'll ignore the following quantity unless we do open x
        ++number_of_centers_to_close;  
        cost_of_opening_x -= low;
      }
    }
  }
  //use the rest of working memory to store the following
  work_mem[K] = number_of_centers_to_close;
  work_mem[K+1] = cost_of_opening_x;
  gl_cost_of_opening_x = z;
  //aggregate
  for( int p = 0; p < 1; p++ ) {
    gl_number_of_centers_to_close += (int)work_mem[p*stride + K];
    gl_cost_of_opening_x += work_mem[p*stride+K+1];
  }
  // Now, check whether opening x would save cost; if so, do it, and otherwise do nothing
  if ( gl_cost_of_opening_x < 0 ) {
    //  we'd save money by opening x; we'll do it
    #pragma omp parallel for
    for ( int i = k1; i < k2; i++ ) {
      bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
      if ( switch_membership[i] || close_center ) {
        // Either i's median (which may be i itself) is closing,
        // or i is closer to x than to its current median
        points->p[i].cost = points->p[i].weight *
          dist(points->p[i], points->p[x], points->dim);
        points->p[i].assign = x;
      }
    }

    for( int i = k1; i < k2; i++ ) {
      if( is_center[i] && gl_lower[center_table[i]] > 0 ) {
        is_center[i] = false;
      }
    }
    if( x >= k1 && x < k2 ) {
      is_center[x] = true;
    }
    *numcenters = *numcenters + 1 - gl_number_of_centers_to_close;
  } else {
    gl_cost_of_opening_x = 0;  // the value we'll return
  }
  free(work_mem);
#ifdef PROFILE
  double t3 = gettime();
  time_gain += t3-t0;
#endif
  //printf("cost=%f\n", -gl_cost_of_opening_x);
  return -gl_cost_of_opening_x;
}

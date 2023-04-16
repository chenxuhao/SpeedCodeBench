#include "stream.h"

double pgain(long x, Points *points, double z, long int *numcenters, int kmax,
             bool *is_center, int *center_table, bool *switch_membership, bool isCoordChanged);

// tells whether two points of D dimensions are identical
int isIdentical(float *i, float *j, int D) {
  int a = 0;
  int equal = 1;
  while (equal && a < D) {
    if (i[a] != j[a]) equal = 0;
    else a++;
  }
  if (equal) return 1;
  else return 0;
}
/*
// comparator for floating point numbers 
static int floatcomp(const void *i, const void *j) {
  float a, b;
  a = *(float *)(i);
  b = *(float *)(j);
  if (a > b) return (1);
  if (a < b) return (-1);
  return(0);
}
*/
// shuffle points into random order
void shuffle(Points *points) {
#ifdef PROFILE
  double t1 = gettime();
#endif
  long i, j;
  Point temp;
  for (i=0;i<points->num-1;i++) {
    j=(lrand48()%(points->num - i)) + i;
    temp = points->p[i];
    points->p[i] = points->p[j];
    points->p[j] = temp;
  }
#ifdef PROFILE
  double t2 = gettime();
  time_shuffle += t2-t1;
#endif
}

/* shuffle an array of integers */
void intshuffle(int *intarray, int length) {
#ifdef PROFILE
  double t1 = gettime();
#endif
  long i, j;
  int temp;
  for (i=0;i<length;i++) {
    j=(lrand48()%(length - i))+i;
    temp = intarray[i];
    intarray[i]=intarray[j];
    intarray[j]=temp;
  }
#ifdef PROFILE
  double t2 = gettime();
  time_shuffle += t2-t1;
#endif
}

/* run speedy on the points, return total cost of solution */
float pspeedy(Points *points, float z, long *kcenter) {
#ifdef PROFILE
  double t1 = gettime();
#endif

  //my block
  long bsize = points->num;
  long k1 = 0;
  long k2 = k1 + bsize;
  k2 = points->num;
  static double totalcost;
  static double* costs; //cost for each thread. 
  static int i;

#ifdef PRINTINFO
  fprintf(stderr, "Speedy: facility cost %lf\n", z);
#endif

  /* create center at first point, send it to itself */
  for( int k = k1; k < k2; k++ ) {
    float distance = dist(points->p[k],points->p[0],points->dim);
    points->p[k].cost = distance * points->p[k].weight;
    points->p[k].assign=0;
  }

  *kcenter = 1;
  costs = (double*)malloc(sizeof(double));
  for (i = 1; i < points->num; i++ )  {
    bool to_open = ((float)lrand48()/(float)INT_MAX)<(points->p[i].cost/z);
    if( to_open ) {
      (*kcenter)++;
      for( int k = k1; k < k2; k++ )  {
        float distance = dist(points->p[i],points->p[k],points->dim);
        if( distance*points->p[k].weight < points->p[k].cost )  {
          points->p[k].cost = distance * points->p[k].weight;
          points->p[k].assign=i;
        }
      }
    }
  }
  double mytotal = 0;
  for( int k = k1; k < k2; k++ )  {
    mytotal += points->p[k].cost;
  }
  costs[0] = mytotal;
  // aggregate costs from each thread
  totalcost=z*(*kcenter);
  for( int i = 0; i < 1; i++ )
  {
    totalcost += costs[i];
  } 
  free(costs);
#ifdef PRINTINFO
  fprintf(stderr, "Speedy opened %d facilities for total cost %lf\n",
      *kcenter, totalcost);
  fprintf(stderr, "Distance Cost %lf\n", totalcost - z*(*kcenter));
#endif

#ifdef PROFILE
  double t2 = gettime();
  time_speedy += t2 -t1;
#endif
  return(totalcost);
}

/* facility location on the points using local search */
/* z is the facility cost, returns the total cost and # of centers */
/* assumes we are seeded with a reasonable solution */
/* cost should represent this solution's cost */
/* halt if there is < e improvement after iter calls to gain */
/* feasible is an array of numfeasible points which may be centers */

float pFL(Points *points, int *feasible, 
          bool *is_center, int *center_table, bool *switch_membership,
          int numfeasible, float z, long *k, int kmax, 
          double cost, long iter, float e, bool isCoordChanged) {
  long i = 0, x = 0;
  int c = 0;
  double change;
  change = cost;
  /* continue until we run iter iterations without improvement */
  /* stop instead if improvement is less than e */
  while (change/cost > 1.0*e) {
    change = 0.0;
    /* randomize order in which centers are considered */
    intshuffle(feasible, numfeasible);
    for (i=0;i<iter;i++) {
      x = i%numfeasible;
      change += pgain(feasible[x], points, z, k, kmax, 
                      is_center, center_table, switch_membership, isCoordChanged);
      c++;
    }
    cost -= change;
#ifdef PRINTINFO
    fprintf(stderr, "%d centers, cost %lf, total distance %lf\n",
        *k, cost, cost - z*(*k));
#endif
  }
  return(cost);
}

int selectfeasible_fast(Points *points, int **feasible, int kmin) {
#ifdef PROFILE
  double t1 = gettime();
#endif
  int numfeasible = points->num;
  if (numfeasible > (ITER*kmin*log((double)kmin)))
    numfeasible = (int)(ITER*kmin*log((double)kmin));
  *feasible = (int *)malloc(numfeasible*sizeof(int));
  float* accumweight;
  float totalweight;
  /* 
     Calcuate my block. 
     For now this routine does not seem to be the bottleneck, so it is not parallelized. 
     When necessary, this can be parallelized by setting k1 and k2 to 
     proper values and calling this routine from all threads ( it is called only
     by thread 0 for now ). 
     Note that when parallelized, the randomization might not be the same and it might
     not be difficult to measure the parallel speed-up for the whole program. 
     */
  //  long bsize = numfeasible;
  long k1 = 0;
  long k2 = numfeasible;

  float w;
  int l,r,k;

  /* not many points, all will be feasible */
  if (numfeasible == points->num) {
    for (int i=k1;i<k2;i++)
      (*feasible)[i] = i;
    return numfeasible;
  }

  accumweight= (float*)malloc(sizeof(float)*points->num);
  accumweight[0] = points->p[0].weight;
  totalweight=0;
  for( int i = 1; i < points->num; i++ ) {
    accumweight[i] = accumweight[i-1] + points->p[i].weight;
  }
  totalweight=accumweight[points->num-1];
  for(int i=k1; i<k2; i++ ) {
    w = (lrand48()/(float)INT_MAX)*totalweight;
    //binary search
    l=0;
    r=points->num-1;
    if( accumweight[0] > w )  { 
      (*feasible)[i]=0; 
      continue;
    }
    while( l+1 < r ) {
      k = (l+r)/2;
      if( accumweight[k] > w ) {
        r = k;
      } 
      else {
        l=k;
      }
    }
    (*feasible)[i]=r;
  }
  free(accumweight); 
#ifdef PROFILE
  double t2 = gettime();
  time_select_feasible += t2-t1;
#endif
  return numfeasible;
}

/* compute approximate kmedian on the points */
float pkmedian(Points *points, long kmin, long kmax, long* kfinal,
               bool *is_center, int *center_table, bool *switch_membership, bool isCoordChanged) {
  int i;
  double cost;
  double hiz, loz, z;
  static long k;
  static int *feasible;
  static int numfeasible;
  static double* hizs;

  hizs = (double*)calloc(1, sizeof(double));
  hiz = loz = 0.0;
  long ptDimension = points->dim;

  //my block
  long bsize = points->num;
  long k1 = 0;
  long k2 = k1 + bsize;
  k2 = points->num;

#ifdef PRINTINFO
  long numberOfPoints = points->num;
  printf("Starting Kmedian procedure\n");
  printf("%i points in %i dimensions\n", numberOfPoints, ptDimension);
#endif

  double myhiz = 0;
  for (long kk=k1;kk < k2; kk++ ) {
    myhiz += dist(points->p[kk], points->p[0],
        ptDimension)*points->p[kk].weight;
  }
  hizs[0] = myhiz;

  for( int i = 0; i < 1; i++ )   {
    hiz += hizs[i];
  }
  loz=0.0; z = (hiz+loz)/2.0;
  /* NEW: Check whether more centers than points! */
  if (points->num <= kmax) {
    /* just return all points as facilities */
    for (long kk=k1;kk<k2;kk++) {
      points->p[kk].assign = kk;
      points->p[kk].cost = 0;
    }
    cost = 0;
    free(hizs); 
    *kfinal = k;
    return cost;
  }
  shuffle(points);
  cost = pspeedy(points, z, &k);
#ifdef PRINTINFO
  printf("Finished first call to speedy, cost=%lf, k=%i\n", cost, k);
#endif
  i=0;
  /* give speedy SP chances to get at least kmin/2 facilities */
  while ((k < kmin)&&(i<SP)) {
    cost = pspeedy(points, z, &k);
    i++;
  }

#ifdef PRINTINFO
  printf("second call to speedy, cost=%lf, k=%d\n", cost, k);
#endif 
  /* if still not enough facilities, assume z is too high */
  while (k < kmin) {
#ifdef PRINTINFO
    printf("%lf %lf\n", loz, hiz);
    printf("Speedy indicates we should try lower z\n");
#endif
    if (i >= SP) {hiz=z; z=(hiz+loz)/2.0; i=0;}
    shuffle(points);
    cost = pspeedy(points, z, &k);
    i++;
  }
  /* now we begin the binary search for real */
  /* must designate some points as feasible centers */
  /* this creates more consistancy between FL runs */
  /* helps to guarantee correct # of centers at the end */
  numfeasible = selectfeasible_fast(points,&feasible,kmin);
  for( int i = 0; i< points->num; i++ ) {
    is_center[points->p[i].assign]= true;
  }
  int d = 0;
  while(1) {
    d++;
#ifdef PRINTINFO
    printf("loz = %lf, hiz = %lf\n", loz, hiz);
    printf("Running Local Search...\n");
#endif
    /* first get a rough estimate on the FL solution */
    cost = pFL(points, feasible, 
               is_center, center_table, switch_membership,
               numfeasible, z, &k, kmax, cost,
               (long)(ITER*kmax*log((double)kmax)), 0.1,
               isCoordChanged);
    /* if number of centers seems good, try a more accurate FL */
    if (((k <= (1.1)*kmax)&&(k >= (0.9)*kmin))||
        ((k <= kmax+2)&&(k >= kmin-2))) {
#ifdef PRINTINFO
      printf("Trying a more accurate local search...\n");
#endif
      /* may need to run a little longer here before halting without
         improvement */
      cost = pFL(points, feasible, 
                 is_center, center_table, switch_membership,
                 numfeasible, z, &k, kmax, cost,
                 (long)(ITER*kmax*log((double)kmax)), 0.001,
                 isCoordChanged);
    }
    if (k > kmax) {
      /* facilities too cheap */
      /* increase facility cost and up the cost accordingly */
      loz = z; z = (hiz+loz)/2.0;
      cost += (z-loz)*k;
    }
    if (k < kmin) {
      /* facilities too expensive */
      /* decrease facility cost and reduce the cost accordingly */
      hiz = z; z = (hiz+loz)/2.0;
      cost += (z-hiz)*k;
    }
    /* if k is good, return the result */
    /* if we're stuck, just give up and return what we have */
    if (((k <= kmax)&&(k >= kmin))||((loz >= (0.999)*hiz)) )
      break;
  }
  printf("loops=%d\n", d);
  //clean up...
  free(feasible); 
  free(hizs);
  *kfinal = k;
  return cost;
}

/* compute the means for the k clusters */
int contcenters(Points *points) {
  long i, ii;
  float relweight;
  for (i=0;i<points->num;i++) {
    /* compute relative weight of this point to the cluster */
    if (points->p[i].assign != i) {
      relweight=points->p[points->p[i].assign].weight + points->p[i].weight;
      relweight = points->p[i].weight/relweight;
      for (ii=0;ii<points->dim;ii++) {
        points->p[points->p[i].assign].coord[ii]*=1.0-relweight;
        points->p[points->p[i].assign].coord[ii]+=
          points->p[i].coord[ii]*relweight;
      }
      points->p[points->p[i].assign].weight += points->p[i].weight;
    }
  }
  return 0;
}

/* copy centers from points to centers */
void copycenters(Points *points, Points* centers, long* centerIDs, long offset) {
  long i;
  long k;
  bool *is_a_median = (bool *) calloc(points->num, sizeof(bool));

  /* mark the centers */
  for ( i = 0; i < points->num; i++ ) {
    is_a_median[points->p[i].assign] = 1;
  }
  k=centers->num;

  /* count how many  */
  for ( i = 0; i < points->num; i++ ) {
    if ( is_a_median[i] ) {
      memcpy( centers->p[k].coord, points->p[i].coord, points->dim * sizeof(float));
      centers->p[k].weight = points->p[i].weight;
      centerIDs[k] = i + offset;
      k++;
    }
  }
  centers->num = k;
  free(is_a_median);
}

void outcenterIDs( Points* centers, long* centerIDs, char* outfile ) {
  FILE* fp = fopen(outfile, "w");
  if( fp==NULL ) {
    fprintf(stderr, "error opening %s\n",outfile);
    exit(1);
  }
  int* is_a_median = (int*)calloc( sizeof(int), centers->num );
  for( int i =0 ; i< centers->num; i++ ) {
    is_a_median[centers->p[i].assign] = 1;
  }

  for( int i = 0; i < centers->num; i++ ) {
    if( is_a_median[i] ) {
      fprintf(fp, "%lu\n", centerIDs[i]);
      fprintf(fp, "%lf\n", centers->p[i].weight);
      for( int k = 0; k < centers->dim; k++ ) {
        fprintf(fp, "%lf ", centers->p[i].coord[k]);
      }
      fprintf(fp,"\n\n");
    }
  }
  fclose(fp);
}

void streamCluster( PStream* stream, long kmin, long kmax, int dim,
    long chunksize, long centersize, char* outfile ) {
  bool *switch_membership, *is_center;
  int *center_table;
  float *block = (float*)malloc( chunksize*dim*sizeof(float) );
  float* centerBlock = (float*)malloc(centersize*dim*sizeof(float) );
  long* centerIDs = (long*)malloc(centersize*dim*sizeof(long));
  if( block == NULL ) { 
    fprintf(stderr,"not enough memory for a chunk!\n");
    exit(1);
  }
  Points points;
  points.dim = dim;
  points.num = chunksize;
  points.p = (Point *)malloc(chunksize*sizeof(Point));
  for( int i = 0; i < chunksize; i++ ) {		
    points.p[i].coord = &block[i*dim];
  }
  Points centers;
  centers.dim = dim;
  centers.p = (Point *)malloc(centersize*sizeof(Point));
  centers.num = 0;
  for( int i = 0; i< centersize; i++ ) {
    centers.p[i].coord = &centerBlock[i*dim];
    centers.p[i].weight = 1.0;
  }
  long IDoffset = 0;
  long kfinal;
  while (1) {
    size_t numRead  = stream->read(block, dim, chunksize ); 
    fprintf(stderr,"read %ld points\n",numRead);
    if ( stream->ferror() || (numRead < (unsigned int)chunksize && !stream->feof()) ) {
      fprintf(stderr, "error reading data!\n");
      exit(1);
    }
    points.num = numRead;
    for( int i = 0; i < points.num; i++ ) {
      points.p[i].weight = 1.0;
    }
    switch_membership = (bool*)malloc(points.num*sizeof(bool));
    is_center = (bool*)calloc(points.num,sizeof(bool));
    center_table = (int*)malloc(points.num*sizeof(int));
    pkmedian(&points, kmin, kmax, &kfinal, is_center, center_table, switch_membership, false);
    fprintf(stderr,"finish local search\n");
    contcenters(&points);
    if( kfinal + centers.num > centersize ) {
      //here we don't handle the situation where # of centers gets too large. 
      fprintf(stderr,"oops! no more space for centers\n");
      exit(1);
    }
#ifdef PRINTINFO
    printf("finish cont center\n");
#endif
    copycenters(&points, &centers, centerIDs, IDoffset);
    IDoffset += numRead;
#ifdef PRINTINFO
    printf("finish copy centers\n"); 
#endif
    free(is_center);
    free(switch_membership);
    free(center_table);
    if( stream->feof() ) {
      break;
    }
  }

  //finally cluster all temp centers
  //whether to switch membership in pgain
  switch_membership = (bool*)malloc(centers.num*sizeof(bool));
  //whether a point is a center
  is_center = (bool*)calloc(centers.num,sizeof(bool));
  //index table of centers
  center_table = (int*)malloc(centers.num*sizeof(int));
  pkmedian(&centers, kmin, kmax, &kfinal, is_center, center_table, switch_membership, true);
  contcenters(&centers);
  outcenterIDs( &centers, centerIDs, outfile);
}


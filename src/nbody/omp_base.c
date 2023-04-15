#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAXDEPTH 32

void Initialization();
void ClearKernel1(int nnodes, int nbodies, int * child);
void ClearKernel2(int nnodes, int * start, float * mass);
int TreeBuilding(int nnodes, int nbodies, int * child, float * posx, float * posy, float * posz);
void Summarization(int nnodes, int nbodies, int * count, int * child, float * mass, float * posx, float * posy, float * posz);
void SortKernel(int nnodes, int nbodies, int * sort, int * count, int * start, int * child);
void BoundingBox(int nnodes, int nbodies,
                 volatile int * start, volatile int * child,
                 volatile float * mass, volatile float * posx,
                 volatile float * posy, volatile float * posz);
void ForceCalculation(int nnodes, int nbodies, int maxdepth,
                      int * err, float dthf, float itolsq, float epssq,
                      int * sort, int * child, float * mass, float * posx,
                      float * posy, float * posz, float * velx, float * vely, 
                      float * velz, float * accx, float * accy, float * accz);
 void Integration(int nbodies, float dtime, float dthf, 
                 float * posx, float * posy, float * posz,
                 float * velx, float * vely, float * velz,
                 float * accx, float * accy, float * accz);

inline bool compare_and_swap(int *x, int old_val, int new_val) {
  return __sync_bool_compare_and_swap(x, old_val, new_val);
}

inline int fetch_and_add(int *x, int inc) {
  return __sync_fetch_and_add(x, inc);
}

int step, bottom;
unsigned int blkcnt;
float radius;

inline int max(int a, int b) {
  return a>=b?a:b;
}

void BarnesHut(int nbodies, int nnodes, int timesteps, float* mass,
               float *posx, float *posy, float *posz,
               float *velx, float *vely, float *velz,
               float *accx, float *accy, float *accz,
               int *sort, int *child, int *count, int *start) {
  int error = 0;
  double time, timing[7];
  double starttime, endtime;
  float dtime, dthf, epssq, itolsq;
  dtime = 0.025;  dthf = dtime * 0.5f;
  epssq = 0.05 * 0.05;
  itolsq = 1.0f / (0.5 * 0.5);
  for (int i = 0; i < 7; i++) timing[i] = 0.0f;
  starttime = omp_get_wtime();
  printf("Starting kernels, timesteps=%d\n", timesteps);
  timing[0] = omp_get_wtime();
  Initialization();
  timing[0] = omp_get_wtime() - timing[0];
  for (int st = 0; st < timesteps; st++) {
    timing[1] = omp_get_wtime();
    BoundingBox(nnodes, nbodies, start, child, mass, posx, posy, posz);
    timing[1] = omp_get_wtime() - timing[1];
    timing[2] = omp_get_wtime();
    ClearKernel1(nnodes, nbodies, child);
    int max_depth = TreeBuilding(nnodes, nbodies, child, posx, posy, posz);
    ClearKernel2(nnodes, start, mass);
    timing[2] = omp_get_wtime() - timing[2];
    timing[3] = omp_get_wtime();
    Summarization(nnodes, nbodies, count, child, mass, posx, posy, posz);
    timing[3] = omp_get_wtime() - timing[3];
    timing[4] = omp_get_wtime();
    SortKernel(nnodes, nbodies, sort, count, start, child);
    timing[4] = omp_get_wtime() - timing[4];
    timing[5] = omp_get_wtime();
    ForceCalculation(nnodes, nbodies, max_depth, &error, dthf, itolsq, epssq, sort, child, mass, posx, posy, posz, velx, vely, velz, accx, accy, accz);
    timing[5] = omp_get_wtime() - timing[5];
    timing[6] = omp_get_wtime();
    Integration(nbodies, dtime, dthf, posx, posy, posz, velx, vely, velz, accx, accy, accz);
    timing[6] = omp_get_wtime() - timing[6];
  }
  endtime = omp_get_wtime();
  double runtime = endtime - starttime; 
  printf("runtime: %.3f ms  (", 1000 * runtime);
  time = 0;
  for (int i = 1; i < 7; i++) {
    printf(" %.3f ", 1000 * timing[i]);
    time += timing[i];
  }
  if (error == 0) {
    printf(") = %.3f ms\n", 1000 * time);
  } else {
    printf(") = %.3f ms FAILED %d\n", 1000 * time, error);
  }
}

// initialize memory
void Initialization() {
  step = -1;
  blkcnt = 0;
}

// compute center and radius
void BoundingBox(int nnodes, int nbodies,
                 volatile int * start, volatile int * child,
                 volatile float * mass, volatile float * posx,
                 volatile float * posy, volatile float * posz) {
  printf("BoundingBox\n");
  float minx, maxx, miny, maxy, minz, maxz;
  minx = maxx = posx[0];
  miny = maxy = posy[0];
  minz = maxz = posz[0];

  // scan all bodies
  #pragma omp parallel for reduction(max:maxx,maxy,maxz) reduction(min:minx,miny,minz)
  for(int i = 1; i < nbodies; i ++) {
    float val = posx[i];
    minx = val<minx?val:minx;
    maxx = val>maxx?val:maxx;
    val = posy[i];
    miny = val<miny?val:miny;
    maxy = val>maxy?val:maxy;
    val = posz[i];
    minz = val<minz?val:minz;
    maxz = val>maxz?val:maxz;
  }

  // compute 'radius'
  float val = fmaxf(maxx - minx, maxy - miny);
  radius = fmaxf(val, maxz - minz) * 0.5f;

  // create root node
  int k = nnodes;
  bottom = k;

  mass[k] = -1.0f;
  start[k] = 0;
  posx[k] = (minx + maxx) * 0.5f;
  posy[k] = (miny + maxy) * 0.5f;
  posz[k] = (minz + maxz) * 0.5f;
  k *= 8;
  for (int i = 0; i < 8; i++) child[k + i] = -1;
  step++;
  printf("root(%f,%f,%f), radius=%f, bottom=%d\n", posx[bottom], posy[bottom], posz[bottom], radius, bottom);
}

// build tree
void ClearKernel1(int nnodes, int nbodies, int * child) {
  int begin = 8 * nbodies;
  int end = 8 * nnodes;
  // iterate over all cells
  #pragma omp parallel for schedule(static)
  for(int id = begin; id < end; id++) {
    child[id] = -1;
  }
}

void ClearKernel2(int nnodes, int * start, float * mass) {
  // iterate over all cells
  #pragma omp parallel for schedule(static)
  for(int id = bottom; id < nnodes; id++) {
    mass[id] = -1.0f;
    start[id] = -1;
  }
}

// compute force
void ForceCalculation(int nnodes, int nbodies, int maxdepth,
                      int * err, float dthf, float itolsq, float epssq,
                      int * sort, int * child, float * mass, float * posx,
                      float * posy, float * posz, float * velx, float * vely, 
                      float * velz, float * accx, float * accy, float * accz) {
  printf("ForceCalculation\n");
  if (maxdepth > MAXDEPTH) {
    *err = maxdepth;
    return;
  }
  // precompute values that depend only on tree level
  float tmp = radius * 2;
  float dq[MAXDEPTH];
  dq[0] = tmp * tmp * itolsq;
  int i;
  for (i = 1; i < maxdepth; i++) {
    dq[i] = dq[i - 1] * 0.25f;
    dq[i - 1] += epssq;
  }
  dq[i - 1] += epssq;

  // iterate over all bodies
  #pragma omp parallel for schedule(static)
  for (int k = 0; k < nbodies; k ++) {
    int pos[MAXDEPTH], node[MAXDEPTH];
    int i = sort[k];  // get permuted/sorted index
    // cache position info
    float px = posx[i];
    float py = posy[i];
    float pz = posz[i];

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    // initialize iteration stack, i.e., push root node onto stack
    int depth = 0;
    pos[depth] = 0;
    node[depth] = nnodes * 8;
    do {
      // stack is not empty, pop the cell on top of stack
      int pd = pos[depth];
      int nd = node[depth];
      while (pd < 8) {
        // node on top of stack has more children to process
        int n = child[nd + pd];  // load child pointer
        pd++;

        if (n >= 0) {
          float dx = posx[n] - px;
          float dy = posy[n] - py;
          float dz = posz[n] - pz;
          // compute distance squared (plus softening)
          float distance = dx*dx + (dy*dy + (dz*dz + epssq));
          // check if cell is far enough away (or is a body)
          if ((n < nbodies) || distance >= dq[depth]) {
            //distance = rsqrtf(distance);  // compute distance
            distance = 1.0f/sqrtf(distance);  // compute distance
            distance = mass[n] * distance * distance * distance;
            ax += dx * distance;
            ay += dy * distance;
            az += dz * distance;
          } else {
            // push cell onto stack
            pos[depth] = pd;
            node[depth] = nd;
            depth++;
            pd = 0;
            nd = n * 8;
          }
        } else {
          pd = 8;  // early out because all remaining children are also zero
        }
      }
      depth--;  // done with this level
    } while (depth >= 0);

    if (step > 0) {
      // update velocity
      velx[i] += (ax - accx[i]) * dthf;
      vely[i] += (ay - accy[i]) * dthf;
      velz[i] += (az - accz[i]) * dthf;
    }

    // save computed acceleration
    accx[i] = ax;
    accy[i] = ay;
    accz[i] = az;
  }
}

// advance bodies
void Integration(int nbodies, float dtime, float dthf, 
                 float * posx, float * posy, float * posz,
                 float * velx, float * vely, float * velz,
                 float * accx, float * accy, float * accz) {
  // iterate over all bodies
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nbodies; i ++) {
    // integrate
    float dvelx = accx[i] * dthf;
    float dvely = accy[i] * dthf;
    float dvelz = accz[i] * dthf;

    float velhx = velx[i] + dvelx;
    float velhy = vely[i] + dvely;
    float velhz = velz[i] + dvelz;

    posx[i] += velhx * dtime;
    posy[i] += velhy * dtime;
    posz[i] += velhz * dtime;

    velx[i] = velhx + dvelx;
    vely[i] = velhy + dvely;
    velz[i] = velhz + dvelz;
  }
}

// sort bodies
void SortKernel(int nnodes, int nbodies, int * sort, int * count, int * start, int * child) {
  // iterate over all cells
  //#pragma omp parallel for schedule(static)
  for (int k = nnodes; k >= bottom; k--) {
    int lstart = start[k];
    if (lstart >= 0) {
      int j = 0;
      for (int i = 0; i < 8; i++) {
        int ch = child[k*8+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            child[k*8+i] = -1;
            child[k*8+j] = ch;
          }
          j++;
          if (ch >= nbodies) {
            // child is a cell
            start[ch] = lstart;  // set start ID of child
            lstart += count[ch];  // add #bodies in subtree
          } else {
            // child is a body
            sort[lstart] = ch;  // record body in 'sorted' array
            lstart++;
          }
        }// end if ch >= 0
      }// end for
    }// end if lstart >= 0
  }// end for
}

// compute center of mass
void Summarization(int nnodes, int nbodies, int * count, int * child, float * mass, float * posx, float * posy, float * posz) {
  printf("Summarization\n");
  int i, ch, cnt;
  float m, cm, px, py, pz;

  for (int j = 0; j < 5; j++) {  // wait-free pre-passes
    // iterate over all cells
    //#pragma omp parallel for schedule(static)
    for(int k = bottom; k < nnodes; k++) {
      if (mass[k] < 0.0f) {
        for (i = 0; i < 8; i++) {
          ch = child[k*8+i];
          if ((ch >= nbodies) && (mass[ch] < 0.0f)) {
            break;
          }
        }
        if (i == 8) {
          // all children are ready
          cm = 0.0f;
          px = 0.0f;
          py = 0.0f;
          pz = 0.0f;
          cnt = 0;
          for (i = 0; i < 8; i++) {
            ch = child[k*8+i];
            if (ch >= 0) {
              if (ch >= nbodies) {
                m = mass[ch];
                cnt += count[ch];
              } else {
                m = mass[ch];
                cnt++;
              }
              // add child's contribution
              cm += m;
              px += posx[ch] * m;
              py += posy[ch] * m;
              pz += posz[ch] * m;
            }
          }
          count[k] = cnt;
          m = 1.0f / cm;
          posx[k] = px * m;
          posy[k] = py * m;
          posz[k] = pz * m;
          mass[k] = cm;
        }
      }
    }
  }

  int flag = 0;
  int j = 0;
  // iterate over all cells
  //	#pragma omp parallel for schedule(static)
  for(int k = bottom; k <= nnodes; k++) {
    if (mass[k] < 0.0f) {
      j = 8;
      for (i = 0; i < 8; i++) {
        ch = child[k*8+i];
        if ((ch < nbodies) || (mass[ch] >= 0.0f)) {
          j--;
        }
      }
      if (j == 0) {
        // all children are ready
        cm = 0.0f;
        px = 0.0f;
        py = 0.0f;
        pz = 0.0f;
        cnt = 0;
        for (i = 0; i < 8; i++) {
          ch = child[k*8+i];
          if (ch >= 0) {
            m = mass[ch];
            if (ch >= nbodies) {
              cnt += count[ch];
            } else {
              cnt++;
            }
            // add child's contribution
            cm += m;
            px += posx[ch] * m;
            py += posy[ch] * m;
            pz += posz[ch] * m;
          }
        }
        count[k] = cnt;
        m = 1.0f / cm;
        posx[k] = px * m;
        posy[k] = py * m;
        posz[k] = pz * m;
        flag = 1;
      }
    }
    if (flag != 0) {
      mass[k] = cm;
      flag = 0;
    }
  }
}

int TreeBuilding(int nnodes, int nbodies, int * child, float * posx, float * posy, float * posz) {
  printf("TreeBuilding\n");
  // cache root data
  float rootx, rooty, rootz;
  rootx = posx[nnodes];
  rooty = posy[nnodes];
  rootz = posz[nnodes];
  int maxdepth = 1;

  // iterate over all bodies
  //#pragma omp parallel for reduction(max: maxdepth) //schedule(static)
  for (int i = 0; i < nbodies; i ++) {
    int j, depth;
    int ch, n, cell, locked, patch;
    float x, y, z, r;
    float px, py, pz, dx, dy, dz;

    // new body, so start traversing at root
    int skip = 0;
    px = posx[i];
    py = posy[i];
    pz = posz[i];
    n = nnodes;
    depth = 1;
    r = radius * 0.5f;
    dx = dy = dz = -r;
    j = 0;
    // determine which child to follow
    if (rootx < px) {j = 1; dx = r;}
    if (rooty < py) {j |= 2; dy = r;}
    if (rootz < pz) {j |= 4; dz = r;}
    x = rootx + dx;
    y = rooty + dy;
    z = rootz + dz;

    // follow path to leaf cell
    ch = child[n*8+j];
    while (ch >= nbodies) {
      n = ch;
      depth++;
      r *= 0.5f;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
      if (x < px) {j = 1; dx = r;}
      if (y < py) {j |= 2; dy = r;}
      if (z < pz) {j |= 4; dz = r;}
      x += dx;
      y += dy;
      z += dz;
      ch = child[n*8+j];
    }

    do{
      if (ch != -2) {  // skip if child pointer is locked and try again later
        locked = n*8+j;
        if (ch == -1) {
          if (compare_and_swap(&child[locked], -1, i)) {  // if null, just insert the new body
            maxdepth = depth > maxdepth ? depth : maxdepth;
            skip = 1;
          }
        } else {  // there already is a body in this position
          if (compare_and_swap(&child[locked], ch, -2)) {  // try to lock
            patch = -1;
            // create new cell(s) and insert the old and new body
            do {
              depth++;
              cell = fetch_and_add(&bottom, -1) - 1;
              if (cell <= nbodies) {
                exit(0);
                bottom = nnodes;
              }
              if (patch != -1) {
                child[n*8+j] = cell;
              }
              patch = cell > patch ? cell : patch;
              j = 0;
              if (x < posx[ch]) j = 1;
              if (y < posy[ch]) j |= 2;
              if (z < posz[ch]) j |= 4;
              child[cell*8+j] = ch;
              n = cell;
              r *= 0.5f;
              dx = dy = dz = -r;
              j = 0;
              if (x < px) {j = 1; dx = r;}
              if (y < py) {j |= 2; dy = r;}
              if (z < pz) {j |= 4; dz = r;}
              x += dx;
              y += dy;
              z += dz;
              ch = child[n*8+j];
              // repeat until the two bodies are different children
            } while (ch >= 0);
            child[n*8+j] = i;
            maxdepth = depth > maxdepth ? depth : maxdepth;
            skip = 2;
          }
        }// end else
      }// end if ch != -2
      if (skip == 2) {
        child[locked] = patch;
      }
    }while(skip==0);
  }
  // record maximum tree depth
  printf("maxdepth=%d\n", maxdepth);
  return maxdepth;
}


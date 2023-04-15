//Copyright (c) 2023, Xuhao Chen. All rights reserved.
//Author: Xuhao Chen <cxh@mit.edu>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
 
// random number generator
#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;

static void drndset(int seed) {
  A = 1;
  B = 0;
  randx = (A * seed + B) & MASK;
  A = (MULT * A) & MASK;
  B = (MULT * B + ADD) & MASK;
}

static double drnd() {
  lastrand = randx;
  randx = (A * randx + B) & MASK;
  return (double)lastrand / TWOTO31;
}

void BarnesHut(int nbodies, int nnodes, int timesteps, float* mass,
               float *posx, float *posy, float *posz,
               float *velx, float *vely, float *velz,
               float *accx, float *accy, float *accz,
               int *sort, int *child, int *count, int *start);
 
int main(int argc, char *argv[]) {
  int nnodes, nbodies, timesteps;
  float *mass, *posx, *posy, *posz, *velx, *vely, *velz, *accx, *accy, *accz;
  double rsc, vsc, r, v, x, y, z, sq, scale;
  int *sort, *child, *count, *start;
  if (argc < 3) {
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps\n");
    exit(-1);
  }
  nbodies = atoi(argv[1]);
  timesteps = atoi(argv[2]);
  if (nbodies < 1) {
    fprintf(stderr, "nbodies is too small: %d\n", nbodies);
    exit(-1);
  }
  if (nbodies > (1 << 30)) {
    fprintf(stderr, "nbodies is too large: %d\n", nbodies);
    exit(-1);
  }
  // allocate memory
  nnodes = nbodies * 2;
  nnodes--;
  printf("configuration: %d bodies, %d nodes, %d time steps\n", nbodies, nnodes, timesteps);
  mass = (float *)malloc(sizeof(float) * (nnodes+1));
  if (mass == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  posx = (float *)malloc(sizeof(float) * (nnodes+1));
  if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
  posy = (float *)malloc(sizeof(float) * (nnodes+1));
  if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
  posz = (float *)malloc(sizeof(float) * (nnodes+1));
  if (posz == NULL) {fprintf(stderr, "cannot allocate posz\n");  exit(-1);}
  velx = (float *)malloc(sizeof(float) * (nnodes+1));
  if (velx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  vely = (float *)malloc(sizeof(float) * (nnodes+1));
  if (vely == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  velz = (float *)malloc(sizeof(float) * (nnodes+1));
  if (velz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  accx = (float *)malloc(sizeof(float) * (nnodes+1));
  if (accx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
  accy = (float *)malloc(sizeof(float) * (nnodes+1));
  if (accy == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
  accz = (float *)malloc(sizeof(float) * (nnodes+1));
  if (accz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}
  child = (int *)malloc(sizeof(int) * (nnodes+1) * 8);
  if (child == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  start = (int *)malloc(sizeof(int) * (nnodes+1));
  if (start == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  count = (int *)malloc(sizeof(int) * (nnodes+1));
  if (count == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
  sort = (int *)malloc(sizeof(int) * (nnodes+1));
  if (sort == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}

  for (int run = 0; run < 1; run++) {
    // generate input
    drndset(7);
    rsc = (3 * 3.1415926535897932384626433832795) / 16;
    vsc = sqrt(1.0 / rsc);
    for (int i = 0; i < nbodies; i++) {
      mass[i] = 1.0 / nbodies;
      r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
      do {
        x = drnd()*2.0 - 1.0;
        y = drnd()*2.0 - 1.0;
        z = drnd()*2.0 - 1.0;
        sq = x*x + y*y + z*z;
      } while (sq > 1.0);
      scale = rsc * r / sqrt(sq);
      posx[i] = x * scale;
      posy[i] = y * scale;
      posz[i] = z * scale;

      do {
        x = drnd();
        y = drnd() * 0.1;
      } while (y > x*x * pow(1 - x*x, 3.5));
      v = x * sqrt(2.0 / sqrt(1 + r*r));
      do {
        x = drnd()*2.0 - 1.0;
        y = drnd()*2.0 - 1.0;
        z = drnd()*2.0 - 1.0;
        sq = x*x + y*y + z*z;
      } while (sq > 1.0);
      scale = vsc * v / sqrt(sq);
      velx[i] = x * scale;
      vely[i] = y * scale;
      velz[i] = z * scale;
    }
    BarnesHut(nbodies, nnodes, timesteps, mass,
              posx, posy, posz,
              velx, vely, velz,
              accx, accy, accz,
              sort, child, count, start);
  }
  // print output
  FILE *fout = fopen("output.txt", "w");
  int i = 0;
  for (i = 0; i < nbodies; i++) {
    fprintf(fout, "%.2e %.2e %.2e\n", posx[i], posy[i], posz[i]);
  }
  fclose(fout);
  free(mass);
  free(posx);
  free(posy);
  free(posz);
  free(velx);
  free(vely);
  free(velz);
  free(accx);
  free(accy);
  free(accz);
  free(child);
  free(start);
  free(count);
  return 0;
}


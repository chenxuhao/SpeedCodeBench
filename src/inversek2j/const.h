#pragma once

#define MAX_LOOP 25
#define MAX_DIFF 0.15f
#define NUM_JOINTS 3
#define PI 3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)
#define BLOCK_SIZE 128

void invkin_cpu(float *xTarget_in, float *yTarget_in, float *angles, int size);
void invkin_omp(float *xTarget_in, float *yTarget_in, float *angles, int size);


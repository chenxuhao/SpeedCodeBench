#include "const.h"
#include <math.h>

void invkin_omp(float *xTarget_in_h, float *yTarget_in_h, float *angle_out_h, int data_size) {
  #pragma omp parallel for simd 
  for (int idx = 0; idx < data_size; idx++) {  
    float angle_out[NUM_JOINTS];
    float curr_xTargetIn = xTarget_in_h[idx];
    float curr_yTargetIn = yTarget_in_h[idx];
    for (int i = 0; i < NUM_JOINTS; i++) {
      angle_out[i] = 0.0;
    }
    float angle;
    // Initialize x and y data
    float xData[NUM_JOINTS_P1];
    float yData[NUM_JOINTS_P1];
    for (int i = 0 ; i < NUM_JOINTS_P1; i++) {
      xData[i] = i;
      yData[i] = 0.f;
    }
    for (int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++) {
      for (int iter = NUM_JOINTS; iter > 0; iter--) {
        float pe_x = xData[NUM_JOINTS];
        float pe_y = yData[NUM_JOINTS];
        float pc_x = xData[iter-1];
        float pc_y = yData[iter-1];
        float diff_pe_pc_x = pe_x - pc_x;
        float diff_pe_pc_y = pe_y - pc_y;
        float diff_tgt_pc_x = curr_xTargetIn - pc_x;
        float diff_tgt_pc_y = curr_yTargetIn - pc_y;
        float len_diff_pe_pc = sqrtf(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
        float len_diff_tgt_pc = sqrtf(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
        float a_x = diff_pe_pc_x / len_diff_pe_pc;
        float a_y = diff_pe_pc_y / len_diff_pe_pc;
        float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
        float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
        float a_dot_b = a_x * b_x + a_y * b_y;
        if (a_dot_b > 1.f)
          a_dot_b = 1.f;
        else if (a_dot_b < -1.f)
          a_dot_b = -1.f;
        angle = acosf(a_dot_b) * (180.f / PI);
        // Determine angle direction
        float direction = a_x * b_y - a_y * b_x;
        if (direction < 0.f)
          angle = -angle;
        // Make the result look more natural (these checks may be omitted)
        if (angle > 30.f)
          angle = 30.f;
        else if (angle < -30.f)
          angle = -30.f;
        // Save angle
        angle_out[iter - 1] = angle;
        for (int i = 0; i < NUM_JOINTS; i++) {
          if(i < NUM_JOINTS - 1) {
            angle_out[i+1] += angle_out[i];
          }
        }
      }
    }
    angle_out_h[idx * NUM_JOINTS + 0] = angle_out[0];
    angle_out_h[idx * NUM_JOINTS + 1] = angle_out[1];
    angle_out_h[idx * NUM_JOINTS + 2] = angle_out[2];
  }
}

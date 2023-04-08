#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include "points2image.h"
#include "platform_atomics.h"

/** 
 * The kernel to execute.
 * Performs the transformation for a single point.
 * cp: pointer to cloud memory
 * msg_distance: image distances
 * msg_intensity: image intensities
 * msg_min_height: image minimum heights
 * width: cloud width
 * height: cloud height
 * point_step: point stride used for indexing cloud memory
 * w: image width
 * h: image height
 * invR: matrix to apply to cloud points
 * invT: translation to apply to cloud points
 * distCoeff: distance coefficients to apply to cloud points
 * cameraMat: camera intrinsic matrix
 * min_y: lower image extend bound
 * max_y: higher image extend bound
 */
void compute_point_from_pointcloud(
    const float*  __restrict cp, // cloud data pointer to read the data correctly
          float* volatile msg_distance,
          float* volatile msg_intensity,
          float* __restrict msg_min_height,
          float* __restrict msg_max_height,
    int width, int height, int point_step,
    int w, int h, Mat33 invR, Mat13 invT,
    Vec5 distCoeff, Mat33 cameraMat,
    int* __restrict miny, int* __restrict maxy) {
  int min_y = -1, max_y = h;
  // determine index in cloud memory
  for (int y = 0; y < height; y++) {
    #pragma omp parallel for reduction(min:min_y) reduction(max:max_y) schedule(static)
    for (int x = 0; x < width; x++) {
      const float* fp = (float *)((uintptr_t)cp + (x + y*width) * point_step);
      float intensity = fp[4];
      // first step of the transformation
      Mat13 point, point2;
      point2.data[0] = double(fp[0]);
      point2.data[1] = double(fp[1]);
      point2.data[2] = double(fp[2]);
      for (int row = 0; row < 3; row++) {
        point.data[row] = invT.data[row];
        for (int col = 0; col < 3; col++) 
          point.data[row] += point2.data[col] * invR.data[row][col];
      }
      // discard points of low depth
      if (point.data[2] <= 2.5) continue;
      // second transformation step
      double tmpx = point.data[0] / point.data[2];
      double tmpy = point.data[1] / point.data[2];
      double r2 = tmpx * tmpx + tmpy * tmpy;
      double tmpdist = 1.0 + distCoeff.data[0] * r2
                       + distCoeff.data[1] * r2 * r2
                       + distCoeff.data[4] * r2 * r2 * r2;
      Point2d imagepoint;
      imagepoint.x = tmpx * tmpdist + 2.0 * distCoeff.data[2] * tmpx * tmpy
                     + distCoeff.data[3] * (r2 + 2.0 * tmpx * tmpx);
      imagepoint.y = tmpy * tmpdist + distCoeff.data[2] * (r2 + 2.0 * tmpy * tmpy)
                     + 2.0 * distCoeff.data[3] * tmpx * tmpy;
      // apply camera intrinsics to yield a point on the image
      imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
      imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];
      int px = int(imagepoint.x + 0.5);
      int py = int(imagepoint.y + 0.5);

      // continue with points inside image bounds
      if (0 <= px && px < w && 0 <= py && py < h) {
        int pid = py * w + px;
        float cm_point = point.data[2] * 100.0;  // double precision multiply
        // update intensity, height and image extends
        #pragma omp critical
        {
          if (msg_distance[pid] == 0 || msg_distance[pid] > cm_point) {
            msg_distance[pid] = cm_point;
            msg_intensity[pid] = float(intensity);
            max_y = py > max_y ? py : max_y;
            min_y = py < min_y ? py : min_y;
          }
        }
        #pragma omp critical
        {
          //process simultaneously min and max during the first layer
          if (0 == y && height == 2) {
            float* fp2 = (float *)(cp + (x + (y+1)*width) * point_step);
            msg_min_height[pid] = fp[2];
            msg_max_height[pid] = fp2[2];
          } else {
            msg_min_height[pid] = -1.25;
            msg_max_height[pid] = 0;
          }
        }
      }
    }
  }
  *maxy = max_y;
  *miny = min_y;
}


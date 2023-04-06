#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define WIDTH 800
#define HEIGHT 600

struct Point {
  double x, y, z;
};

struct Sphere {
  struct Point center;
  double radius;
};

double dot(struct Point a, struct Point b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(struct Point p) {
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

struct Point normalize(struct Point p) {
  double n = norm(p);
  struct Point result = {p.x / n, p.y / n, p.z / n};
  return result;
}

int ray_intersect_sphere(struct Point ray_origin, struct Point ray_direction, struct Sphere sphere, double* t) {
  double a = dot(ray_direction, ray_direction);
  struct Point distance = {ray_origin.x - sphere.center.x, ray_origin.y - sphere.center.y, ray_origin.z - sphere.center.z};
  double b = 2 * dot(ray_direction, distance);
  double c = dot(distance, distance) - sphere.radius * sphere.radius;
  double discriminant = b * b - 4 * a * c;
  if (discriminant < 0) {
    return 0;
  } else {
    double t0 = (-b - sqrt(discriminant)) / (2 * a);
    double t1 = (-b + sqrt(discriminant)) / (2 * a);
    if (t0 > 0) {
      *t = t0;
      return 1;
    } else if (t1 > 0) {
      *t = t1;
      return 1;
    } else {
      return 0;
    }
  }
}

int main() {
  struct Sphere sphere = {{0, 0, -10}, 3};
  double image[WIDTH][HEIGHT];

  double start_time = omp_get_wtime();
  #pragma omp parallel for schedule(dynamic)
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      struct Point ray_direction = {x - WIDTH / 2, y - HEIGHT / 2, 1};
      ray_direction = normalize(ray_direction);
      struct Point ray_origin = {0, 0, 0};
      double t;
      if (ray_intersect_sphere(ray_origin, ray_direction, sphere, &t)) {
        image[x][y] = t;
      } else {
        image[x][y] = 0;
      }
    }
  }
  double end_time = omp_get_wtime();
  printf("Time taken: %f seconds\n", end_time - start_time);
  return 0;
}


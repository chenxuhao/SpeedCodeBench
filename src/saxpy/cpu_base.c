void saxpy(const int n, const float a, const float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
}

#include <iostream>

void rayTracing(int nx, int ny, int ns);

int main() {
  int nx = 1200;
  int ny = 800;
  int ns = 10;
  std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel \n";
  rayTracing(nx, ny, ns);
}


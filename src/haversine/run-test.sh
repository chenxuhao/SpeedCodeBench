#!/bin/bash

FILE=./locations.txt
if [ ! -f "$FILE" ]; then
  wget https://github.com/zjin-lcf/HeCBench/raw/master/geodesic-sycl/locations.tar.gz
  tar zxvf locations.tar.gz
fi
../../bin/haversine_omp_base locations.txt 100

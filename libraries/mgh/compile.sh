#!/bin/bash
gfortran -O3 -c -o set_precision.o fortran/set_precision.f08
gfortran -O3 -c -o mgh.o fortran/mgh.f08 -lm
mex -largeArrayDims set_precision.o mgh.o mgh.F90

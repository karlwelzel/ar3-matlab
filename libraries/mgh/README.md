# MATLAB interface to Moré, Garbow, and Hillstrom test set problems

The file `mgh.F90` contains the Matlab interface (MEX file) to the Fortran code written by E. G. Birgin, J. L. Gardenghi, J. M. Martinez and S. A. Santos inside the `fortran` folder.
The code evaluates the first three derivatives for a range of problems from the Moré, Garbow, and Hillstrom test set of optimization problems.
It is available at https://github.com/johngardenghi/mgh/ and https://www.ime.usp.br/~egbirgin/sources/bgms2/.
For more information consult the technical report "Third-order derivatives of the Moré, Garbow, and Hillstrom test set problems" by the same authors at https://www.ime.usp.br/~egbirgin/publications/bgms2.pdf.

The interface was written by Nick Gould and slightly adapted by Karl Welzel.


## Compilation

To compile the Matlab interface file on Linux, run `./compile.sh` or follow the steps within it.


## Usage

The simplest way of using the interface is through `mgh_function`.
It takes the problem number (between 1 and 35) and optionally the dimension and returns the associated test problem.
Its signature is
```MATLAB
function [n, m, name, x0, f_handle] = mgh_function(problem_number, dimension)
```
where `n` is problem dimension, `m` is the number of components of the function (problem-dependent), `name` is the name of the test problem, `x0` is the default starting point of the optimization problem, and `f_handle` is a function handle to evaluate the function and its derivatives.

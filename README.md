
The test case for the one-dimensional Burger's Equation with stochastic forcing is based on the publication by:

Sukanta Basu
Can the dynamic eddy-viscosity class of subgrid-scale models capture inertial-range properties of Burgers turbulence?
https://doi.org/10.1080/14685240902852719

The code was written in Python v3.8.5 and imported the following modules:
  * NumPy 1.21.1
  * dataclasses 0.8
  * pandas 1.1.3
  * SciPy 1.5.2
  * configparser 3.12.1

The application of Parareal with and without spatial coarsening to Burger's Equation with stochastic forcing 
is intended to provide a playground for various numerical methods. The test case mimics turbulent characteristics
and allows for the approximation of the diagnostics pseudo-turbulent kinetic energy and energy spectrum. 

Comparatively simple to setup, the test case allows for investigations of the impact of interpolation, time integration
and spatial discretization combinations on the convergence behavior of Parareal. The execution of the time-parallel algorithms is
serialized. The Python code comes with a configuration file that allows for various modifications for conceivable numerical 
experiment settings. 

For future purposed the implementation of FEM and FVM discretizations is considered.
  

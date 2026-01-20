# Installation
pip install -r requirements.txt
streamlit run app.py

# ode_solver
Interactive Laplace-based solver for second-order linear ODEs with visualization.

# Second-Order Linear ODE Solver (Symbolic + Visualization)

An interactive Streamlit application for solving second-order linear ODEs with constant coefficients using:

- Symbolic Laplace transforms
- Symbolic convolution (Green's function)
- Exact handling of real, repeated, and complex roots
- Phase plane visualization
- Arbitrary forcing functions (including Heaviside)

## Features
- Solves: y'' + b y' + c y = x(t)
- Exact symbolic solution with applied initial conditions
- Time-domain plot of y(t)
- Phase plane plot (y vs y')
- Automatic handling of complex conjugate roots
- Matches Wolfram Alpha on standard test cases

## Example
```math
y'' + 2y' + 5y = \sin(t), \quad y(0)=1, \; y'(0)=0

Bayesian Estimation of 1D Diffusion Models
==========================================

Sometimes the dynamics of a high-dimensional system, such as a protein, can
modeled as diffusion along a 1D reaction coordinate. Given a timeseries of a 1D
reaction coordinate, x, this module estimates the most-likely potential V(x)
and diffusion coefficient D(x) that reproduced the observed dynamics, assuming
they evolved from a 1D Smoluchowski diffusion equation.

References:


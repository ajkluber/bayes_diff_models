Bayesian Estimation of 1D Diffusion Models
==========================================

Sometimes the dynamics of a high-dimensional system, such as a protein, can
modeled as diffusion along a 1D reaction coordinate. Given a timeseries of a 1D
reaction coordinate, x, this module estimates the most-likely potential V(x)
and diffusion coefficient D(x) that reproduced the observed dynamics, assuming
they evolved from a 1D Smoluchowski diffusion equation.

References:
-----------
1. Hummer, G. Position-Dependent Diffusion Coefficients and Free Energies from
Bayesian Analysis of Equilibrium and Replica Molecular Dynamics Simulations.
New J. Phys. 2005, 7, 34–34. 

2. Best, R. B.; Hummer, G. Diffusion Models of Protein Folding. Phys. Chem.
Chem. Phys. 2011, 13, 16902–16911.


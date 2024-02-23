## EBPhaseField
Code repository for the paper "A phase-field model for the brittle fracture of Euler-Bernoulli beams", authors G. Corsi, A. Favata and S. Vidoli.
Submitted to CMAME

# Dependencies
* legacy FEniCS (with PETSc): latest version 2019.2.0.dev0 from the repository was used 
* meshio, pygmsh, gmsh python packages

# Contents
Two scripts are provided:
* Euler-Bernoulli beam fracture (clamped_clamped_beam_CDG.py): the damage model introduced in the paper is demontrated. The application is the case of a beam clamped at both sides
* 2D beam fracture (frac_clamp_2D.py): a 2D phase-field damage model is used to obtain the numerical evidences shown in figure 13 of the paper (fracture of a cantilever beam-like domain under strong compressions).

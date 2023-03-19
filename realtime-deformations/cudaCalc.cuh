#pragma once
#include <material_point_method.hpp>

void cuP2G(MaterialPointMethod::Particle* particles,
           MaterialPointMethod::Cell* grid,
           int nParticles, int MAX_I, int MAX_J, int MAX_K, ftype* w);
extern ftype devH;

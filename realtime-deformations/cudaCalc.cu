#include "utils.h"
#include "material_point_method.hpp"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

__constant__ ftype devH;

namespace MaterialPointMethod {
    namespace WeightCalculator {
        __device__ ftype wip(glm::ivec3 idx, v3t pos) {
            const auto xcomp = pos.x / devH - idx.x;
            const auto ycomp = pos.y / devH - idx.y;
            const auto zcomp = pos.z / devH - idx.z;
            return weightNx(xcomp) * weightNx(ycomp) * weightNx(zcomp);
        }
    }
}

__global__
void realPrecomputeWeights(MaterialPointMethod::Particle* particles, int nParticles, int MAX_I, int MAX_J, int MAX_K, ftype* w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const long long BigIdx = (long long)nParticles * MAX_I * MAX_J * MAX_K;
    for (long long iii = index; iii < BigIdx; iii += stride) {
        auto temp = iii;
        const auto p = temp % nParticles;
        temp /= nParticles;
        const auto k = temp % MAX_K;
        temp /= MAX_K;
        const auto j = temp % MAX_J;
        temp /= MAX_J;
        const auto i = temp % MAX_I;
        w[(i * MAX_J * MAX_K + j * MAX_K + k) * nParticles + p] = MaterialPointMethod::WeightCalculator::wip({ i, j, k }, particles[p].pos);
    }
}

/*
    i, j, k   -> p[0], p[1], ..., p[n-1]
    i, j, k+1 -> p[0], p[1], ..., p[n-1]
*/
__global__
void realRasterizeMass(
    MaterialPointMethod::Particle* particles,
    MaterialPointMethod::Cell* grid,
    int nParticles, int MAX_I, int MAX_J, int MAX_K, ftype* w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float mass = 0.0f;
    // TODO parallel reduction
    if (index < MAX_I * MAX_J * MAX_K) {
        for (int p = 0; p < nParticles; ++p) {
            mass += particles[p].mass * w[index * nParticles + p];
        }
        grid[index].mass = mass;
    }
}

void cuP2G(MaterialPointMethod::Particle* particles, MaterialPointMethod::Cell* grid, int nParticles, int MAX_I, int MAX_J, int MAX_K, ftype* w) {
    int blockSize = 1024;
    int numBlocks = ((long long)nParticles * MAX_I * MAX_J * MAX_K + blockSize - 1) / blockSize;
    realPrecomputeWeights << <numBlocks, blockSize >> > (particles, nParticles, MAX_I, MAX_J, MAX_K, w);

    blockSize = 1024;
    numBlocks = ((long long)MAX_I * MAX_J * MAX_K + blockSize - 1) / blockSize;
    realRasterizeMass << <numBlocks, blockSize >> > (particles, grid, nParticles, MAX_I, MAX_J, MAX_K, w);
}


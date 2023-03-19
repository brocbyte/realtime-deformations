#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <constants.hpp>
#include <logger.hpp>
#include "cuda_runtime_api.h"

namespace MaterialPointMethod {
    const auto DEFAULT_LOG_LEVEL_OPTIMIZER = Logger::LogLevel::WARNING;
    const auto DEFAULT_LOG_LEVEL_MPM = Logger::LogLevel::WARNING;

    namespace WeightCalculator {
        extern ftype h;
        __host__ __device__ inline ftype weightNx(ftype x) {
            const ftype modx = abs(x);
            const ftype modx2 = modx * modx;
            const ftype modx3 = modx * modx * modx;
            if (modx < 1.0) {
                return 0.5 * modx3 - modx2 + 2.0 / 3.0;
            }
            if (modx < 2.0) {
                return (1.0 / 6.0) * (2 - modx) * (2 - modx) * (2 - modx);
            }
            return 0.0;
        }
        __host__ __device__ inline ftype weightNxDerivative(ftype x) {
            const auto modx = abs(x);
            const auto modx2 = modx * modx;
            if (modx < 1.0f) {
                if (x >= 0) {
                    return +3.0 / 2.0 * x * x - 2 * x;
                }
                else {
                    return -3.0 / 2.0 * x * x - 2 * x;
                }
            }
            else if (modx < 2.0) {
                if (x >= 0) {
                    return -0.5 * (2 - x) * (2 - x);
                }
                else {
                    return 0.5 * (2 + x) * (2 + x);
                }
            }
            return 0.0;
        }
        __host__ __device__ inline v3t wipGrad(glm::ivec3 idx, v3t pos) {
            const auto xcomp = (pos.x - idx.x * h) / h;
            const auto ycomp = (pos.y - idx.y * h) / h;
            const auto zcomp = (pos.z - idx.z * h) / h;
            const auto weightNxXComp = weightNx(xcomp);
            const auto weightNxYComp = weightNx(ycomp);
            const auto weightNxZComp = weightNx(zcomp);
            return {
                (1.0 / h) * weightNxDerivative(xcomp) * weightNxYComp * weightNxZComp,
                (1.0 / h) * weightNxXComp * weightNxDerivative(ycomp) * weightNxZComp,
                (1.0 / h) * weightNxXComp * weightNxYComp * weightNxDerivative(zcomp),
            };
        }
    };

    struct Particle {
        ftype mass;
        v3t velocity;
        ftype volume;
        v3t pos;
        m3t FElastic{ 1.0 };
        m3t FPlastic{ 1.0 };
        m3t B{ 0.0 };
        m3t D{ 0.0 };

        unsigned char r, g, b, a; // Color
        float size;
    };

    struct Cell {
        ftype mass;
        v3t velocity{ 0.0 };
        int nParticles{ 0 };
    };

    class Grid {
    public:
        Grid(int max_i, int max_j, int max_k)
            : MAX_I(max_i), MAX_J(max_j), MAX_K(max_k) {
            grid.resize(max_i * max_j * max_k);
            const size_t gridByteSize = MAX_I * MAX_J * MAX_K * sizeof(Cell);
            if (cudaMalloc((void**)&devGrid, gridByteSize) != cudaError::cudaSuccess) {
                std::cout << "cudaMalloc error\n";
            }
        }
        Cell& operator ()(size_t i, size_t j, size_t k) {
            return grid[i * MAX_J * MAX_K + j * MAX_K + k];
        }
        Cell operator ()(size_t i, size_t j, size_t k) const {
            return grid[i * MAX_J * MAX_K + j * MAX_K + k];
        }
        void clear() {
            memset(&grid[0], 0, grid.size() * sizeof(grid[0]));
        }
        ~Grid() {
            cudaFree(devGrid);
        }
        Cell* devGrid;
        std::vector<Cell> grid;
    private:
        int MAX_I, MAX_J, MAX_K;
    };

    class WeightStorage {
    public:
        WeightStorage(int max_i, int max_j, int max_k, int nParticles)
            : MAX_I(max_i), MAX_J(max_j), MAX_K(max_k), _nParticles(nParticles) {
            size_t wSize = (long long)MAX_I * MAX_J * MAX_K * nParticles * sizeof(ftype);
            if (cudaMalloc((void**)&devW, wSize) != cudaError::cudaSuccess) {
                std::cout << "cudaMalloc error\n";
            }
            w.resize((long long)MAX_I * MAX_J * MAX_K * nParticles);
        }

        ftype& operator ()(size_t i, size_t j, size_t k, size_t p) {
            return w[(i * MAX_I * MAX_J + j * MAX_J + k) * _nParticles + p];
        }
        ftype operator ()(size_t i, size_t j, size_t k, size_t p) const {
            return w[(i * MAX_I * MAX_J + j * MAX_J + k) * _nParticles + p];
        }

        ~WeightStorage() {
            cudaFree(devW);
        }
        ftype* devW;
        std::vector<ftype> w;
    private:
        int MAX_I, MAX_J, MAX_K, _nParticles;
    };

    struct LagrangeEulerView : Loggable {
    public:
        LagrangeEulerView(int max_i, int max_j, int max_k, int particlesNum);
        ~LagrangeEulerView();
        void initializeParticles(const v3t& particlesOrigin, const v3t& velocity);
        Particle* getParticles() {
            return particles.data();
        };
        int getNumParticles() {
            return nParticles;
        }
        void precalculateWeights();
        void rasterizeParticlesToGrid();
        void computeParticleVolumesAndDensities();
        void timeIntegration(ftype timeDelta);
        void gridBasedCollisions();

        void updateDeformationGradient(ftype timeDelta);

        void updateParticleVelocities();
        void updateParticlePositions(ftype timeDelta);

        const int MAX_I, MAX_J, MAX_K;
    private:
        std::vector<Particle> particles;
        Particle* devParticles;

        std::vector<glm::ivec3> used_cells;
        WeightStorage w;

        Grid grid;
        int nParticles;

        // constants
        const ftype mu0 = 1.0f;
        const ftype xi = 10.0f;
        const ftype lambda0 = 1.0f;

        ftype Energy(const Eigen::VectorXf& velocities, ftype timeDelta);
        ftype ElasticPotential(const Eigen::VectorXf& velocities, ftype timeDelta);
        ftype ElasticPlasticEnergyDensity(const m3t& FE, const m3t& FP);

        v3t clampPosition(const v3t& vec);

        // debugging
        ftype averagePPC();
        ftype averageParticleDensity();
        ftype averageCellDensity();

        v3t gridMomentum();
        v3t particleMomentum();
        ftype gridMass();
    };

}

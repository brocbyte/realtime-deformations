#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <constants.hpp>
#include <logger.hpp>
#include "cuda_runtime_api.h"
#include <mesh.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

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
        inline ftype wipHost(glm::ivec3 idx, v3t pos) {
            const auto xcomp = pos.x / h - idx.x;
            const auto ycomp = pos.y / h - idx.y;
            const auto zcomp = pos.z / h - idx.z;
            return weightNx(xcomp) * weightNx(ycomp) * weightNx(zcomp);
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

    class MeshCollider {
    public:
        Mesh mesh;
        MeshCollider(GLuint programID, glm::mat4& VP, const std::vector<GLfloat>& vertices, const std::vector<GLfloat>& colors, const glm::vec3& vel)
            : mesh{ programID, VP, vertices, colors }, velocity{ vel } {
            sdf = [this](const v3t& pos) -> ftype {
                glm::vec4 b4 = glm::scale(glm::mat4(), mesh.scale) * glm::vec4{ 1, 1, 1, 1 };
                glm::vec3 b = { b4.x, b4.y, b4.z };
                glm::vec4 p4 = (glm::inverse(glm::translate(glm::mat4(), mesh.translation) * glm::toMat4(mesh.rotation)) * glm::vec4{ pos, 1 });
                glm::vec3 p = { p4.x, p4.y, p4.z };
                glm::vec3 q = abs(p) - b;
                return glm::length(std::max({ q.x, q.y, q.z, 0.0f })) + std::min({ std::max(q.x, std::max(q.y, q.z)), 0.0f });
            };
        }
        std::function<ftype(const v3t&)> sdf;
        glm::vec3 velocity;
        void move(ftype timeDelta) {
            mesh.applyMatrix4(glm::translate(glm::mat4(1.0), timeDelta * velocity));
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

        m3t particleDeformationTmpValue;
        std::vector<glm::ivec3> neighs;

        unsigned char r, g, b, a; // Color
        float size;
    };

    struct Cell {
        ftype mass;
        v3t force{ 0.0f, 0.0f, 0.0f };
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
        bool useCuda = false;
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
        void computeExplicitGridForces();
        void gridVelocitiesUpdate(ftype timeDelta);
        void timeIntegration(ftype timeDelta);
        void gridBasedCollisions(ftype timeDelta, const std::vector<MeshCollider>& objects);

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

        v3t bodyCollision(const v3t& pos, const v3t& velocity, ftype currentTime, const std::vector<MeshCollider>& objects);

        ftype Energy(const Eigen::VectorXf& velocities, ftype timeDelta);
        ftype ElasticPotential(const Eigen::VectorXf& velocities, ftype timeDelta);
        ftype ElasticPlasticEnergyDensity(const m3t& FE, const m3t& FP);

        v3t clampPosition(const v3t& vec);

        // debugging
        ftype averagePPC();
        ftype averageParticleDensity();
        ftype averageCellDensity();
        m3t averageDeformationGradient();

        v3t gridMomentum();
        v3t particleMomentum();
        ftype gridMass();
    };

}

#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <stdint.h>
#include <Eigen/Dense>

typedef float ftype;
typedef glm::vec3 v3t;
typedef glm::mat3 m3t;

namespace MaterialPointMethod {
    struct GridIndex;

    struct WeightCalculator {
    public:
        static ftype wip(glm::ivec3 idx, v3t pos);
        static v3t wipGrad(glm::ivec3 idx, v3t pos);
        static float h;
    private:
        static ftype weightNx(ftype x);
        static ftype weightNxDerivative(ftype x);
    };

    struct GridIndex {
    public:
        int i, j, k;
    };

    struct Particle {
    public:
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
    public:
        ftype mass;
        v3t velocity{ 0.0 };
        v3t oldVelocity{ 0.0 };
        v3t force{ 0.0 };
        int nParticles{ 0 };
    };

    struct LagrangeEulerView {
    public:
        LagrangeEulerView(uint16_t max_i, uint16_t max_j, uint16_t max_k, uint16_t particlesNum);
        void initParticles(const glm::vec3& particlesOrigin);
        const std::vector<std::vector<std::vector<Cell>>>& getGrid() const {
            return grid;
        };
        std::vector<Particle>& getParticles() {
            return particles;
        };
        void rasterizeParticlesToGrid();
        void computeParticleVolumesAndDensities();
        void timeIntegration(ftype timeDelta, bool implicit = true);
        void updateDeformationGradient(ftype timeDelta);

        void updateParticleVelocities();
        void updateParticlePositions(ftype timeDelta);

        void saveGridVelocities();

        void printGrid();
        v3t gridMomentum();
        v3t particleMomentum();
        const int MAX_I, MAX_J, MAX_K;
    private:
        std::vector<std::vector<std::vector<Cell>>> grid;
        std::vector<Particle> particles;
        std::vector<glm::ivec3> used_cells;

        // constants
        const ftype mu0 = 1.0f;
        const ftype xi = 10.0f;
        const ftype lambda0 = 1.0f;

        ftype Energy(const Eigen::VectorXf& velocities, ftype timeDelta);
        ftype ElasticPotential(const Eigen::VectorXf& velocities, ftype timeDelta);
        ftype ElasticPlasticEnergyDensity(const m3t& FE, const m3t& FP);
    };

}

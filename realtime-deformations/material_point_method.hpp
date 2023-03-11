#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <stdint.h>
#include <Eigen/Dense>
#include <constants.hpp>
#include <logger.hpp>

namespace MaterialPointMethod {
    const auto DEFAULT_LOG_LEVEL_OPTIMIZER = Logger::LogLevel::WARNING;
    const auto DEFAULT_LOG_LEVEL_MPM = Logger::LogLevel::WARNING;

    struct WeightCalculator {
    public:
        static ftype wip(glm::ivec3 idx, v3t pos);
        static v3t wipGrad(glm::ivec3 idx, v3t pos);
        static ftype h;
    private:
        static ftype weightNx(ftype x);
        static ftype weightNxDerivative(ftype x);
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
        int nParticles{ 0 };
    };

    struct LagrangeEulerView : Loggable {
    public:
        LagrangeEulerView(uint16_t max_i, uint16_t max_j, uint16_t max_k, uint16_t particlesNum);
        void initializeParticles(const v3t& particlesOrigin, const v3t& velocity);
        const std::vector<std::vector<std::vector<Cell>>>& getGrid() const {
            return grid;
        };
        std::vector<Particle>& getParticles() {
            return particles;
        };
        void rasterizeParticlesToGrid();
        void computeParticleVolumesAndDensities();
        void timeIntegration(ftype timeDelta);
        void gridBasedCollisions();

        void updateDeformationGradient(ftype timeDelta);

        void updateParticleVelocities();
        void updateParticlePositions(ftype timeDelta);

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

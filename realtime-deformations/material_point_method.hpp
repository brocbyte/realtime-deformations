#pragma once
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <stdint.h>

namespace MaterialPointMethod {
    struct GridIndex;

    struct WeightCalculator {
    public:
        static float wip(GridIndex idx, glm::vec3 pos);
        static glm::vec3 wipGrad(GridIndex idx, glm::vec3 pos);
        static float h;
    private:
        static float weightNx(float x);
        static float weightNxDerivative(float x);
    };

    struct GridIndex {
    public:
        uint16_t i, j, k;
    };

    struct Particle {
    public:
        float mass;
        glm::vec3 velocity;
        float volume;
        glm::vec3 pos;
        glm::mat3 FElastic{ 1.0 };
        glm::mat3 FPlastic{ 1.0 };
        glm::mat3 B{ 1.0 };

        // tmp...
        float cameradistance;
        unsigned char r, g, b, a; // Color
        float size, angle, weight;
        float life; // Remaining life of the particle. if < 0 : dead and unused.

        bool operator<(const Particle& that) const {
            // Sort in reverse order : far particles drawn first.
            return this->cameradistance > that.cameradistance;
        }
    };

    struct Cell {
    public:
        float mass;
        glm::vec3 velocity{ 0.0 };
        glm::vec3 oldVelocity{ 0.0 };
        glm::vec3 starVelocity{ 0.0 };
        glm::vec3 force{ 0.0 };
        float forceLen;
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
        void computeGridForces();
        void updateVelocitiesOnGrid(float timeDelta);
        void gridBasedBodyCollisions();
        void timeIntegration(bool implicit = false);
        void updateDeformationGradient(float timeDelta);
        void particleBasedBodyCollisions();

        void updateParticleVelocities();
        void updateParticlePositions(float timeDelta);

        void saveGridVelocities();

        void printGrid();
    private:
        const uint16_t MAX_I, MAX_J, MAX_K;
        std::vector<std::vector<std::vector<Cell>>> grid;
        std::vector<Particle> particles;

        // constants
        const float mu0 = 1.0f;
        const float xi = 10.0f;
        const float lambda0 = 1.0f;

        glm::vec3 bodyCollision(const glm::vec3& pos, const glm::vec3& velocity);
    };

}

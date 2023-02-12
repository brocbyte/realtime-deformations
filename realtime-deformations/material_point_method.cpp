#include "material_point_method.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <ranges>
#include "utils.h"



namespace MaterialPointMethod {
    float WeightCalculator::h = 1.0f;
    float WeightCalculator::weightNx(float x) {
        const auto modx = abs(x);
        const auto modx3 = modx * modx * modx;
        const auto modx2 = modx * modx;
        if (modx < 1.0f) {
            return 0.5f * modx3 - modx2 + 2.0f / 3.0f;
        }
        else if (modx >= 1.0f && modx < 2.0f) {
            return (-1.0f / 6.0f) * modx3 + modx2 - 2.0f * modx + 4.0f / 3.0f;
        }
        return 0.0;
    }

    float WeightCalculator::weightIdxPoint(GridIndex idx, glm::vec3 pos) {
        const auto xcomp = (1.0f / h) * (pos.x - idx.i * h);
        const auto ycomp = (1.0f / h) * (pos.y - idx.j * h);
        const auto zcomp = (1.0f / h) * (pos.z - idx.k * h);
        return weightNx(xcomp) * weightNx(ycomp) * weightNx(zcomp);
    }

    glm::vec3 WeightCalculator::weightIdxPointGradient(GridIndex idx, glm::vec3 pos) {
        const auto eps = 0.001f;
        glm::vec3 gradient;
        glm::vec3 dfs[3] = { glm::vec3{1.0, 0, 0}, glm::vec3{0.0, 1.0, 0.0}, glm::vec3{0.0, 0.0, 1.0} };
        for (int i = 0; i < 3; ++i) {
            glm::vec3 posMinus = pos - dfs[i] * eps;
            glm::vec3 posPlus = pos + dfs[i] * eps;
            gradient[i] = (weightIdxPoint(idx, posPlus) - weightIdxPoint(idx, posMinus)) / (2.0f * eps);
        }
        return gradient;
    }

    LagrangeEulerView::LagrangeEulerView(uint16_t max_i, uint16_t max_j, uint16_t max_k, uint16_t particlesNum)
        : MAX_I(max_i), MAX_J(max_j), MAX_K(max_k) {
        grid.resize(MAX_I);
        std::for_each(std::begin(grid), std::end(grid), [=](auto& plane) {
            plane.resize(MAX_J);
            std::for_each(std::begin(plane), std::end(plane), [=](auto& row) {
                row.resize(MAX_K);
                });
            });
        particles.resize(particlesNum);
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            grid[i][j][k].mass = 0.0;
        }
        std::for_each(std::begin(particles), std::end(particles), [](Particle& p) {
            p.mass = 0.0;
            p.pos = glm::vec3(0, 0, 0);

            p.life = -1.0f;
            p.cameradistance = -1.0f;
            });
    }

    void LagrangeEulerView::initParticles() {
        std::for_each(std::begin(particles), std::end(particles), [](auto& p) {
            p.pos = glm::vec3(0, 0, -20.0f);

            float spread = 2.5f;
            glm::vec3 maindir = glm::vec3(0.0f, 20.0f, 0.0f);
            auto randFloat = [](auto low, auto high) {
                return low + (rand() % 2000 - 1000.0f) / 1000.0f * (high - low);
            };
            const auto phi = randFloat(0, 2 * 3.1415);
            const auto costheta = randFloat(-1, 1);
            const auto u = randFloat(0, 1);
            const auto theta = acos(costheta);
            const auto R = 3.0f;
            const auto r = R * std::cbrt(u);

            glm::vec3 randomdir = glm::vec3(
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta)
            );


            p.velocity = maindir + randomdir;

            p.r = rand() % 256;
            p.g = rand() % 256;
            p.b = rand() % 256;
            p.a = (rand() % 256) / 3;
            p.size = (rand() % 1000) / 2000.0f + 0.1f;
            });
    }

    void LagrangeEulerView::rasterizeParticlesToGrid() {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            grid[i][j][k].mass = std::accumulate(particles.cbegin(), particles.cend(), 0.0, [=](int acc, const auto& p) {
                return acc + p.mass * WeightCalculator::weightIdxPoint({ i, j, k }, p.pos);
                });
            grid[i][j][k].velocity = std::accumulate(particles.cbegin(), particles.cend(), glm::vec3{}, [=](const auto acc, const auto& p) {
                return acc + p.velocity * p.mass * WeightCalculator::weightIdxPoint({ i, j, k }, p.pos) / grid[i][j][k].mass;
                });
        }
    }

    void LagrangeEulerView::computeParticleVolumesAndDensities() {
        for (auto& p : particles) {
            auto density = 0.0;
            MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
                density += grid[i][j][k].mass * WeightCalculator::weightIdxPoint({ i, j, k }, p.pos);
            }
            density /= WeightCalculator::h;
            p.volume = p.mass / density;
        }
    }

    void LagrangeEulerView::computeGridForces() {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            grid[i][j][k].force =
                -std::accumulate(particles.begin(), particles.end(), glm::vec3{}, [=](glm::vec3 acc, const auto& p) {
                const glm::mat3& FE = p.FElastic; // shortcut
                const glm::mat3& FP = p.FPlastic; // shortcut
                const float& JE = glm::determinant(FE);
                const float& JP = glm::determinant(FP);
                const float J = glm::determinant(FE * FP);
                const float volume = J * p.volume;
                const float mu = mu0 * glm::exp(xi * (1 - JP));
                const float lambda = lambda0 * glm::exp(xi * (1 - JP));
                const glm::mat3 RE = glm::mat3(1.0); // TODO polar decomposition
                const glm::mat3 cauchyStress =
                    (((FE - RE) * glm::transpose(FE)) * 2.0f * mu + (glm::mat3(1.0) * lambda * (JE - 1.0f) * JE)) * (1.0f / J);
                return acc + volume * cauchyStress * WeightCalculator::weightIdxPointGradient({ i, j, k }, p.pos);
                    });
        }
    }

    void LagrangeEulerView::updateParticleVelocities() {
        const auto alpha = 0.95f;
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto picVelocity = glm::vec3{ 0.0f, 0.0f, 0.0f };
            auto flipVelocity = p.velocity;
            MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
                const auto& cell = grid[i][j][k]; // shortcut
                const auto weight = WeightCalculator::weightIdxPoint({ i, j, k }, p.pos);
                picVelocity += cell.velocity * weight;
                flipVelocity += (cell.velocity - cell.oldVelocity) * weight;
            }
            p.velocity = (1 - alpha) * picVelocity + alpha * flipVelocity;
            });
    }

    void LagrangeEulerView::updateParticlePositions(float timeDelta) {
        std::for_each(std::begin(particles), std::end(particles), [timeDelta](auto& p) {
            p.pos += p.velocity * timeDelta;
            });
    }
};

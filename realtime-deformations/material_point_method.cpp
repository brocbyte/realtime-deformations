#include "material_point_method.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include "utils.h"
#include <iostream>
#include <vector>
#include <functional>
#include <glm/gtx/string_cast.hpp>
#include <Eigen/Dense>
#include "mathy.hpp"
#include <limits>

#define APIC


namespace MaterialPointMethod {
    const auto massEps = 0.0f;
    ftype WeightCalculator::h = 0.6f;
    inline ftype WeightCalculator::weightNx(ftype x) {
        const auto modx = abs(x);
        const auto modx3 = modx * modx * modx;
        const auto modx2 = modx * modx;
        if (modx < 1.0) {
            return 0.5 * modx3 - modx2 + 2.0 / 3.0;
        }
        else if (modx >= 1.0 && modx < 2.0) {
            return (1.0 / 6.0) * (2 - modx) * (2 - modx) * (2 - modx);
        }
        return 0.0;
    }

    inline ftype WeightCalculator::weightNxDerivative(ftype x) {
        const auto modx = abs(x);
        const auto modx3 = modx * modx * modx;
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

    inline ftype WeightCalculator::wip(glm::ivec3 idx, v3t pos) {
        const auto xcomp = (pos.x - idx.x * h) / h;
        const auto ycomp = (pos.y - idx.y * h) / h;
        const auto zcomp = (pos.z - idx.z * h) / h;
        return weightNx(xcomp) * weightNx(ycomp) * weightNx(zcomp);
    }

    inline v3t WeightCalculator::wipGrad(glm::ivec3 idx, v3t pos) {
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
        used_cells.reserve(MAX_I * MAX_J * MAX_K);
    }

    void LagrangeEulerView::initParticles(const glm::vec3& particlesOrigin) {
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto randFloat = [](auto low, auto high) {
                return low + (rand() % 2000) / 2000.0 * (high - low);
            };
            const auto phi = randFloat(0.0f, 2.0 * 3.1415);
            const auto costheta = randFloat(0.0, 2.0) - 1.0;
            const auto u = randFloat(0.0, 1.0);
            const auto theta = acos(costheta);
            const auto R = 1.0;
            const auto r = R * std::cbrt(u);

            const auto randomDir = v3t(
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta)
            );
            p.pos = particlesOrigin + (ftype)0.6 * randomDir;
            p.pos.x = std::clamp(p.pos.x, (ftype)0.0, MAX_I * WeightCalculator::h);
            p.pos.y = std::clamp(p.pos.y, (ftype)0.0, MAX_J * WeightCalculator::h);
            p.pos.z = std::clamp(p.pos.z, (ftype)0.0, MAX_K * WeightCalculator::h);

            const glm::ivec3 gridPos = p.pos / WeightCalculator::h;
            grid[gridPos.x][gridPos.y][gridPos.z].nParticles++;

            p.velocity = v3t(0.0, -20.0, 0.0);

            p.r = rand() % 256;
            p.g = rand() % 256;
            p.b = rand() % 256;
            p.a = (rand() % 256);

            p.r = 255;
            p.g = 255;
            p.b = 255;
            p.a = 255;

            p.size = 0.02;
            p.mass = 1.0;
            });
        auto sum = 0;
        auto cnt = 0;
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            if (grid[i][j][k].nParticles > 0) {
                sum += grid[i][j][k].nParticles;
                cnt++;
            }
        }
        std::cout << "Avg PPC: " << (ftype)sum / cnt << "\n";
        std::cout << "Particles momentum: " << glm::to_string(particleMomentum()) << "\n";
    }

    void LagrangeEulerView::rasterizeParticlesToGrid() {
        used_cells.clear();
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            grid[i][j][k].mass = std::accumulate(particles.cbegin(), particles.cend(), 0.0, [idx = glm::ivec3(i, j, k)](const int acc, const auto& p) {
                return acc + p.mass * WeightCalculator::wip(idx, p.pos);
                });
            if (grid[i][j][k].mass > massEps) {
                used_cells.push_back(glm::ivec3(i, j, k));
            }
        }
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto Dp = m3t(0.0);
            MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
                const auto Xi = v3t(i, j, k) * WeightCalculator::h;
                const auto XiXp = Xi - p.pos;
                Dp += WeightCalculator::wip({ i, j, k }, p.pos) * glm::outerProduct(XiXp, XiXp);
            }
            p.D = Dp;
            });
        for (const auto& idx : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
            const auto momentum = std::accumulate(particles.cbegin(), particles.cend(), v3t(0, 0, 0), [=](const auto acc, const auto& p) {
                const auto weight = WeightCalculator::wip(idx, p.pos);
                const auto Xi = v3t(idx) * WeightCalculator::h;
                return acc + weight * p.mass * (p.velocity + p.B * glm::inverse(p.D) * (Xi - p.pos));
                });
            grid[i][j][k].velocity = momentum / grid[i][j][k].mass;
        }
        std::cout << "Grid momentum after rasterization: " << glm::to_string(gridMomentum()) << "\n";
    }

    void LagrangeEulerView::computeParticleVolumesAndDensities() {
        auto sum = 0.0;
        auto cnt = 0;
        for (auto& p : particles) {
            auto density = 0.0;
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                density += grid[i][j][k].mass * WeightCalculator::wip(idx, p.pos);
            }
            density /= (WeightCalculator::h * WeightCalculator::h * WeightCalculator::h);
            sum += density;
            if (density > 0.0f) {
                cnt++;
            }
            p.volume = density > 0 ? p.mass / density : 0;
        }
        std::cout << "Avg particle density: " << sum / cnt << "\n";
        auto cellDensity = 0.0f;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            cellDensity += grid[i][j][k].mass;
        }
        std::cout << "Avg cell density: " << cellDensity / used_cells.size() << "\n";
    }

    ftype LagrangeEulerView::Energy(const Eigen::VectorXf& velocities, ftype timeDelta) {
        auto energy = 0.0;
        auto velIdx = 0;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            const auto& rCell = grid[i][j][k]; // shortcut
            const auto velocityNew = v3t(velocities[velIdx], velocities[velIdx + 1], velocities[velIdx + 2]);
            const auto velDiffNorm = glm::length(velocityNew - rCell.velocity);
            energy += 0.5 * rCell.mass * velDiffNorm * velDiffNorm;
            velIdx += 3;
        }
        energy += ElasticPotential(velocities, timeDelta);
        return energy;
    }

    ftype LagrangeEulerView::ElasticPotential(const Eigen::VectorXf& velocities, ftype timeDelta) {
        return std::accumulate(std::cbegin(particles), std::cend(particles), 0.0f, [&](const auto acc, const auto& p) {
            const auto& FP = p.FPlastic;
            auto FE = m3t(1.0);
            auto velIdx = 0;
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                const auto velocity = v3t(velocities[velIdx], velocities[velIdx + 1], velocities[velIdx + 2]);
                FE += glm::outerProduct(velocity * timeDelta, WeightCalculator::wipGrad(idx, p.pos));
                velIdx += 3;
            }
            FE = FE * p.FElastic;
            return acc + p.volume * ElasticPlasticEnergyDensity(FE, FP);
            });
    }

    ftype LagrangeEulerView::ElasticPlasticEnergyDensity(const m3t& FE, const m3t& FP) {
        auto mu = [=](const auto& fp) {
            return mu0 * glm::exp(xi * 1 - glm::determinant(fp));
        };
        auto lambda = [=](const auto& fp) {
            return lambda0 * glm::exp(xi * 1 - glm::determinant(fp));
        };
        const auto& [RE, SE] = polarDecomposition(FE);
        auto fNorm2 = [](const auto& m) -> float {
            auto val = 0.0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    val += m[i][j] * m[i][j];
            return val;
        };
        const auto JE = glm::determinant(FE);
        return mu(FP) * fNorm2(FE - RE) + lambda(FP) / 2 * (JE - 1) * (JE - 1);
    }

    void LagrangeEulerView::timeIntegration(float timeDelta, bool implicit) {
        auto E = [&](const Eigen::VectorXf& vel) {
            return Energy(vel, timeDelta);
        };
        Eigen::VectorXf initialVelocities(used_cells.size() * 3);
        auto velIdx = 0;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            initialVelocities[velIdx] = grid[i][j][k].velocity[0];
            initialVelocities[velIdx + 1] = grid[i][j][k].velocity[1];
            initialVelocities[velIdx + 2] = grid[i][j][k].velocity[2];
            velIdx += 3;
        }
        const auto velocities = optimize(E, initialVelocities, 0.1);
        velIdx = 0;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            grid[i][j][k].velocity = glm::vec3(velocities[velIdx], velocities[velIdx + 1], velocities[velIdx + 2]);
            velIdx += 3;
        }
        std::cout << "Grid momentum after optimization: " << glm::to_string(gridMomentum()) << "\n";
    }

    void LagrangeEulerView::updateDeformationGradient(ftype timeDelta) {
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto velGradient = m3t(0.0);
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                const auto grad = WeightCalculator::wipGrad(idx, p.pos);
                velGradient += glm::outerProduct(grid[i][j][k].velocity, grad);
            }
            const auto FEpKryshka = (m3t(1.0) + velGradient * timeDelta) * p.FElastic;
            const auto FPpKryshka = p.FPlastic;

            const auto FPn1 = FEpKryshka * FPpKryshka;
            const auto m = glmToEigen(FEpKryshka);
            Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(m);
            if (svd.info() != Eigen::ComputationInfo::Success) {
                std::cout << "Error!: " << svd.info() << "\n";
                std::cout << glmToEigen(velGradient) << "\n";
                return;
            }
            Eigen::VectorXf s = svd.singularValues();
            for (int i = 0; i < 3; i++) {
                s(i) = std::clamp(s(i), (float)(1 - 1.9f * 1e-2), (float)(1 + 7.5f * 1e-3));
            }
            Eigen::MatrixXf _S{ {s(0), 0, 0}, {0, s(1), 0}, {0, 0, s(2)} };
            const auto U = eigenToGlm(svd.matrixU());
            const auto V = eigenToGlm(svd.matrixV());
            const auto S = eigenToGlm(_S);
            const auto pFElastic = U * S * glm::transpose(V);
            const auto SInv = glm::inverse(S);
            const auto pFPlastic = V * SInv * glm::transpose(U) * FPn1;

            p.FElastic = pFElastic;
            p.FPlastic = pFPlastic;
            });
    }

    void LagrangeEulerView::updateParticleVelocities() {
        std::cout << "Grid momentum before to particles: " << glm::to_string(gridMomentum()) << "\n";
#ifdef APIC
        auto particlesMomentum = glm::vec3(0.0f, 0.0f, 0.0f);
        std::for_each(std::begin(particles), std::end(particles), [&](auto& p) {
            p.velocity = glm::vec3(0.0f, 0.0f, 0.0f);
            p.B = glm::mat3(0.0f);
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                const auto weight = WeightCalculator::wip(idx, p.pos);
                p.velocity += grid[i][j][k].velocity * weight;
                const auto Xi = v3t(idx) * WeightCalculator::h;
                p.B += weight * glm::outerProduct(grid[i][j][k].velocity, (Xi - p.pos));
            }
            particlesMomentum += p.velocity * p.mass;
            });
        std::cout << "Particles momentum: " << glm::to_string(particleMomentum()) << "\n";

#else
        const ftype alpha = 0.95;
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto picVelocity = v3t{ 0.0f, 0.0f, 0.0f };
            auto flipVelocity = p.velocity;
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                const auto& cell = grid[i][j][k]; // shortcut
                const auto weight = WeightCalculator::wip(idx, p.pos);
                picVelocity += cell.velocity * weight;
                flipVelocity += (cell.velocity - cell.oldVelocity) * weight;
            }
            p.velocity = picVelocity * (1 - alpha) + (flipVelocity * alpha);
            });
#endif
    }

    void LagrangeEulerView::updateParticlePositions(ftype timeDelta) {
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            p.pos += p.velocity * timeDelta;
            std::cout << glm::to_string(p.velocity) << "\n";
            p.pos.x = std::clamp(p.pos.x, (ftype)0.0, MAX_I * WeightCalculator::h);
            p.pos.y = std::clamp(p.pos.y, (ftype)0.0, MAX_J * WeightCalculator::h);
            p.pos.z = std::clamp(p.pos.z, (ftype)0.0, MAX_K * WeightCalculator::h);
            });
    }
    void LagrangeEulerView::saveGridVelocities() {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            auto& cell = grid[i][j][k]; // shortcut
            cell.oldVelocity = cell.velocity;
        }
    }
    void LagrangeEulerView::printGrid() {
    }
    glm::vec3 LagrangeEulerView::gridMomentum() {
        glm::vec3 momentum(0, 0, 0);
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            momentum += grid[i][j][k].velocity * grid[i][j][k].mass;
        }
        return momentum;
    }

    glm::vec3 LagrangeEulerView::particleMomentum() {
        return std::accumulate(std::cbegin(particles), std::cend(particles), v3t(0, 0, 0), [&](const auto acc, const auto& p) {
            return acc + p.velocity * p.mass;
            });
    }
};

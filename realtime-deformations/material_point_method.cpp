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
            return (1.0f / 6.0f) * (2 - modx) * (2 - modx) * (2 - modx);
        }
        return 0.0;
    }

    float WeightCalculator::weightNxDerivative(float x) {
        const auto modx = abs(x);
        const auto modx3 = modx * modx * modx;
        const auto modx2 = modx * modx;
        if (modx < 1.0f) {
            if (x >= 0) {
                return +3.0f / 2.0f * x * x - 2 * x;
            }
            else {
                return -3.0f / 2.0f * x * x - 2 * x;
            }
        }
        else if (modx >= 1.0f && modx < 2.0f) {
            if (x >= 0) {
                return -0.5f * (2 - x) * (2 - x);
            }
            else {
                return 0.5f * (2 + x) * (2 + x);
            }
        }
        return 0.0;
    }

    float WeightCalculator::wip(GridIndex idx, glm::vec3 pos) {
        const auto xcomp = (pos.x - idx.i * h) / h;
        const auto ycomp = (pos.y - idx.j * h) / h;
        const auto zcomp = (pos.z - idx.k * h) / h;
        return weightNx(xcomp) * weightNx(ycomp) * weightNx(zcomp);
    }

    glm::vec3 WeightCalculator::wipGrad(GridIndex idx, glm::vec3 pos) {
        const auto xcomp = (pos.x - idx.i * h) / h;
        const auto ycomp = (pos.y - idx.j * h) / h;
        const auto zcomp = (pos.z - idx.k * h) / h;
        return {
            (1.0f / h) * weightNxDerivative(xcomp) * weightNx(ycomp) * weightNx(zcomp),
            (1.0f / h) * weightNx(xcomp) * weightNxDerivative(ycomp) * weightNx(zcomp),
            (1.0f / h) * weightNx(xcomp) * weightNx(ycomp) * weightNxDerivative(zcomp),
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
                return low + (rand() % 2000) / 2000.0f * (high - low);
            };
            const auto phi = randFloat(0.0f, 2.0f * 3.1415);
            const auto costheta = randFloat(0.0f, 2.0f) - 1.0f;
            const auto u = randFloat(0.0f, 1.0f);
            const auto theta = acos(costheta);
            const auto R = 1.0f;
            const auto r = R * std::cbrt(u);

            glm::vec3 randomDir = glm::vec3(
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta)
            );
            p.pos = particlesOrigin + 0.5f * randomDir;
            p.pos.x = std::clamp(p.pos.x, 0.001f, MAX_I * WeightCalculator::h);
            p.pos.y = std::clamp(p.pos.y, 0.001f, MAX_J * WeightCalculator::h);
            p.pos.z = std::clamp(p.pos.z, 0.001f, MAX_K * WeightCalculator::h);

            p.velocity = glm::vec3{ 0.0f, -30.0f, 0.0f };

            p.r = rand() % 256;
            p.g = rand() % 256;
            p.b = rand() % 256;
            p.a = (rand() % 256) / 3;

            //p.r = 255;
            //p.g = 255;
            //p.b = 255;

            p.size = 0.02f;
            p.mass = 0.01f;
            });
    }

    void LagrangeEulerView::rasterizeParticlesToGrid() {
        auto cnt = 0;
        used_cells.clear();
        const auto prod = glm::outerProduct(glm::vec3{ 1.0, 2.0, 3.0 }, glm::vec3{ -4.0, 5.0, 9.0 });
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            const GridIndex idx = { i, j, k };
            grid[i][j][k].mass = std::accumulate(particles.cbegin(), particles.cend(), 0.0, [&idx](const int acc, const auto& p) {
                return acc + p.mass * WeightCalculator::wip(idx, p.pos);
                });
            if (grid[i][j][k].mass > 0) {
                used_cells.push_back(glm::ivec3{ i, j, k });
                cnt++;
            }
        }

        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            const auto momentum = std::accumulate(particles.cbegin(), particles.cend(), glm::vec3{ 0, 0, 0 }, [=](const auto acc, const auto& p) {
                return acc + p.velocity * p.mass * WeightCalculator::wip({ i, j, k }, p.pos);
                });
            grid[i][j][k].velocity = grid[i][j][k].mass != 0.0f ? momentum / grid[i][j][k].mass
                : glm::vec3{}; // doing mass < eps is not that good, see paper
        }
    }

    void LagrangeEulerView::computeParticleVolumesAndDensities() {
        for (auto& p : particles) {
            auto density = 0.0;
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                density += grid[i][j][k].mass * WeightCalculator::wip({ i, j, k }, p.pos);
            }
            density /= (WeightCalculator::h * WeightCalculator::h * WeightCalculator::h);
            p.volume = (density > 0 ? p.mass / density : 0) * 1e-1;
        }
    }


    void LagrangeEulerView::computeGridForces() {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            grid[i][j][k].force =
                -std::accumulate(std::cbegin(particles), std::cend(particles), glm::vec3{}, [=](auto acc, const auto& p) {
                const glm::mat3& FE = p.FElastic; // shortcut
                const glm::mat3& FP = p.FPlastic; // shortcut
                const auto JE = glm::determinant(FE);
                const auto JP = glm::determinant(FP);
                const auto J = glm::determinant(FE * FP);
                const float volume = J * p.volume;
                const float mu = mu0 * glm::exp(xi * (1 - JP));
                const float lambda = lambda0 * glm::exp(xi * (1 - JP));

                const auto& [RE, SE] = polarDecomposition(FE);
                checkPolarDecomposition(RE, SE, FE);

                glm::mat3 cauchyStress{ 1.0 };
                if (abs(J) > 0) {
                    cauchyStress = (FE - RE) * glm::transpose(FE) * 2.0f * mu / J + glm::mat3(1.0) * lambda * (JE - 1.0f) * JE / J;
                }
                return acc + cauchyStress * volume * WeightCalculator::wipGrad({ i, j, k }, p.pos);
                    });
        }
    }

    void LagrangeEulerView::updateVelocitiesOnGrid(float timeDelta) {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            auto& cell = grid[i][j][k];
            cell.starVelocity = glm::vec3{};
            if (cell.mass <= 0.001)
                continue;
            cell.starVelocity = cell.oldVelocity + cell.force * timeDelta * (1.0f / cell.mass);
        }
    }


    glm::vec3 LagrangeEulerView::bodyCollision(const glm::vec3& pos, const glm::vec3& velocity) {
        std::vector<std::function<float(const glm::vec3&)>> sdfs;

        //sdfs.push_back([](const glm::vec3& pos) {
        //    return pos.y + (pos.x - 3) * (pos.x - 3) - 2;
        //    });
        sdfs.push_back([](const glm::vec3& pos) {
            return std::max(pos.y - pos.x + 1, pos.y + pos.x - 5);
            });

        return std::accumulate(std::cbegin(sdfs), std::cend(sdfs), velocity,
            [savedDistance = -std::numeric_limits<float>::infinity(), &pos, &velocity](const auto acc, const auto& sdf) mutable {
                const auto currentDistance = sdf(pos);
                // need to pick sdf with maximum negative distance
                if (currentDistance > 0 || currentDistance < savedDistance) {
                    return acc;
                }
                savedDistance = currentDistance;
                const auto normal = gradient(sdf)(pos);
                const auto objectVelocity = glm::vec3{}; // TODO moving objects...
                const auto relVelocity = velocity - objectVelocity;
                const float vn = glm::dot(relVelocity, normal);
                if (vn >= 0) {
                    return acc;
                }
                const auto vt = relVelocity - normal * vn;
                const auto mu = 0.5f;
                auto vrel = glm::vec3{};
                if (vt.length() > -mu * vn && vt.length() > 0) {
                    vrel = vt + vt * (mu * vn / vt.length());
                }
                return vrel + objectVelocity;
            });
    }

    void LagrangeEulerView::gridBasedBodyCollisions() {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            auto& cell = grid[i][j][k];
            const auto gridPosition = glm::vec3(i, j, k) * WeightCalculator::h;
            cell.starVelocity = bodyCollision(gridPosition, cell.starVelocity);
        }
    }

    void LagrangeEulerView::timeIntegration(bool implicit) {
        if (!implicit) {
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                grid[i][j][k].velocity = grid[i][j][k].starVelocity;
            }
            return;
        }
        // Conjugate residual method (10-30 iterations)

    }

    void LagrangeEulerView::updateDeformationGradient(float timeDelta) {
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto velGradient = glm::mat3{};
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                velGradient += grid[i][j][k].velocity * WeightCalculator::wipGrad({ i, j, k }, p.pos);
            }
            const auto FEpKryshka = (glm::mat3(1.0) + velGradient * timeDelta) * p.FElastic;
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
                s(i) = std::clamp(s(i), (float)(1 - 1.9f * 1e-2), (float)(1 + 5.0f * 1e-3));
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

    void LagrangeEulerView::particleBasedBodyCollisions() {
        std::for_each(std::begin(particles), std::end(particles), [&](auto& p) {
            p.velocity = bodyCollision(p.pos, p.velocity);
            });
    }

    void LagrangeEulerView::updateParticleVelocities() {
        const auto alpha = 0.95f;
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto picVelocity = glm::vec3{ 0.0f, 0.0f, 0.0f };
            auto flipVelocity = p.velocity;
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                const auto& cell = grid[i][j][k]; // shortcut
                const auto weight = WeightCalculator::wip({ i, j, k }, p.pos);
                picVelocity += cell.velocity * weight;
                flipVelocity += (cell.velocity - cell.oldVelocity) * weight;
            }
            p.velocity = picVelocity * (1 - alpha) + (flipVelocity * alpha);
            });
    }

    void LagrangeEulerView::updateParticlePositions(float timeDelta) {
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            p.pos += p.velocity * timeDelta;
            p.pos.x = std::clamp(p.pos.x, 0.0001f, MAX_I * WeightCalculator::h);
            p.pos.y = std::clamp(p.pos.y, 0.0001f, MAX_J * WeightCalculator::h);
            p.pos.z = std::clamp(p.pos.z, 0.0001f, MAX_K * WeightCalculator::h);
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
};

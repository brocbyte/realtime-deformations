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
            return (-1.0f / 6.0f) * modx3 + modx2 - 2.0f * modx + 4.0f / 3.0f;
        }
        return 0.0;
    }

    float WeightCalculator::weightIdxPoint(GridIndex idx, glm::vec3 pos) {
        const auto xcomp = (pos.x - idx.i * h) / h;
        const auto ycomp = (pos.y - idx.j * h) / h;
        const auto zcomp = (pos.z - idx.k * h) / h;
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
            p.pos = particlesOrigin + 0.2f * randomDir;
            p.pos.x = std::clamp(p.pos.x, 0.001f, MAX_I * WeightCalculator::h);
            p.pos.y = std::clamp(p.pos.y, 0.001f, MAX_J * WeightCalculator::h);
            p.pos.z = std::clamp(p.pos.z, 0.001f, MAX_K * WeightCalculator::h);

            p.velocity = glm::vec3{ 30.0f, -15.0f, 0.0f };

            p.r = rand() % 256;
            p.g = rand() % 256;
            p.b = rand() % 256;
            p.a = (rand() % 256) / 3;

            p.r = 255;
            p.g = 255;
            p.b = 255;

            p.size = 0.02f;
            p.mass = 0.01f;
            });
    }

    void LagrangeEulerView::rasterizeParticlesToGrid() {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            grid[i][j][k].mass = std::accumulate(particles.cbegin(), particles.cend(), 0.0, [=](int acc, const auto& p) {
                return acc + p.mass * WeightCalculator::weightIdxPoint({ i, j, k }, p.pos);
                });
            grid[i][j][k].velocity = grid[i][j][k].mass > 0 ? std::accumulate(particles.cbegin(), particles.cend(), glm::vec3{}, [=](const auto acc, const auto& p) {
                return acc + p.velocity * p.mass * WeightCalculator::weightIdxPoint({ i, j, k }, p.pos) / grid[i][j][k].mass;
                }) : glm::vec3{};
                if (grid[i][j][k].mass > 0) {
                    std::cout << "(" << i << "," << j << "," << k << "): " << grid[i][j][k].mass << "\n";
                }
        }
    }

    void LagrangeEulerView::computeParticleVolumesAndDensities() {
        for (auto& p : particles) {
            auto density = 0.0;
            MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
                density += grid[i][j][k].mass * WeightCalculator::weightIdxPoint({ i, j, k }, p.pos);
            }
            density /= (WeightCalculator::h * WeightCalculator::h * WeightCalculator::h);
            p.volume = (density > 0 ? p.mass / density : 0) * 1e-1;
        }
    }


    void LagrangeEulerView::computeGridForces() {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
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
                return acc + cauchyStress * volume * WeightCalculator::weightIdxPointGradient({ i, j, k }, p.pos);
                    });
        }
    }

    void LagrangeEulerView::updateVelocitiesOnGrid(float timeDelta) {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
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
        //    const auto center = glm::vec3{ 2.0f, -2.0f, 2.0f };
        //    const auto radius = 3.0f;
        //    return (pos.x - center.x) * (pos.x - center.x) + (pos.y - center.y) * (pos.y - center.y) + (pos.z - center.z) * (pos.z - center.z) - radius * radius;
        //    });
        sdfs.push_back([](const glm::vec3& pos) {
            return pos.y;
            });
        //sdfs.push_back([](const glm::vec3& pos) {
        //    return pos.y - pos.x + 1;
        //    });
        //sdfs.push_back([](const glm::vec3& pos) {
        //    return pos.y + pos.x - 3;
        //    });

        std::vector<std::function<glm::vec3(const glm::vec3&)>> gradients;
        std::transform(std::cbegin(sdfs), std::cend(sdfs), std::back_inserter(gradients), [](const auto& sdf) {
            return gradient(sdf);
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
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            auto& cell = grid[i][j][k];
            const auto gridPosition = glm::vec3(i, j, k) * WeightCalculator::h;
            cell.starVelocity = bodyCollision(gridPosition, cell.starVelocity);
        }
    }

    void LagrangeEulerView::timeIntegration(bool implicit) {
        if (!implicit) {
            MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
                grid[i][j][k].velocity = grid[i][j][k].starVelocity;
            }
            return;
        }
        // Conjugate residual method (10-30 iterations)

    }

    void LagrangeEulerView::updateDeformationGradient(float timeDelta) {
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto velGradient = glm::mat3{};
            MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
                velGradient += grid[i][j][k].velocity * WeightCalculator::weightIdxPointGradient({ i, j, k }, p.pos);
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
            MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
                const auto& cell = grid[i][j][k]; // shortcut
                const auto weight = WeightCalculator::weightIdxPoint({ i, j, k }, p.pos);
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
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            auto& cell = grid[i][j][k]; // shortcut
            cell.oldVelocity = cell.velocity;
        }
    }
    void LagrangeEulerView::printGrid() {
        for (u16 i = 0; i < MAX_I; i++) {
            for (u16 k = 0; k < MAX_K; k++) {
                std::cout << glm::to_string(grid[i][0][k].velocity) << " | ";
            }
            std::cout << "\n";
        }
    }
};

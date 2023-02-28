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
    const auto massEps = 0.05f;
    float WeightCalculator::h = 0.6f;
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
            p.pos = particlesOrigin + 0.6f * randomDir;
            p.pos.x = std::clamp(p.pos.x, 0.0f, MAX_I * WeightCalculator::h);
            p.pos.y = std::clamp(p.pos.y, 0.0f, MAX_J * WeightCalculator::h);
            p.pos.z = std::clamp(p.pos.z, 0.0f, MAX_K * WeightCalculator::h);

            const glm::ivec3 gridPos = p.pos / WeightCalculator::h;
            grid[gridPos.x][gridPos.y][gridPos.z].nParticles++;

            p.velocity = glm::vec3{ 5.0f, -40.0f, 0.0f };

            p.r = rand() % 256;
            p.g = rand() % 256;
            p.b = rand() % 256;
            p.a = (rand() % 256);

            p.r = 255;
            p.g = 255;
            p.b = 255;
            p.a = 255;

            p.size = 0.02f;
            p.mass = 22.0f;
            });
        auto sum = 0;
        auto cnt = 0;
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            if (grid[i][j][k].nParticles > 0) {
                sum += grid[i][j][k].nParticles;
                cnt++;
            }
        }
        std::cout << "Avg PPC: " << (float)sum / cnt << "\n";
    }

    void LagrangeEulerView::rasterizeParticlesToGrid() {
        auto cnt = 0;
        used_cells.clear();
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            const GridIndex idx = { i, j, k };
            grid[i][j][k].mass = std::accumulate(particles.cbegin(), particles.cend(), 0.0, [&idx](const int acc, const auto& p) {
                return acc + p.mass * WeightCalculator::wip(idx, p.pos);
                });
            if (grid[i][j][k].mass > massEps) {
                used_cells.push_back(glm::ivec3{ i, j, k });
                cnt++;
            }
        }

        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            const auto momentum = std::accumulate(particles.cbegin(), particles.cend(), glm::vec3{ 0, 0, 0 }, [=](const auto acc, const auto& p) {
                return acc + p.velocity * p.mass * WeightCalculator::wip({ i, j, k }, p.pos);
                });
            grid[i][j][k].velocity = momentum / grid[i][j][k].mass;
        }
    }

    void LagrangeEulerView::computeParticleVolumesAndDensities() {
        auto sum = 0.0f;
        auto cnt = 0;
        for (auto& p : particles) {
            auto density = 0.0f;
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                density += grid[i][j][k].mass * WeightCalculator::wip({ i, j, k }, p.pos);
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
                float mu = mu0 * glm::exp(xi * (1 - JP));
                float lambda = lambda0 * glm::exp(xi * (1 - JP));

                const auto& [RE, SE] = polarDecomposition(FE);
                //checkPolarDecomposition(RE, SE, FE);

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
            // see mpm-review.pdf, 2.5.4 on how to get rid of this
            if (cell.mass <= massEps)
                continue;
            cell.starVelocity = cell.oldVelocity + cell.force * timeDelta * (1.0f / cell.mass);
        }
    }


    glm::vec3 LagrangeEulerView::bodyCollision(const glm::vec3& pos, const glm::vec3& velocity) {
        std::vector<std::function<float(const glm::vec3&)>> sdfs;

        sdfs.push_back([](const glm::vec3& pos) {
            return pos.y;
            });
        sdfs.push_back([](const glm::vec3& pos) {
            if (abs(pos.x - 3.0f) < 0.3)
                return std::max({ pos.y - pos.x + 2, pos.y + pos.x - 4 });
            else
                return std::numeric_limits<float>::infinity();
            });

        return std::accumulate(std::cbegin(sdfs), std::cend(sdfs), velocity,
            [savedDistance = -std::numeric_limits<float>::infinity(), &pos, &velocity](const auto acc, const auto& sdf) mutable {
                const auto currentDistance = sdf(pos);
                // need to pick sdf with maximum negative distance
                if (currentDistance > 0 || currentDistance < savedDistance) {
                    return acc;
                }
                savedDistance = currentDistance;
                const auto normal = gradient(sdf, pos, 3);
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

    float LagrangeEulerView::Energy(const Eigen::VectorXf& velocities, float timeDelta) {
        auto energy = 0.0f;
        auto velIdx = 0;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            const auto& rCell = grid[i][j][k]; // shortcut
            const auto g0Position = glm::vec3(i, j, k) * WeightCalculator::h;
            const auto velocity = glm::vec3(velocities[velIdx], velocities[velIdx + 1], velocities[velIdx + 2]);
            const auto velDiffNorm2 = glm::dot(velocity - rCell.velocity, velocity - rCell.velocity);
            energy += 0.5 * rCell.mass * velDiffNorm2;
            ++velIdx;
        }
        energy += ElasticPotential(timeDelta);
        return energy;
    }

    float LagrangeEulerView::ElasticPotential(float timeDelta) {
        return std::accumulate(std::cbegin(particles), std::cend(particles), 0.0f, [&](const auto acc, const auto& p) {
            const auto& FP = p.FPlastic;
            auto FE = glm::mat3(1.0);
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                FE += glm::outerProduct(glm::vec3{} *timeDelta/*TODO*/, WeightCalculator::wipGrad({ i, j, k }, p.pos));
            }
            FE *= p.FElastic;
            return acc + p.volume * ElasticPlasticEnergyDensity(FE, FP);
            });
    }

    float LagrangeEulerView::ElasticPlasticEnergyDensity(const glm::mat3& FE, const glm::mat3& FP) {
        auto mu = [=](const auto& fp) {
            return mu0 * glm::exp(xi * 1 - glm::determinant(fp));
        };
        auto lambda = [=](const auto& fp) {
            return lambda0 * glm::exp(xi * 1 - glm::determinant(fp));
        };
        const auto& [RE, SE] = polarDecomposition(FE);
        auto fNorm2 = [](const glm::mat3& m) -> float {
            auto val = 0.0f;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    val += m[i][j] * m[i][j];
            return val;
        };
        const auto JE = glm::determinant(FE);
        return mu(FP) * fNorm2(FE - RE) + lambda(FP) / 2 * (JE - 1) * (JE - 1);
    }

    void LagrangeEulerView::gridBasedBodyCollisions() {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            auto& cell = grid[i][j][k];
            const auto gridPosition = glm::vec3(i, j, k) * WeightCalculator::h;
            cell.starVelocity = bodyCollision(gridPosition, cell.starVelocity);
        }
    }

    void LagrangeEulerView::timeIntegration(float timeDelta, bool implicit) {
        if (!implicit) {
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                grid[i][j][k].velocity = grid[i][j][k].starVelocity;
            }
            return;
        }
        else {
            auto E = [&](const Eigen::VectorXf& vel) {
                return Energy(vel, timeDelta);
            };
            Eigen::VectorXf initialVelocitites(used_cells.size() * 3);
            for (size_t i = 0; i < initialVelocitites.size(); ++i) {
                initialVelocitites[i] = 0.0f;
            }
            const auto velocities = optimize(E, initialVelocitites, 0.01);
            auto velIdx = 0;
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                grid[i][j][k].velocity = glm::vec3(velocities[velIdx], velocities[velIdx + 1], velocities[velIdx + 2]);
                ++velIdx;
            }
        }
        // Conjugate residual method (10-30 iterations)

    }

    void LagrangeEulerView::updateDeformationGradient(float timeDelta) {
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            auto velGradient = glm::mat3{ 0.0 };
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                const auto grad = WeightCalculator::wipGrad({ i, j, k }, p.pos);
                velGradient += glm::outerProduct(grid[i][j][k].velocity, grad);
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
            //std::cout << s(0) << " " << s(1) << " " << s(2) << "\n";
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
            p.pos.x = std::clamp(p.pos.x, 0.0f, MAX_I * WeightCalculator::h);
            p.pos.y = std::clamp(p.pos.y, 0.0f, MAX_J * WeightCalculator::h);
            p.pos.z = std::clamp(p.pos.z, 0.0f, MAX_K * WeightCalculator::h);
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

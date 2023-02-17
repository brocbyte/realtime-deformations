#include "material_point_method.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <ranges>
#include "utils.h"
#include <iostream>
#include <vector>
#include <functional>
#include <glm/gtx/string_cast.hpp>
#include "polar_decomposition_3x3.h"
#include <Eigen/Dense>



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
            p.velocity = glm::vec3{};

            p.life = -1.0f;
            p.cameradistance = -1.0f;
            });
    }

    void LagrangeEulerView::initParticles() {
        std::for_each(std::begin(particles), std::end(particles), [=](auto& p) {
            p.pos = glm::vec3(5.0f, 0.0001f, 0.0001f);

            float spread = 2.5f;
            glm::vec3 maindir = glm::vec3(0.0f, -5.0f, 0.0f);
            auto randFloat = [](auto low, auto high) {
                return low + (rand() % 2000) / 2000.0f * (high - low);
            };
            const auto phi = randFloat(0.0f, 2.0f * 3.1415);
            const auto costheta = randFloat(0.0f, 2.0f) - 1.0f;
            const auto u = randFloat(0.0f, 1.0f);
            const auto theta = acos(costheta);
            const auto R = 1.0f;
            const auto r = R * std::cbrt(u);

            glm::vec3 randomdir = glm::vec3(
                r * sin(theta) * cos(phi),
                r * sin(theta) * sin(phi),
                r * cos(theta)
            );
            p.pos += randomdir;
            p.pos.x = std::clamp(p.pos.x, 0.001f, MAX_I * WeightCalculator::h);
            p.pos.y = std::clamp(p.pos.y, 0.001f, MAX_J * WeightCalculator::h);
            p.pos.z = std::clamp(p.pos.z, 0.001f, MAX_K * WeightCalculator::h);
            p.pos.y = p.pos.z = 0.0f;

            p.velocity = glm::vec3{ 0.0f, 0, 1.57f };
            //p.velocity = randomdir * 0.5f;//maindir + (randomdir * 3.0f);
            //p.velocity.z *= p.velocity.z > 0 ? 1 : -1;
            p.velocity.y = 0;

            p.r = rand() % 256;
            p.g = rand() % 256;
            p.b = rand() % 256;
            p.a = (rand() % 256) / 3;

            p.r = 255;
            p.g = 255;
            p.b = 255;

            p.size = 0.05f;
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
            p.volume = (density > 0 ? p.mass / density : 0) * 1e-6;
        }
    }

    void LagrangeEulerView::computeGridForces() {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            auto& cell = grid[i][j][k];
            cell.force =
                -std::accumulate(particles.begin(), particles.end(), glm::vec3{}, [=](glm::vec3 acc, const auto& p) {
                const glm::mat3& FE = p.FElastic; // shortcut
                const glm::mat3& FP = p.FPlastic; // shortcut
                const float& JE = glm::determinant(FE);
                const float& JP = glm::determinant(FP);
                const float J = glm::determinant(FE * FP);
                const float volume = J * p.volume;
                const float mu = mu0 * glm::exp(xi * (1 - JP));
                const float lambda = lambda0 * glm::exp(xi * (1 - JP));
                glm::mat3 RE{ 1.0 }, SE{};
                float _FE[9], _RE[9], _SE[9];
                for (int q = 0; q < 3; q++) {
                    for (int w = 0; w < 3; w++) {
                        _FE[q * 3 + w] = FE[q][w];
                    }
                }
                polar::polar_decomposition(_RE, _SE, _FE);
                for (int q = 0; q < 3; q++) {
                    for (int w = 0; w < 3; w++) {
                        SE[q][w] = _SE[q * 3 + w];
                        RE[q][w] = _RE[q * 3 + w];
                    }
                }
                const auto check = (RE * SE - FE);
                const glm::mat3 cauchyStress = J > 0 ?
                    (((FE - RE) * glm::transpose(FE)) * 2.0f * mu + (glm::mat3(1.0) * lambda * (JE - 1.0f) * JE)) * (1.0f / J) : glm::mat3(1.0);
                return acc + volume * cauchyStress * WeightCalculator::weightIdxPointGradient({ i, j, k }, p.pos);
                    });
        }
    }

    void LagrangeEulerView::updateVelocitiesOnGrid(float timeDelta) {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            auto& cell = grid[i][j][k];
            cell.starVelocity = cell.mass > 0.001 ? cell.oldVelocity + cell.force * timeDelta * (1.0f / cell.mass) : glm::vec3{};
        }
    }


    glm::vec3 LagrangeEulerView::bodyCollision(const glm::vec3& pos, const glm::vec3& velocity) {
        std::vector<std::function<float(const glm::vec3&)>> phis;

        //phis.push_back([](const glm::vec3& pos) {
        //    const auto center = glm::vec3{ 10, 0, 10 };
        //    const auto radius = 3;
        //    return (pos.x - center.x) * (pos.x - center.x) + (pos.y - center.y) * (pos.y - center.y) + (pos.z - center.z) * (pos.z - center.z) - radius * radius;
        //    });
        //phis.push_back([](const glm::vec3& pos) {
        //    const auto center = glm::vec3{ 8, 3, 8 };
        //    const auto radius = 2;
        //    return (pos.x - center.x) * (pos.x - center.x) + (pos.y - center.y) * (pos.y - center.y) + (pos.z - center.z) * (pos.z - center.z) - radius * radius;
        //    });
        //phis.push_back([](const glm::vec3& pos) {
        //    const auto center = glm::vec3{ 12, 3, 12 };
        //    const auto radius = 2.5f;
        //    return (pos.x - center.x) * (pos.x - center.x) + (pos.y - center.y) * (pos.y - center.y) + (pos.z - center.z) * (pos.z - center.z) - radius * radius;
        //    });
        phis.push_back([](const glm::vec3& pos) {
            return 2 - pos.z;
            });
        phis.push_back([](const glm::vec3& pos) {
            return 6 - pos.x;
            });
        phis.push_back([](const glm::vec3& pos) {
            return pos.x - 2.3;
            });
        phis.push_back([](const glm::vec3& pos) {
            return 1.5f / 5.0f * pos.x - pos.z;
            });
        if (std::all_of(std::cbegin(phis), std::cend(phis), [&pos](const auto& phi) { return phi(pos) > 0; })) {
            return velocity;
        }

        std::vector<std::function<glm::vec3(const glm::vec3&)>> gradients;
        for (const auto& phi : phis) {
            gradients.push_back([&phi](const glm::vec3& pos) {
                const auto eps = 0.001f;
                glm::vec3 gradient;
                glm::vec3 dfs[3] = { glm::vec3{1.0, 0, 0}, glm::vec3{0.0, 1.0, 0.0}, glm::vec3{0.0, 0.0, 1.0} };
                for (int i = 0; i < 3; ++i) {
                    glm::vec3 posMinus = pos - dfs[i] * eps;
                    glm::vec3 posPlus = pos + dfs[i] * eps;
                    gradient[i] = (phi(posPlus) - phi(posMinus)) / (2.0f * eps);
                }
                return gradient;
                });
        }
        auto outVelocity = velocity;
        for (int i = 0; i < phis.size(); i++) {
            if (phis[i](pos) > 0) {
                continue;
            }
            const auto normal = gradients[i](pos);
            const auto objectVelocity = glm::vec3{}; // TODO moving objects...
            const auto relVelocity = outVelocity - objectVelocity;
            const float vn = glm::dot(relVelocity, normal);
            if (vn >= 0) {
                outVelocity = velocity;
                continue;
            }
            const auto vt = relVelocity - normal * vn;
            const auto mu = 0.5f;
            auto vrel = glm::vec3{};
            if (vt.length() > -mu * vn && vt.length() > 0) {
                vrel = vt + vt * (mu * vn / vt.length());
            }
            outVelocity = vrel + objectVelocity;
        }
        return outVelocity;
    }

    void LagrangeEulerView::gridBasedBodyCollisions() {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            auto& cell = grid[i][j][k];
            const auto gridPosition = glm::vec3{ i, j, k } *WeightCalculator::h;
            cell.starVelocity = bodyCollision(gridPosition, cell.velocity);
        }
    }

    void LagrangeEulerView::timeIntegration() {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            grid[i][j][k].velocity = grid[i][j][k].starVelocity;
        }
    }

    static Eigen::MatrixXf glmToEigen(const glm::mat3& mat) {
        Eigen::MatrixXf m{ 3, 3 };
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m(i, j) = mat[i][j];
            }
        }
        return m;
    }

    static glm::mat3 eigenToGlm(const Eigen::MatrixXf mat) {
        glm::mat3 m;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] = mat(i, j);
            }
        }
        return m;
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
            //std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
            Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(m);
            if (svd.info() != Eigen::ComputationInfo::Success) {
                std::cout << "Error!: " << svd.info() << "\n";
                std::cout << glmToEigen(velGradient) << "\n";
                printGrid();
            }
            //std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
            Eigen::VectorXf s = svd.singularValues();
            for (int i = 0; i < 3; i++) {
                s(i) = std::clamp(s(i), (float)(1 - 1.9f * 1e-2), (float)(1 + 5.0f * 1e-3));
            }
            Eigen::MatrixXf _S{ {s(0), 0, 0}, {0, s(1), 0}, {0, 0, s(2)} };
            //std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
            //std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
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

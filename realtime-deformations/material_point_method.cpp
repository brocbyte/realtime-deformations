#include "material_point_method.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include "utils.h"
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "mathy.hpp"
#include <limits>
#include <constants.hpp>
#include <chrono>
#include "cudaCalc.cuh"

namespace MaterialPointMethod {
    ftype WeightCalculator::h = 0.05f;
    void LagrangeEulerView::initializeParticles(const v3t& particlesOrigin, const v3t& velocity) {
        for (int pi = 0; pi < nParticles; ++pi) {
            auto& p = particles[pi];
            p.pos = clampPosition(particlesOrigin + generateRandomInsideUnitBall(0.2));
            p.velocity = velocity;
            p.r = rand() % 256;
            p.g = rand() % 256;
            p.b = rand() % 256;
            p.r = 255;
            p.g = 255;
            p.b = 255;
            p.a = 255;
            p.size = 0.02;
            p.mass = 0.00001;
            p.F = glm::mat3(1.0f);
        }
        logger.log(Logger::LogLevel::INFO, "Particles momentum", particleMomentum());
        logger.log(Logger::LogLevel::INFO, "Average PPC", averagePPC());
    }

    void LagrangeEulerView::precalculateWeights() {
        cudaMemcpy(devParticles, particles.data(), nParticles * sizeof(Particle), cudaMemcpyHostToDevice);
        cudaMemset(grid.devGrid, 0, (long long)MAX_I * MAX_J * MAX_K * sizeof(Cell));

        cuP2G(devParticles, grid.devGrid, nParticles, MAX_I, MAX_J, MAX_K, w.devW);

        cudaMemcpy(w.w.data(), w.devW, (long long)MAX_I * MAX_J * MAX_K * nParticles * sizeof(ftype), cudaMemcpyDeviceToHost);
        cudaMemcpy(grid.grid.data(), grid.devGrid, (long long)MAX_I * MAX_J * MAX_K * sizeof(Cell), cudaMemcpyDeviceToHost);
        if (cudaDeviceSynchronize() != cudaError::cudaSuccess) {
            logger.log(Logger::LogLevel::ERROR, cudaGetErrorString(cudaGetLastError()));
        }
    }

    void LagrangeEulerView::rasterizeParticlesToGrid() {
        precalculateWeights();

        used_cells.clear();
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            if (grid(i, j, k).mass != 0.0f) {
                used_cells.push_back({ i, j, k });
            }
        }

        const auto DpInverse = glm::inverse(glm::mat3(1.0) * (1.0f / 3.0f) * WeightCalculator::h * WeightCalculator::h);
        for (const auto& idx : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
            auto momentum = v3t(0, 0, 0);
            for (size_t p = 0; p < nParticles; ++p) {
                const auto Xi = v3t(idx) * WeightCalculator::h;
                const auto& pRef = particles[p];
                momentum += w(i, j, k, p) * pRef.mass * (pRef.velocity + pRef.B * DpInverse * (Xi - pRef.pos));
            }
            grid(i, j, k).velocity = momentum / grid(i, j, k).mass;
        }
        const auto pMomentum = particleMomentum();
        const auto gMomentum = gridMomentum();
        logger.log(Logger::LogLevel::INFO, "ParticlesMomentum", pMomentum);
        logger.log(Logger::LogLevel::INFO, "GridMomentum", gMomentum);
        if (glm::length(pMomentum - gMomentum) / glm::length(gMomentum) > 1e-2) {
            logger.log(Logger::LogLevel::WARNING, "GridMomentum != ParticlesMomentum after rasterization");
        }
    }

    void LagrangeEulerView::computeParticleVolumesAndDensities() {
        for (size_t p = 0; p < nParticles; ++p) {
            auto density = 0.0f;
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                density += grid(i, j, k).mass * w(i, j, k, p);
            }
            density /= (WeightCalculator::h * WeightCalculator::h * WeightCalculator::h);
            particles[p].volume = density != 0.0f ? (particles[p].mass / density) : 0;
        }
        logger.log(Logger::LogLevel::INFO, "Average cell density", averageCellDensity());
        logger.log(Logger::LogLevel::INFO, "Average particle density", averageParticleDensity());
    }

    LagrangeEulerView::LagrangeEulerView(int max_i, int max_j, int max_k, int particlesNum)
        : MAX_I(max_i), MAX_J(max_j), MAX_K(max_k), grid{ max_i, max_j, max_k }, w{ max_i, max_j, max_k, particlesNum }, nParticles(particlesNum) {
        if (cudaMalloc((void**)&devParticles, nParticles * sizeof(Particle)) != cudaError::cudaSuccess) {
            std::cout << "cudaMalloc error\n";
        }
        particles.resize(nParticles);
        if (cudaMemcpyToSymbol((void*)&devH, &WeightCalculator::h, sizeof(ftype), 0, cudaMemcpyHostToDevice) != cudaError::cudaSuccess) {
            std::cout << "cudaMemcpyToSymbol error\n";
        }
        used_cells.reserve(MAX_I * MAX_J * MAX_K);
    }

    LagrangeEulerView::~LagrangeEulerView() {
        cudaFree(devParticles);
    }

    ftype LagrangeEulerView::Energy(const Eigen::VectorXf& velocities, ftype timeDelta) {
        auto energy = 0.0;
        auto velIdx = 0;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            const auto velocityNew = v3t(velocities[velIdx], velocities[velIdx + 1], velocities[velIdx + 2]);
            const auto velDiffNorm = glm::length(velocityNew - grid(i, j, k).velocity);
            energy += 0.5 * grid(i, j, k).mass * velDiffNorm * velDiffNorm;
            velIdx += 3;
        }
        energy += ElasticPotential(velocities, timeDelta);
        return energy;
    }

    ftype LagrangeEulerView::ElasticPotential(const Eigen::VectorXf& velocities, ftype timeDelta) {
        auto acc = 0.0f;
        for (size_t p = 0; p < nParticles; ++p) {
            const auto& FP = particles[p].FPlastic;
            auto FE = m3t(1.0);
            auto velIdx = 0;
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                const auto velocity = v3t(velocities[velIdx], velocities[velIdx + 1], velocities[velIdx + 2]);
                FE += glm::outerProduct(velocity * timeDelta, WeightCalculator::wipGrad(idx, particles[p].pos));
                velIdx += 3;
            }
            FE = FE * particles[p].FElastic;
            acc += particles[p].volume * ElasticPlasticEnergyDensity(FE, FP);
        }
        return acc;
    }

    ftype LagrangeEulerView::ElasticPlasticEnergyDensity(const m3t& FE, const m3t& FP) {
        auto mu = [=](const auto& fp) {
            return mu0 * glm::exp(xi * 1 - glm::determinant(fp));
        };
        auto lambda = [=](const auto& fp) {
            return lambda0 * glm::exp(xi * 1 - glm::determinant(fp));
        };
        const auto [RE, SE] = polarDecomposition(FE);
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

    void LagrangeEulerView::timeIntegration(ftype timeDelta) {
        auto E = [&](const Eigen::VectorXf& vel) {
            return Energy(vel, timeDelta);
        };
        Eigen::VectorXf initialVelocities(used_cells.size() * 3);
        auto velIdx = 0;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            initialVelocities[velIdx] = grid(i, j, k).velocity[0];
            initialVelocities[velIdx + 1] = grid(i, j, k).velocity[1];
            initialVelocities[velIdx + 2] = grid(i, j, k).velocity[2];
            velIdx += 3;
        }
        Optimizer optimizer{ E };
        optimizer.setLevel(DEFAULT_LOG_LEVEL_OPTIMIZER);
        const auto velocities = optimizer.optimize(E, initialVelocities, used_cells);
        velIdx = 0;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            grid(i, j, k).velocity = glm::vec3(velocities[velIdx], velocities[velIdx + 1], velocities[velIdx + 2]);
            velIdx += 3;
        }
    }

    void LagrangeEulerView::computeExplicitGridForces() {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            grid(i, j, k).force = v3t(0, 0, 0);
            for (const auto& p : particles) {
                const auto poisson = 0.2f;
                const auto E = 1.4e5f;
                const auto mu = E / (2.0f * (1 + poisson));
                const auto lamda = (E * poisson) / ((1 + poisson) * (1 - 2 * poisson));
                const auto [R, S] = polarDecomposition(p.F);
                const auto detR = glm::determinant(R);
                const auto J = glm::determinant(p.F);
                const auto FT = glm::transpose(glm::inverse(p.F));
                const m3t derivative = 2 * mu * (p.F - R) + lamda * (J - 1) * J * FT;
                grid(i, j, k).force -= p.volume * derivative * glm::transpose(p.F) * WeightCalculator::wipGrad({ i, j, k }, p.pos);
            }
        }
    }

    void LagrangeEulerView::gridVelocitiesUpdate(ftype timeDelta) {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            auto& cellRef = grid(i, j, k);
            cellRef.velocity += timeDelta * cellRef.force / cellRef.mass;
        }
    }

    v3t LagrangeEulerView::bodyCollision(const v3t& pos, const v3t& velocity, ftype timeDelta, const std::vector<MeshCollider>& objects) {
        static ftype currentTime{ 0.0f };
        currentTime += timeDelta;
        std::vector<std::pair<std::function<float(const glm::vec3&)>, v3t>> phisWithVels;
        for (const auto& obj : objects) {
            phisWithVels.push_back({ obj.sdf, obj.velocity });
        }

        if (std::all_of(phisWithVels.cbegin(), phisWithVels.cend(), [&pos](const auto& phiWithVel) { return phiWithVel.first(pos) > 0; })) {
            return velocity;
        }

        auto outVelocity = velocity;
        for (const auto& [phi, objectVelocity] : phisWithVels) {
            if (phi(pos) > 0) {
                continue;
            }
            const auto normal = gradient(phi, pos, 3);
            const auto relVelocity = outVelocity - objectVelocity;
            const float vn = glm::dot(relVelocity, normal);
            if (vn >= 0) {
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

    void LagrangeEulerView::gridBasedCollisions(ftype timeDelta, const std::vector<MeshCollider>& objects) {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            const auto pos = v3t(cell) * MaterialPointMethod::WeightCalculator::h;
            grid(i, j, k).velocity = bodyCollision(pos, grid(i, j, k).velocity, timeDelta, objects);

        }
    }

    void LagrangeEulerView::updateDeformationGradient(ftype timeDelta) {
        for (int pi = 0; pi < nParticles; ++pi) {
            auto& p = particles[pi];
            m3t velocityGradient{ 0.0f };
            for (const auto& cell : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
                const auto grad = WeightCalculator::wipGrad(cell, p.pos);
                velocityGradient += glm::outerProduct(grid(i, j, k).velocity, grad);
            }
            p.F = (glm::mat3(1.0f) + timeDelta * velocityGradient) * p.F;
        }
        logger.log(Logger::LogLevel::INFO, "Deformation gradient", averageDeformationGradient());
    }

    void LagrangeEulerView::updateParticleVelocities() {
        for (size_t p = 0; p < nParticles; ++p) {
            auto& pRef = particles[p];
            pRef.velocity = glm::vec3(0.0f, 0.0f, 0.0f);
            pRef.B = glm::mat3(0.0f);
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                pRef.velocity += grid(i, j, k).velocity * w(i, j, k, p);
                const auto Xi = v3t(idx) * WeightCalculator::h;
                pRef.B += w(i, j, k, p) * glm::outerProduct(grid(i, j, k).velocity, (Xi - pRef.pos));
            }
        }
    }

    void LagrangeEulerView::updateParticlePositions(ftype timeDelta) {
        for (int i = 0; i < nParticles; ++i) {
            auto& p = particles[i];
            p.pos += p.velocity * timeDelta;
            p.pos = clampPosition(p.pos);
        }
    }

    v3t LagrangeEulerView::gridMomentum() {
        v3t momentum(0, 0, 0);
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            momentum += grid(i, j, k).velocity * grid(i, j, k).mass;
        }
        return momentum;
    }

    v3t LagrangeEulerView::particleMomentum() {
        auto out = v3t(0, 0, 0);
        for (int i = 0; i < nParticles; ++i) {
            const auto& p = particles[i];
            out += p.velocity * p.mass;
        }
        return out;
    }

    ftype LagrangeEulerView::gridMass() {
        ftype mass = 0.0f;
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };
            mass += grid(i, j, k).mass;
        }
        return mass;
    }

    // when a particle approaches some of the borders its momentum starts fading,
    // cause some of it gets interpolated into inexistent grid nodes (outside the simulation)
    v3t LagrangeEulerView::clampPosition(const v3t& vec) {
        v3t out;
        const auto m = (ftype)(2 * WeightCalculator::h);
        out.x = std::clamp(vec.x, m, (ftype)((MAX_I - 2) * WeightCalculator::h));
        out.y = std::clamp(vec.y, m, (ftype)((MAX_J - 2) * WeightCalculator::h));
        out.z = std::clamp(vec.z, m, (ftype)((MAX_K - 2) * WeightCalculator::h));
        return out;
    }

    ftype LagrangeEulerView::averagePPC() {
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            grid(i, j, k).nParticles = 0;
        }
        for (int i = 0; i < nParticles; ++i) {
            auto& p = particles[i];
            const glm::ivec3 gridPos = p.pos / WeightCalculator::h;
            grid(gridPos.x, gridPos.y, gridPos.z).nParticles++;
        }
        auto sum = 0;
        auto cnt = 0;
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            if (grid(i, j, k).nParticles > 0) {
                sum += grid(i, j, k).nParticles;
                ++cnt;
            }
        }
        return (ftype)sum / cnt;
    }

    ftype LagrangeEulerView::averageParticleDensity() {
        ftype sum = 0.0;
        auto cnt = 0;
        for (int i = 0; i < nParticles; ++i) {
            auto& p = particles[i];
            if (p.volume == 0.0f) {
                continue;
            }
            sum += p.mass / p.volume;
            ++cnt;;
        }
        return sum / cnt;
    }

    ftype LagrangeEulerView::averageCellDensity() {
        ftype cellMassSum = 0.0f;
        auto cnt = 0;
        MAKE_LOOP(i, MAX_I, j, MAX_J, k, MAX_K) {
            if (grid(i, j, k).mass > 0) {
                cellMassSum += grid(i, j, k).mass;
                ++cnt;
            }
        }
        const auto cellVolume = WeightCalculator::h * WeightCalculator::h * WeightCalculator::h;
        return cellMassSum / cnt / cellVolume;
    }

    m3t LagrangeEulerView::averageDeformationGradient() {
        m3t avg(0.0f);
        for (const auto& p : particles) {
            avg += p.F;
        }
        return avg / (ftype)nParticles;
    }
};

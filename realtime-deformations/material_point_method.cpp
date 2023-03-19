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
            p.pos = clampPosition(particlesOrigin + generateRandomInsideUnitBall(0.5));
            p.velocity = velocity;
            p.r = 255;
            p.g = 255;
            p.b = 255;
            p.a = 255;
            p.size = 0.02;
            p.mass = 1.0;
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
        used_cells.clear();
        auto start = std::chrono::high_resolution_clock::now();
        precalculateWeights();
        auto end = std::chrono::high_resolution_clock::now();
        int duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        logger.log(Logger::LogLevel::ERROR, "CUDA kernel call duration", duration);

        start = std::chrono::high_resolution_clock::now();
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
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        logger.log(Logger::LogLevel::ERROR, "The rest of rasterization duration", duration);
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

    void LagrangeEulerView::gridBasedCollisions() {
        for (const auto& cell : used_cells) {
            const auto& [i, j, k] = std::array<int, 3>{ cell.x, cell.y, cell.z };

            const auto pos = v3t(cell) * MaterialPointMethod::WeightCalculator::h;
            auto distance = std::numeric_limits<ftype>::infinity();
            if (abs(pos.x - 1.0f) < 0.33)
                distance = std::max({ pos.y - pos.x - 0, pos.y + pos.x - 2 });
            if (distance < 0) {
                grid(i, j, k).velocity = v3t(0, 0, 0);
            }
        }
    }

    void LagrangeEulerView::updateDeformationGradient(ftype timeDelta) {
        for (int pi = 0; pi < nParticles; ++pi) {
            auto& p = particles[pi];
            auto velGradient = m3t(0.0);
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                const auto grad = WeightCalculator::wipGrad(idx, p.pos);
                velGradient += glm::outerProduct(grid(i, j, k).velocity, grad);
            }
            const auto FEpKryshka = (m3t(1.0) + velGradient * timeDelta) * p.FElastic;
            const auto FPpKryshka = p.FPlastic;

            const auto FPn1 = FEpKryshka * FPpKryshka;
            const auto m = glmToEigen(FEpKryshka);
            Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(m);
            if (svd.info() != Eigen::ComputationInfo::Success) {
                logger.log(Logger::LogLevel::ERROR, "SVD Error", svd.info());
                logger.log(Logger::LogLevel::ERROR, "velocityGradient", velGradient);
                return;
            }
            Eigen::VectorXf s = svd.singularValues();
            logger.log(Logger::LogLevel::INFO, "SVD Singular", glm::vec3(s(0), s(1), s(2)));
            for (int i = 0; i < 3; i++) {
                s(i) = std::clamp(s(i), (float)(1 - 2.5f * 1e-2), (float)(1 + 7.5f * 1e-3));
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
        }
    }

    void LagrangeEulerView::updateParticleVelocities() {
        for (size_t p = 0; p < nParticles; ++p) {
            auto& pRef = particles[p];
            pRef.velocity = glm::vec3(0.0f, 0.0f, 0.0f);
            pRef.B = glm::mat3(0.0f);
            for (const auto& idx : used_cells) {
                const auto& [i, j, k] = std::array<int, 3>{ idx.x, idx.y, idx.z };
                const auto weight = w(i, j, k, p);
                pRef.velocity += grid(i, j, k).velocity * weight;
                const auto Xi = v3t(idx) * WeightCalculator::h;
                pRef.B += weight * glm::outerProduct(grid(i, j, k).velocity, (Xi - pRef.pos));
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
        out.x = std::clamp(vec.x, (ftype)(WeightCalculator::h * 1.2), (ftype)((MAX_I - 1) * WeightCalculator::h));
        out.y = std::clamp(vec.y, (ftype)(WeightCalculator::h * 1.2), (ftype)((MAX_J - 1) * WeightCalculator::h));
        out.z = std::clamp(vec.z, (ftype)(WeightCalculator::h * 1.2), (ftype)((MAX_K - 1) * WeightCalculator::h));
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
};

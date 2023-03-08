#pragma once
#include <glm/glm.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
#define FIXED_FLOAT(x) std::fixed<<std::setprecision(5)<<(x)

// common optimization parameters
static const auto MAX_ITERATIONS = 300;
static const auto terminationCriterion = 1e-2f;

// line search parameters
static const auto c1 = 1e-4f;
static const auto c2 = 0.9f;

inline Eigen::VectorXf gradient(const std::function<float(const Eigen::VectorXf&)>& f, const Eigen::VectorXf& x, size_t dimensions, float delta = 0.001) {
    Eigen::VectorXf gradient(dimensions);
    Eigen::VectorXf dfs(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        dfs[i] = 0.0f;
    }
    for (int i = 0; i < dimensions; ++i) {
        if (i > 0) {
            dfs[i - 1] = 0.0f;
        }
        dfs[i] = 1.0f;
        const auto f1 = f(x - dfs * delta);
        const auto f2 = f(x + dfs * delta);
        gradient[i] = (f2 - f1) / (2.0f * delta);
    }
    return gradient;
}

inline glm::vec3 gradient(const std::function<float(const glm::vec3&)>& f, const glm::vec3& x, size_t dimensions, float delta = 0.001) {
    glm::vec3 gradient(dimensions);
    glm::vec3 dfs(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        dfs[i] = 0.0f;
    }
    for (int i = 0; i < dimensions; ++i) {
        if (i > 0) {
            dfs[i - 1] = 0.0f;
        }
        dfs[i] = 1.0f;
        const auto f1 = f(x - dfs * delta);
        const auto f2 = f(x + dfs * delta);
        gradient[i] = (f1 - f2) / (2.0f * delta);
    }
    return gradient;
}

inline Eigen::SparseMatrix<float> hessian(const std::function<float(const Eigen::VectorXf&)>& f, const Eigen::VectorXf& x, float delta = 0.001) {
    const int dimensions = x.size();
    Eigen::SparseMatrix<float> output(dimensions, dimensions);
    Eigen::VectorXf ei(dimensions), ej(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        ei[i] = ej[i] = 0.0f;
    }
    for (int i = 0; i < dimensions; ++i) {
        if (i > 0) {
            ei[i - 1] = 0.0f;
        }
        for (int j = 0; j < dimensions; ++j) {
            if (j > 0) {
                ej[j - 1] = 0.0f;
            }
            ei[i] = 1;
            ej[j] = 1;

            const auto f1 = f(x + delta * ei + delta * ej);
            const auto f2 = f(x + delta * ei - delta * ej);
            const auto f3 = f(x - delta * ei + delta * ej);
            const auto f4 = f(x - delta * ei - delta * ej);
            if ((f1 - f2 - f3 + f4) != 0.0f) {
                output.insert(i, j) = (f1 - f2 - f3 + f4) / (4 * delta * delta);
            }
        }
    }
    return output;
}

Eigen::VectorXf optimize(const std::function<float(const Eigen::VectorXf&)>& f, const Eigen::VectorXf& initialGuess) {
    auto stepsCnt = 0;
    Eigen::VectorXf guess = initialGuess;
    //std::cout << "##################\n";
    //std::cout << "Initial: " << f(guess) << "\n";
    auto newtonSteps = 0;
    auto gradSteps = 0;
    auto logResult = [&stepsCnt, &newtonSteps, &gradSteps](float value) {
        //std::cout << "solved in " << stepsCnt << "\n";
        //std::cout << "newton steps: " << newtonSteps << "\n";
        //std::cout << "gradient steps: " << gradSteps << "\n";
        //std::cout << "value: " << value << "\n";
        //std::cout << "##################\n";
    };
    while (stepsCnt++ < MAX_ITERATIONS) {
        Eigen::VectorXf grad = gradient(f, guess, guess.size());
        Eigen::VectorXf dx(grad.size());
        if (grad.norm() < terminationCriterion) {
            logResult(f(guess));
            return guess;
        }
        Eigen::SparseMatrix<float> H = hessian(f, guess);
        Eigen::MINRES<Eigen::SparseMatrix<float>, 1, Eigen::DiagonalPreconditioner<float>> mr;
        mr.setTolerance(std::min(0.5f, std::sqrt(std::max(grad.norm(), terminationCriterion))));
        mr.compute(H);
        dx = mr.solve(-grad);
        auto suitableDownhill = [](auto dx, auto gE) -> bool {
            return dx.dot(gE) < -(1e-2) * dx.norm() * gE.norm();
        };
        if (suitableDownhill(dx, grad)) {
            dx = dx;
            ++newtonSteps;
        }
        else if (suitableDownhill(-dx, grad)) {
            dx = -dx;
            ++newtonSteps;
        }
        else {
            dx = -grad;
            ++gradSteps;
        }


        const auto l = 1e3f;
        const auto len = dx.norm();
        if (len > l) {
            dx = dx / (len / l);
        }
        auto aIsSuitable = [=](auto a) {
            const Eigen::VectorXf newGuess = guess + a * dx;
            const auto i1 = f(newGuess) <= f(guess) + c1 * a * dx.dot(grad);
            const auto i2 = abs(dx.dot(gradient(f, newGuess, newGuess.size()))) <= c2 * abs(dx.dot(grad));
            return i1 && i2;
        };
        auto lineSearch = [&]() {
            if (aIsSuitable(1.0f)) {
                return 1.0f;
            }
            auto a = 4096.0f;
            auto cnt = 0;
            while (cnt++ < 50) {
                if (aIsSuitable(a))
                    return a;
                a *= 0.5;
            }
            return a;
        };
        const auto alpha = lineSearch();
        guess += alpha * dx;
        if (f(guess - alpha * dx) - f(guess) < terminationCriterion) {
            break;
        }
    }
    logResult(f(guess));
    return guess;
}

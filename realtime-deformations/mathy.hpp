#pragma once
#include <glm/glm.hpp>
#include <functional>

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

inline Eigen::MatrixXf hessian(const std::function<float(const Eigen::VectorXf&)>& f, const Eigen::VectorXf& x, float delta = 0.001) {
    const int dimensions = x.size();
    Eigen::MatrixXf output(dimensions, dimensions);
    Eigen::VectorXf ei(dimensions), ej(dimensions);
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
            auto f1 = f(x + delta * ei + delta * ej);
            auto f2 = f(x + delta * ei - delta * ej);
            auto f3 = f(x - delta * ei + delta * ej);
            auto f4 = f(x - delta * ei - delta * ej);
            output(i, j) = (f1 - f2 - f3 + f4) / (4 * delta * delta);
        }
    }
    return output;
}

Eigen::VectorXf optimize(const std::function<float(const Eigen::VectorXf&)>& f, const Eigen::VectorXf& initialGuess, float terminationCriterion) {
    auto stepsCnt = 0;
    Eigen::VectorXf guess = initialGuess;
    std::cout << "##################\n";
    std::cout << "Initial: " << f(guess) << "\n";
    while (stepsCnt++ < 30) {
        const Eigen::VectorXf grad = gradient(f, guess, guess.size());
        const Eigen::MatrixXf H = hessian(f, guess);
        if (grad.norm() < terminationCriterion) {
            return guess;
        }
        Eigen::VectorXf dx = -H.inverse() * grad;
        auto suitableDownhill = [](auto dx, auto gE) -> bool {
            return dx.dot(gE) < -(1e-2) * dx.norm() * gE.norm();
        };
        if (suitableDownhill(dx, grad)) {
            dx = dx;
        }
        else if (suitableDownhill(-dx, grad)) {
            dx = -dx;
        }
        else {
            dx = -grad;
        }


        const auto l = 1e3f;
        const auto len = dx.norm();
        if (len > l) {
            dx = dx / (len / l);
        }
        auto lineSearch = [=]() {
            auto a = 1.0f;
            auto c1 = 1e-4f;
            auto c2 = 0.9f;
            auto cnt = 0;
            while (cnt++ < 100) {
                const Eigen::VectorXf newGuess = guess + a * dx;
                const auto i1 = f(newGuess) <= f(guess) + c1 * a * dx.dot(grad);
                const auto i2 = abs(dx.dot(gradient(f, newGuess, newGuess.size()))) <= c2 * abs(dx.dot(grad));
                if (i1 && i2)
                    return a;
                a *= 0.7;
            }
            return a;
        };
        const auto alpha = lineSearch();
        guess += alpha * dx;
        std::cout << "Current: " << f(guess) << "\n";
    }
    std::cout << "##################\n";
    return guess;
}

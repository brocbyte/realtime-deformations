#include "mathy.hpp"

Eigen::VectorXf Optimizer::optimize(const std::function<float(const Eigen::VectorXf&)>& f, const Eigen::VectorXf& initialGuess, const std::vector<glm::ivec3>& used_cells) {
    Eigen::VectorXf guess = initialGuess;
    Objective obj(this);
    optimizer.minimize(obj, guess);
    return guess;
}

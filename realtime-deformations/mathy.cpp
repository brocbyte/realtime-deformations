#include "mathy.hpp"

Eigen::VectorXf Optimizer::optimize(const Eigen::VectorXf& initialGuess) {
    Eigen::VectorXf guess = initialGuess;
    Objective obj(this);
    optimizer.show_denom_warning = true;
    optimizer.minimize(obj, guess);
    return guess;
}

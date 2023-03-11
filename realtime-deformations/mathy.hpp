#pragma once
#include <glm/glm.hpp>
#include <functional>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
#include <logger.hpp>
#include "material_point_method.hpp"
#include "mcloptlib/include/MCL/LBFGS.hpp"

class Optimizer : public Loggable {
public:
    mcl::optlib::LBFGS<ftype, Eigen::Dynamic> optimizer;
    const std::function<float(const Eigen::VectorXf&)> _f;
    Optimizer(const std::function<float(const Eigen::VectorXf&)>& f) : _f(f) {
        optimizer.m_settings.ls_method = mcl::optlib::LSMethod::Backtracking;
    }

    Eigen::VectorXf optimize(const std::function<float(const Eigen::VectorXf&)>& f, const Eigen::VectorXf& initialGuess, const std::vector<glm::ivec3>& used_cells);
};


class Objective : public mcl::optlib::Problem<ftype, Eigen::Dynamic>
{
public:
    Objective(Optimizer* solver_) : solver(solver_) {}
    Optimizer* solver;

    ftype value(const Eigen::VectorXf& v) {
        Eigen::VectorXf grad;
        return gradient(v, grad);
    }

    bool converged(const Eigen::VectorXf& x0, const Eigen::VectorXf& x1, const Eigen::VectorXf& grad) {
        if (grad.norm() < 1e-2) { return true; }
        if ((x0 - x1).norm() < 1e-2) { return true; }
        return false;
    }

    ftype gradient(const Eigen::VectorXf& v, Eigen::VectorXf& grad) {
        return solver->_f(v);
    }

};

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
        gradient[i] = (f2 - f1) / (2.0f * delta);
    }
    return gradient;
}

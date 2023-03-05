#pragma once
#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <iostream>
#include "polar_decomposition_3x3.h"

#define MAKE_LOOP(idx1, mIdx1, idx2, mIdx2, idx3, mIdx3) \
for (int idx1 = 0; idx1 < mIdx1; idx1++) \
    for (int idx2 = 0; idx2 < mIdx2; idx2++) \
        for (int idx3 = 0; idx3 < mIdx3; idx3++)


inline Eigen::MatrixXf glmToEigen(const glm::dmat3& mat) {
    Eigen::MatrixXf m{ 3, 3 };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m(i, j) = mat[i][j];
        }
    }
    return m;
}

inline m3t eigenToGlm(const Eigen::MatrixXf mat) {
    m3t m;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m[i][j] = mat(i, j);
        }
    }
    return m;
}

inline std::array<ftype, 9> glmToPolar(const m3t& mat) {
    std::array<ftype, 9> m{};
    for (int q = 0; q < 3; q++) {
        for (int w = 0; w < 3; w++) {
            m[q * 3 + w] = mat[q][w];
        }
    }
    return m;
}

inline m3t polarToGlm(const std::array<ftype, 9>& mat) {
    m3t m{};
    for (int q = 0; q < 3; q++) {
        for (int w = 0; w < 3; w++) {
            m[q][w] = mat[q * 3 + w];
        }
    }
    return m;
}

inline std::pair<m3t, m3t> polarDecomposition(const m3t& _m, bool usingSVD = false) {
    if (usingSVD) {
        const auto m = glmToEigen(_m);
        Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(m);
        if (svd.info() != Eigen::ComputationInfo::Success) {
            std::cout << "Error!: " << svd.info() << "\n";
            return {};
        }
        Eigen::VectorXf s = svd.singularValues();
        Eigen::MatrixXf _S{ {s(0), 0, 0}, {0, s(1), 0}, {0, 0, s(2)} };
        const auto U = eigenToGlm(svd.matrixU());
        const auto V = eigenToGlm(svd.matrixV());
        const auto S = eigenToGlm(_S);
        return { U * glm::transpose(V), V * S * glm::transpose(V) };
    }
    else {
        std::array<float, 9> _RE, _SE;
        polar::polar_decomposition(_RE.data(), _SE.data(), glmToPolar(_m).data());
        return { polarToGlm(_RE), polarToGlm(_SE) };
    }
}

inline void checkPolarDecomposition(const glm::mat3& RE, const glm::mat3& SE, const glm::mat3& FE) {
    const auto check = RE * SE - FE;
    if (glm::length(check[0]) + glm::length(check[1] + glm::length(check[2])) > 1e-5) {
        std::cout << "Polar decomposition error!\n";
    }
}


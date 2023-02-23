#pragma once
#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <iostream>
#include "polar_decomposition_3x3.h"

#define MAKE_LOOP(idx1, mIdx1, idx2, mIdx2, idx3, mIdx3) \
for (int idx1 = 0; idx1 < mIdx1; idx1++) \
    for (int idx2 = 0; idx2 < mIdx2; idx2++) \
        for (int idx3 = 0; idx3 < mIdx3; idx3++)


inline Eigen::MatrixXf glmToEigen(const glm::mat3& mat) {
    Eigen::MatrixXf m{ 3, 3 };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m(i, j) = mat[i][j];
        }
    }
    return m;
}

inline glm::mat3 eigenToGlm(const Eigen::MatrixXf mat) {
    glm::mat3 m;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m[i][j] = mat(i, j);
        }
    }
    return m;
}

inline std::array<float, 9> glmToPolar(const glm::mat3& mat) {
    std::array<float, 9> m{};
    for (int q = 0; q < 3; q++) {
        for (int w = 0; w < 3; w++) {
            m[q * 3 + w] = mat[q][w];
        }
    }
    return m;
}

inline glm::mat3 polarToGlm(const std::array<float, 9>& mat) {
    glm::mat3 m{};
    for (int q = 0; q < 3; q++) {
        for (int w = 0; w < 3; w++) {
            m[q][w] = mat[q * 3 + w];
        }
    }
    return m;
}

inline std::pair<glm::mat3, glm::mat3> polarDecomposition(const glm::mat3& m) {
    std::array<float, 9> _RE, _SE;
    polar::polar_decomposition(_RE.data(), _SE.data(), glmToPolar(m).data());
    return { polarToGlm(_RE), polarToGlm(_SE) };
}

inline void checkPolarDecomposition(const glm::mat3& RE, const glm::mat3& SE, const glm::mat3& FE) {
    const auto check = RE * SE - FE;
    if (glm::length(check[0]) + glm::length(check[1] + glm::length(check[2])) > 1e-5) {
        std::cout << "Polar decomposition error!\n";
    }
}


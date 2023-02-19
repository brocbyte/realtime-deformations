#pragma once
#include <glm/glm.hpp>
#include <functional>

std::function<glm::vec3(const glm::vec3&)> gradient(const std::function<float(const glm::vec3&)>& f) {
    return [f](const glm::vec3& pos) {
        const auto eps = 0.001f;
        glm::vec3 gradient;
        glm::vec3 dfs[3] = { glm::vec3{1.0, 0, 0}, glm::vec3{0.0, 1.0, 0.0}, glm::vec3{0.0, 0.0, 1.0} };
        for (int i = 0; i < 3; ++i) {
            glm::vec3 posMinus = pos - dfs[i] * eps;
            glm::vec3 posPlus = pos + dfs[i] * eps;
            gradient[i] = (f(posPlus) - f(posMinus)) / (2.0f * eps);
        }
        return gradient;
    };
}

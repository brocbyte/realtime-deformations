#pragma once
#include <glm/glm.hpp>
class Camera {
public:
    glm::vec3 position{5, 0, 0};
    glm::vec3 direction{};
    glm::vec3 up{};
    glm::vec3 right{};
    glm::mat4 view;
    glm::mat4 mvp;
    glm::mat4 projection;
    float hAngle = 3.1415f;
    float vAngle = -3.1415f / 4 + 0.5;
    void updateMatrix();
    void updateDirections();
private:
    float fov = 45.0f;
    glm::mat4 model = glm::mat4(1.0f);
};

#include "camera.hpp"
#include <glm/gtc/matrix_transform.hpp>

void Camera::updateMatrix() {
    view = glm::lookAt(
        position,
        position + direction,
        up
    );
    projection = glm::perspective(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
    mvp = projection * view * model;
}

void Camera::updateDirections() {
    direction = glm::vec3(
        cos(vAngle) * sin(hAngle),
        sin(vAngle),
        cos(vAngle) * cos(hAngle)
    );
    right = glm::vec3(sin(hAngle - 3.14f / 2.0f), 0, cos(hAngle - 3.14f / 2.0f));
    up = glm::cross(right, direction);
}

#include "user_controls.hpp"

void UserControls::update(Camera& iCamera) {
    const auto currentTime = glfwGetTime();
    const auto deltaTime = float(currentTime - lastTime);
    lastTime = currentTime;

    double xpos, ypos;
    glfwGetCursorPos(_window, &xpos, &ypos);
    iCamera.hAngle += mouseSpeed * float(lastX - xpos);
    iCamera.vAngle += mouseSpeed * float(lastY - ypos);
    lastX = xpos;
    lastY = ypos;

    iCamera.updateDirections();
    if (glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS) {
        iCamera.position += iCamera.direction * deltaTime * speed;
    }
    if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS) {
        iCamera.position -= iCamera.direction * deltaTime * speed;
    }
    if (glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS) {
        iCamera.position += iCamera.right * deltaTime * speed;
    }
    if (glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS) {
        iCamera.position -= iCamera.right * deltaTime * speed;
    }
    if (glfwGetKey(_window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        iCamera.position.y += deltaTime * speed;
    }
    if (glfwGetKey(_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        iCamera.position.y -= deltaTime * speed;
    }
    iCamera.updateMatrix();
}
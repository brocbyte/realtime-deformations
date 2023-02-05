#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "camera.hpp"
#include <iostream>
#include "constants.hpp"
class UserControls {
public:
    UserControls(GLFWwindow* window) : _window(window) {
        if (!window) {
            std::cerr << "[x] Window must not be null!\n";
        }
    };
    void update(Camera& iCamera);
private:
    GLFWwindow* _window;
    const float speed = 3.0f;
    const float mouseSpeed = 0.001f;

    double lastX = GLFW_WINDOW_WIDTH / 2.0;
    double lastY = GLFW_WINDOW_HEIGHT / 2.0;
    double lastTime = 0.0;
};

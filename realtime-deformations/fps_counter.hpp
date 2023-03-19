#pragma once
#include <iostream>
class FPSCounter {
private:
    double lastTime = glfwGetTime();
public:
    void log() {
        double currentTime = glfwGetTime();
        std::cout << (currentTime - lastTime) * 1000.0f << " ms / frame\n";
        lastTime = currentTime;
    }
};

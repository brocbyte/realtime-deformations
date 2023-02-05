#pragma once
#include <iostream>
class FPSCounter {
private:
    double lastTime = glfwGetTime();
    int nbFrames = 0;
public:
    void log() {
        double currentTime = glfwGetTime();
        nbFrames++;
        if (currentTime - lastTime >= 1.0) {
            std::cout << 1000.0 / double(nbFrames) << " ms / frame\n";
            nbFrames = 0;
            lastTime += 1.0;
        }
    }
};

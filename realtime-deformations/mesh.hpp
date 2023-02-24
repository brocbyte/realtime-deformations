#pragma once
#include <glm/glm.hpp>
#include <GL/glew.h>

#include <GLFW/glfw3.h>
#include <vector>

class Mesh {
public:
    std::vector<glm::vec3> vertices;
    glm::mat4 matrixWorld{ 1.0 };
};

class BBox {
public:
    static void draw_bbox(GLuint ProgramID, const glm::mat4& VP, const Mesh& mesh);
    static void draw_box(GLuint ProgramID, const glm::mat4& VP, const glm::mat4& transformation);
};


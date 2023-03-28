#pragma once
#include <glm/glm.hpp>
#include <GL/glew.h>

#include <GLFW/glfw3.h>
#include <vector>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

class Object3D {
public:
    glm::mat4 matrixWorld{ 1.0 };
    glm::vec3 scale;
    glm::quat rotation;
    glm::vec3 translation;
    glm::vec3 skew;
    glm::vec4 perspective;
    virtual void draw() = 0;
    void applyMatrix4(const glm::mat4& m) {
        matrixWorld = m * matrixWorld;
        glm::decompose(matrixWorld, scale, rotation, translation, skew, perspective);
    }
};

class Mesh : public Object3D {
public:
    Mesh(GLuint programID, glm::mat4& VP, const std::vector<GLfloat>& vertices, const std::vector<GLfloat>& colors) : _programID(programID), _VP(VP) {
        glGenBuffers(1, &_vertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size() , vertices.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &_colorBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, _colorBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(colors[0]) * colors.size(), colors.data(), GL_STATIC_DRAW);
        _vertices = vertices;
    }
    std::vector<GLfloat> _vertices;
    void draw();
private:
    const GLuint _programID;
    GLuint _vertexBuffer;
    GLuint _colorBuffer;
    const glm::mat4& _VP;
};

class BBox {
public:
    static void draw_bbox(GLuint ProgramID, const glm::mat4& VP, const Mesh& mesh);
    static void draw_box(GLuint ProgramID, glm::mat4& VP, const glm::mat4& transformation);
};

namespace MeshPresets {
    // Cube 1x1x1, centered on origin
    namespace Box {
        extern std::vector<GLfloat> vertices;
        extern std::vector<GLfloat> colors;
    }
};


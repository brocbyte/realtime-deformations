#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <GL/glew.h>

#include <GLFW/glfw3.h>
GLFWwindow* window;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/string_cast.hpp>

#include "common/shader.hpp"
#include "common/objloader.hpp"
#include "common/vboindexer.hpp"
#include "common/texture.hpp"
#include "common/text2D.hpp"

#include "constants.hpp"
#include "user_controls.hpp"
#include "camera.hpp"
#include "fps_counter.hpp"
#include "material_point_method.hpp"
#include "mesh.hpp"

int initializeLibs();

void drawParticles(MaterialPointMethod::LagrangeEulerView& MPM, GLfloat* g_particule_position_size_data, GLubyte* g_particule_color_data, const GLuint& particles_position_buffer, const GLuint& particles_color_buffer, const GLuint& programID, const GLuint& Texture, const GLuint& TextureID, const GLuint& CameraRight_worldspace_ID, glm::mat4& ViewMatrix, const GLuint& CameraUp_worldspace_ID, const GLuint& ViewProjMatrixID, glm::mat4& ViewProjectionMatrix, const GLuint& billboard_vertex_buffer);

int main(void) {
    if (initializeLibs() == -1) {
        return -1;
    }

    const auto numParticles = 50;
    const glm::vec3 particlesOrigin(3.0f, 6.0f, 3.0f);
    MaterialPointMethod::LagrangeEulerView MPM{ 10, 10, 10, numParticles };
    MPM.setLevel(Logger::LogLevel::WARNING);
    MPM.initializeParticles(particlesOrigin, { 0.0f, -20.0f, 0.0f });
    MPM.rasterizeParticlesToGrid();
    MPM.computeParticleVolumesAndDensities();

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    GLuint particlesProgramID = LoadShaders((SHADERS_PATH + "Particle.vertexshader").c_str(), (SHADERS_PATH + "Particle.fragmentshader").c_str());
    GLuint objectsProgramID = LoadShaders((SHADERS_PATH + "Object.vertexshader").c_str(), (SHADERS_PATH + "Object.fragmentshader").c_str());

    // Vertex shader
    GLuint CameraRight_worldspace_ID = glGetUniformLocation(particlesProgramID, "CameraRight_worldspace");
    GLuint CameraUp_worldspace_ID = glGetUniformLocation(particlesProgramID, "CameraUp_worldspace");
    GLuint ViewProjMatrixID = glGetUniformLocation(particlesProgramID, "VP");

    // Fragment shader
    GLuint TextureID = glGetUniformLocation(particlesProgramID, "myTextureSampler");

    static GLfloat* g_particule_position_size_data = new GLfloat[numParticles * 4];
    static GLubyte* g_particule_color_data = new GLubyte[numParticles * 4];

    GLuint Texture = loadDDS((RESOURCES_PATH + "particle.DDS").c_str());

    // The VBO containing the 4 vertices of the particles.
    // Thanks to instancing, they will be shared by all particles.
    static const GLfloat g_vertex_buffer_data[] = {
         -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         -0.5f, 0.5f, 0.0f,
         0.5f, 0.5f, 0.0f,
    };
    GLuint billboard_vertex_buffer;
    glGenBuffers(1, &billboard_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    // The VBO containing the positions and sizes of the particles
    GLuint particles_position_buffer;
    glGenBuffers(1, &particles_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
    // Initialize with empty (NULL) buffer : it will be updated later, each frame.
    glBufferData(GL_ARRAY_BUFFER, numParticles * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);

    // The VBO containing the colors of the particles
    GLuint particles_color_buffer;
    glGenBuffers(1, &particles_color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
    // Initialize with empty (NULL) buffer : it will be updated later, each frame.
    glBufferData(GL_ARRAY_BUFFER, numParticles * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);


    FPSCounter fpsCounter;
    Camera camera;
    camera.position = particlesOrigin + glm::vec3{ 0, 0.3f, 4 };
    UserControls userControls{ window };

    double lastTime = glfwGetTime();
    userControls.update(camera);
    Mesh bboxMesh{ {{0.0, 0.0, 0.0}, v3t(MPM.MAX_I, MPM.MAX_J, MPM.MAX_K) * MaterialPointMethod::WeightCalculator::h} };
    srand(time(NULL));
    do
    {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        double currentTime = glfwGetTime();
        double delta = currentTime - lastTime;
        delta = 1e-3;
        lastTime = currentTime;

        userControls.update(camera);


        glm::mat4 ProjectionMatrix = camera.projection;
        glm::mat4 ViewMatrix = camera.view;
        glm::vec3 CameraPosition(camera.position);
        glm::mat4 ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;

        BBox::draw_bbox(objectsProgramID, ViewProjectionMatrix, bboxMesh);
        glm::vec3 myRotationAxis(0.0, 0.0, 1.0);
        const auto rotation = glm::rotate(glm::mat4(), glm::radians(45.0f), myRotationAxis);
        const auto translation = glm::translate(glm::mat4(), glm::vec3(3, 1.7, 3));
        const auto scaling = glm::scale(glm::mat4(), glm::vec3(0.21f, 0.21f, 0.6f));
        BBox::draw_box(objectsProgramID, ViewProjectionMatrix, translation * rotation * scaling);

        MPM.rasterizeParticlesToGrid();
        MPM.timeIntegration(delta);
        MPM.updateDeformationGradient(delta);
        MPM.updateParticleVelocities();
        MPM.updateParticlePositions(delta);

        drawParticles(MPM, g_particule_position_size_data, g_particule_color_data, particles_position_buffer, particles_color_buffer, particlesProgramID, Texture, TextureID, CameraRight_worldspace_ID, ViewMatrix, CameraUp_worldspace_ID, ViewProjMatrixID, ViewProjectionMatrix, billboard_vertex_buffer);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
        fpsCounter.log();

    } // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0);

    delete[] g_particule_position_size_data;
    delete[] g_particule_color_data;

    // Cleanup VBO and shader
    glDeleteBuffers(1, &particles_color_buffer);
    glDeleteBuffers(1, &particles_position_buffer);
    glDeleteBuffers(1, &billboard_vertex_buffer);
    glDeleteProgram(particlesProgramID);
    glDeleteTextures(1, &Texture);
    glDeleteVertexArrays(1, &VertexArrayID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}

void drawParticles(MaterialPointMethod::LagrangeEulerView& MPM, GLfloat* g_particule_position_size_data, GLubyte* g_particule_color_data, const GLuint& particles_position_buffer, const GLuint& particles_color_buffer, const GLuint& programID, const GLuint& Texture, const GLuint& TextureID, const GLuint& CameraRight_worldspace_ID, glm::mat4& ViewMatrix, const GLuint& CameraUp_worldspace_ID, const GLuint& ViewProjMatrixID, glm::mat4& ViewProjectionMatrix, const GLuint& billboard_vertex_buffer)
{
    const auto MaxParticles = MPM.getParticles().size();
    // Fill the GPU buffer
    for (int i = 0; i < MaxParticles; i++) {
        const auto p = MPM.getParticles()[i];
        g_particule_position_size_data[4 * i + 0] = p.pos.x;
        g_particule_position_size_data[4 * i + 1] = p.pos.y;
        g_particule_position_size_data[4 * i + 2] = p.pos.z;

        g_particule_position_size_data[4 * i + 3] = p.size;

        g_particule_color_data[4 * i + 0] = p.r;
        g_particule_color_data[4 * i + 1] = p.g;
        g_particule_color_data[4 * i + 2] = p.b;
        g_particule_color_data[4 * i + 3] = p.a;
    }


    // Update the buffers that OpenGL uses for rendering.
    // There are much more sophisticated means to stream data from the CPU to the GPU, 
    // but this is outside the scope of this tutorial.
    // http://www.opengl.org/wiki/Buffer_Object_Streaming


    glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
    glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
    glBufferSubData(GL_ARRAY_BUFFER, 0, MaxParticles * sizeof(GLfloat) * 4, g_particule_position_size_data);

    glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
    glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
    glBufferSubData(GL_ARRAY_BUFFER, 0, MaxParticles * sizeof(GLubyte) * 4, g_particule_color_data);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Use our shader
    glUseProgram(programID);

    // Bind our texture in Texture Unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, Texture);
    // Set our "myTextureSampler" sampler to use Texture Unit 0
    glUniform1i(TextureID, 0);

    // Same as the billboards tutorial
    glUniform3f(CameraRight_worldspace_ID, ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
    glUniform3f(CameraUp_worldspace_ID, ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);

    glUniformMatrix4fv(ViewProjMatrixID, 1, GL_FALSE, &ViewProjectionMatrix[0][0]);

    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
    glVertexAttribPointer(
        0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );

    // 2nd attribute buffer : positions of particles' centers
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
    glVertexAttribPointer(
        1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
        4,                                // size : x + y + z + size => 4
        GL_FLOAT,                         // type
        GL_FALSE,                         // normalized?
        0,                                // stride
        (void*)0                          // array buffer offset
    );

    // 3rd attribute buffer : particles' colors
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
    glVertexAttribPointer(
        2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
        4,                                // size : r + g + b + a => 4
        GL_UNSIGNED_BYTE,                 // type
        GL_TRUE,                          // normalized?    *** YES, this means that the unsigned char[4] will be accessible with a vec4 (floats) in the shader ***
        0,                                // stride
        (void*)0                          // array buffer offset
    );

    // These functions are specific to glDrawArrays*Instanced*.
    // The first parameter is the attribute buffer we're talking about.
    // The second parameter is the "rate at which generic vertex attributes advance when rendering multiple instances"
    // http://www.opengl.org/sdk/docs/man/xhtml/glVertexAttribDivisor.xml
    glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
    glVertexAttribDivisor(1, 1); // positions : one per quad (its center)                 -> 1
    glVertexAttribDivisor(2, 1); // color : one per quad                                  -> 1

    // Draw the particules !
    // This draws many times a small triangle_strip (which looks like a quad).
    // This is equivalent to :
    // for(i in ParticlesCount) : glDrawArrays(GL_TRIANGLE_STRIP, 0, 4), 
    // but faster.
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, MaxParticles);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    glDisable(GL_BLEND);
}

int initializeLibs() {
    // Initialise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(GLFW_WINDOW_WIDTH, GLFW_WINDOW_HEIGHT, "Realtime Deformations", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible.\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited movement
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, GLFW_WINDOW_WIDTH / 2, GLFW_WINDOW_HEIGHT / 2);

    // Light blue background
    glClearColor(164.0f / 255, 219.0f / 255, 232.0f / 255, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
}


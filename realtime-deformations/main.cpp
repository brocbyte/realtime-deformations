#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <GL/glew.h>

#include <GLFW/glfw3.h>
GLFWwindow* window;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "common/shader.hpp"
#include "common/objloader.hpp"
#include "common/vboindexer.hpp"
#include "common/texture.hpp"
#include "common/text2D.hpp"

#include "constants.hpp"
#include "user_controls.hpp"
#include "camera.hpp"
#include "fps_counter.hpp"

int initLibs();

int main(void) {
	if (initLibs() == -1) {
		return -1;
	}
	FPSCounter fpsCounter;
	Camera camera;
	UserControls userControls{ window };

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	std::vector<glm::vec3> t_vertices;
	std::vector<glm::vec2> t_uvs;
	std::vector<glm::vec3> t_normals;
	bool res = loadOBJ((RESOURCES_PATH + "suzanne.obj").c_str(), t_vertices, t_uvs, t_normals);

	std::vector<unsigned short> indices;
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;
	indexVBO(t_vertices, t_uvs, t_normals, indices, vertices, uvs, normals);

	// Generate buffers
	GLuint elementbuffer;
	glGenBuffers(1, &elementbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);

	GLuint positionBuffer;
	glGenBuffers(1, &positionBuffer);
	GLuint uvBuffer;
	glGenBuffers(1, &uvBuffer);
	GLuint normalBuffer;
	glGenBuffers(1, &normalBuffer);

	GLuint programID = LoadShaders((SHADERS_PATH + "vertex.vert").c_str(), (SHADERS_PATH + "fragment.frag").c_str());
    glUseProgram(programID);
	GLuint mvpID = glGetUniformLocation(programID, "MVP");
	GLuint modelID = glGetUniformLocation(programID, "Model");
	GLuint viewID = glGetUniformLocation(programID, "View");
	GLuint lightID = glGetUniformLocation(programID, "LightPosition_worldspace");

	GLuint Texture = loadDDS((RESOURCES_PATH + "uvmap.DDS").c_str());
	GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");

	// static parameters
	const auto model = glm::mat4(1.0f);
	glUniformMatrix4fv(modelID, 1, GL_FALSE, &model[0][0]);
    const auto light = glm::vec3(0, 6, 6);
    glUniform3fv(lightID, 1, &light[0]);

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetCursorPos(window, GLFW_WINDOW_WIDTH / 2, GLFW_WINDOW_HEIGHT / 2);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Light blue background
	glClearColor(164.0f / 255, 219.0f / 255, 232.0f / 255, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	initText2D((RESOURCES_PATH + "Holstein.DDS").c_str());

	auto drawBuffers = [&](GLuint matrixID) {
        glUseProgram(programID);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glUniformMatrix4fv(matrixID, 1, GL_FALSE, &camera.mvp[0][0]);
		glUniformMatrix4fv(viewID, 1, GL_FALSE, &camera.view[0][0]);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, Texture);
		glUniform1i(TextureID, 0);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
        glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
        glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), &normals[0], GL_STATIC_DRAW);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_SHORT, (void*)0);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
	};
	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		userControls.update(camera);
		drawBuffers(mvpID);
		char text[256];
		sprintf(text,"%.2f sec", glfwGetTime());
		printText2D(text, 10, 550, 40);
		fpsCounter.log();
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

int initLibs() {
	// Initialise GLFW
	if(!glfwInit()) {
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_RESIZABLE,GL_FALSE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(GLFW_WINDOW_WIDTH, GLFW_WINDOW_HEIGHT, "Realtime Deformations", NULL, NULL);
	if(window == NULL) {
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible.\n" );
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
}


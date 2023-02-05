#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 vertexUV;
layout(location = 2) in vec3 vertexNormal_modelspace;

uniform mat4 MVP;
uniform mat4 Model;
uniform mat4 View;
uniform vec3 LightPosition_worldspace;

out vec2 UV;
out vec3 Normal_cameraspace;
out vec3 LightDirection_cameraspace;
out vec3 EyeDirection_cameraspace;
out float distSquared;

void main(){
  gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);
  UV = vertexUV;
  vec3 modelToLight = (Model * vec4(vertexPosition_modelspace, 1.0)).xyz - LightPosition_worldspace;
  distSquared = dot(modelToLight, modelToLight);

  vec3 vertexPosition_cameraspace = (View * Model * vec4(vertexPosition_modelspace, 1.0)).xyz;
  // Vector that goes from the vertex to the camera, in camera space.
  // In camera space, the camera is at the origin (0,0,0).
  EyeDirection_cameraspace = vec3(0, 0, 0) - vertexPosition_cameraspace;

  // Vector that goes from the vertex to the light, in camera space. M is ommited because it's identity.
  vec3 LightPosition_cameraspace = (View * vec4(LightPosition_worldspace, 1)).xyz;
  LightDirection_cameraspace = LightPosition_cameraspace - vertexPosition_cameraspace;

  // Normal of the the vertex, in camera space
  Normal_cameraspace = (View * Model * vec4(vertexNormal_modelspace, 0)).xyz; // Only correct if ModelMatrix does not scale the model ! Use its inverse transpose if not.
}
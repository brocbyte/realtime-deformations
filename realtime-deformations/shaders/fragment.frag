#version 330 core
in vec2 UV;
in vec3 Normal_cameraspace;
in vec3 LightDirection_cameraspace;
in vec3 EyeDirection_cameraspace;
in float distSquared;
out vec4 color;
uniform sampler2D myTextureSampler;
void main() {
  // Normal of the computed fragment, in camera space
  vec3 n = normalize(Normal_cameraspace);
  // Direction of the light (from the fragment to the light)
  vec3 l = normalize(LightDirection_cameraspace);
  float cosTheta = clamp(dot(n, l), 0, 1);

  // Eye vector (towards the camera)
  vec3 E = normalize(EyeDirection_cameraspace);
  // Direction in which the triangle reflects the light
  vec3 R = reflect(-l,n);
  // Cosine of the angle between the Eye vector and the Reflect vector,
  // clamped to 0
  //  - Looking into the reflection -> 1
  //  - Looking elsewhere -> < 1
  float cosAlpha = clamp( dot( E,R ), 0,1 );

  vec3 materialColor = texture( myTextureSampler, UV).rgb;
  vec3 materialAmbientColor = vec3(0.2,0.2,0.2) * materialColor;
  float lightPower = 60;
  color.xyz = materialAmbientColor +
    materialColor * lightPower * cosTheta / distSquared +
    materialColor * lightPower * pow(cosAlpha,5) / distSquared;
  color.a = 0.8;
}
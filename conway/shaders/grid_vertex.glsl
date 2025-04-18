#version 430

layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 vertex_color;
out vec4 fragment_color;

uniform mat4 MVP;

void main(void)
{
  gl_Position = MVP * vec4(vertex, 1);
  fragment_color = vec4(vertex_color, 1);
}

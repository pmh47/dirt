
#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glext.h>

#include "shaders.h"

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define HEADER \
    "#version 330 core\n" \
    "#extension GL_ARB_separate_shader_objects : enable\n" \
    "#line " STRINGIZE(__LINE__) "\n"

shaders::Shader const shaders::forward_vertex {
    "forward_vertex", GL_VERTEX_SHADER, HEADER R"glsl(

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 colour_in;

layout(location = 0) smooth out vec3 colour_out;

void main() {
    gl_Position = position;
    colour_out = colour_in;
}

)glsl"
};

shaders::Shader const shaders::forward_fragment{
    "forward_fragment", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 colour_in;
layout(location = 0) out vec4 colour_out;

void main() {
    colour_out = vec4(colour_in, 1.f);
}

)glsl"
};

shaders::Shader const shaders::backward_vertex {
    "backward_vertex", GL_VERTEX_SHADER, HEADER R"glsl(

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 barycentric_in;
layout(location = 2) in ivec3 indices_in;

layout(location = 0) smooth out vec3 barycentric_out;
layout(location = 1) flat out ivec3 indices_out;

void main() {
    gl_Position = position;
    barycentric_out = vec3(barycentric_in.x, barycentric_in.y, 1.f - barycentric_in.x - barycentric_in.y);
    indices_out = indices_in;
}

)glsl"
};

shaders::Shader const shaders::backward_fragment{
    "backward_fragment", GL_FRAGMENT_SHADER, HEADER R"glsl(

layout(location = 0) smooth in vec3 barycentric_in;
layout(location = 1) flat in ivec3 indices_in;

layout(location = 0) out vec4 barycentric_and_depth_out;
layout(location = 1) out vec3 indices_out;  // ** integer-valued textures don't seem to work

void main() {
    barycentric_and_depth_out = vec4(barycentric_in, 1.f / gl_FragCoord.w);  // the 'depth' we use here is exactly the clip-space w-coordinate
    indices_out = indices_in;
}

)glsl"
};


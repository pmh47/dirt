
#ifndef SHADERS_H
#define SHADERS_H

#include <string>

namespace shaders
{
    struct Shader {
        std::string name;
        GLenum type;
        std::string source;
    };

    extern Shader const
        forward_vertex,
        forward_fragment,
        backward_vertex,
        backward_fragment;
}

#endif //SHADERS_H

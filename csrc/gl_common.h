
#ifndef GL_COMMON_H
#define GL_COMMON_H

#include <string>
#include <fstream>
#include <sstream>

#define GL_GLEXT_PROTOTYPES

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>

#undef Status
#undef Success
#undef None

#include "tensorflow/core/platform/logging.h"

#include "hwc.h"
#include "shaders.h"

namespace gl_common
{
    // The methods here should only be run on a GL thread!

    static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    inline void initialise_context(int const requiredCudaDevice)
    {
        // Load extensions
        auto const eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");
        auto const eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC) eglGetProcAddress("eglQueryDeviceAttribEXT");
        auto const eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");
        if (!eglQueryDevicesEXT || !eglQueryDeviceAttribEXT || !eglGetPlatformDisplayEXT)
            LOG(FATAL) << "extensions eglQueryDevicesEXT, eglQueryDeviceAttribEXT and eglGetPlatformDisplayEXT not available";

        // Enumerate egl devices
        constexpr int MAX_DEVICES = 16;
        EGLDeviceEXT eglDevs[MAX_DEVICES];
        EGLint numDevices;
        if (!eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices) || numDevices == 0)
            LOG(FATAL) << "no egl devices found";

        // Find which egl device matches the active cuda device, or use device zero for cpu mode (requiredCudaDevice == -1)
        int eglDeviceIndex = 0;
        if (requiredCudaDevice != -1) {
            for (; eglDeviceIndex < numDevices; ++eglDeviceIndex) {
                EGLAttrib cudaDeviceId = -1;
                eglQueryDeviceAttribEXT(eglDevs[eglDeviceIndex], EGL_CUDA_DEVICE_NV, &cudaDeviceId);
                if (cudaDeviceId == requiredCudaDevice)
                    break;
            }
            if (eglDeviceIndex == numDevices)
                LOG(FATAL) << "none of " << numDevices << " egl devices matches the active cuda device";
            LOG(INFO) << "selected egl device #" << eglDeviceIndex << " to match cuda device #" << requiredCudaDevice << " for thread 0x" << std::hex << std::this_thread::get_id();
        }

        // Get and initialise pseudo-display backed by the selected device
        auto const eglDpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[eglDeviceIndex], 0);
        EGLint egl_major, egl_minor;
        eglInitialize(eglDpy, &egl_major, &egl_minor);

        // ** Note that the following will probably return a higher bit depth than we ask for, as it takes the 'best' that satisfies our requirements...
        EGLint numConfigs;
        EGLConfig eglCfg;
        eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

        eglBindAPI(EGL_OPENGL_API);
        auto const eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
        if (eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx) != EGL_TRUE)
            LOG(FATAL) << "eglMakeCurrent failed with error 0x" << std::hex << eglGetError() << " for thread 0x" << std::hex << std::this_thread::get_id();

        LOG(INFO) << "successfully created new GL context on thread 0x" << std::hex << std::this_thread::get_id() <<
            " (EGL = " << egl_major << "." << egl_minor << ", GL = " << glGetString(GL_VERSION) << ", renderer = " << glGetString(GL_RENDERER) << ")";
    }

    inline void print_log(
        void (*const get_log)(GLuint, GLsizei, GLsizei *, char *),
        void (*const get_property)(GLuint, GLenum, GLint *),
        GLenum status_property, GLuint const object, std::string const &name
    ) {
        GLint status;
        get_property(object, status_property, &status);
        if (status != GL_TRUE) {
            char buffer[1024];
            get_log(object, sizeof(buffer), nullptr, buffer);
            LOG(FATAL) << "error compiling / linking " << name << ": " << buffer;
        }
    }

    inline GLuint create_shader(shaders::Shader const &shader)
    {
        auto const source_ptr = shader.source.c_str();
        auto const source_length = static_cast<GLint>(shader.source.length());
        auto const gl_shader = glCreateShader(shader.type);
        glShaderSource(gl_shader, 1, &source_ptr, &source_length);
        glCompileShader(gl_shader);
        print_log(glGetShaderInfoLog, glGetShaderiv, GL_COMPILE_STATUS, gl_shader, shader.name);
        return gl_shader;
    }
}

#endif //GL_COMMON_H

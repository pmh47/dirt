cmake_minimum_required(VERSION 3.8)  # 3.8 gives us built-in CUDA support

project(dirt LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)

# Search for EGL; nvidia drivers ship the library but not headers, so we redistribute those
find_path(EGL_INCLUDE_DIR NAMES EGL/egl.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../external REQUIRED)
find_library(EGL_LIBRARIES NAMES egl EGL REQUIRED)

# Search for cuda headers (using the form of path that tensorflow includes them with), based on cmake-inferred nvcc, or $CUDA_HOME
get_filename_component(NVCC_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
find_path(CUDA_INCLUDE_DIR NAMES cuda/include/cuda.h HINTS ${NVCC_DIR}/../.. PATHS ENV CUDA_HOME REQUIRED)

# Ask tensorflow for its include path; one should therefore make sure cmake is run with the venv active that the op will be used in
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_default_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Tensorflow_INCLUDE_DIRS "${Tensorflow_default_INCLUDE_DIRS}" CACHE PATH "Tensorflow include path")

# Ask tensorflow for its library path
# If using tensorflow earlier than v1.4, this will not work, but can be skipped entirely
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE Tensorflow_default_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Tensorflow_LIB_DIR "${Tensorflow_default_LIB_DIR}" CACHE PATH "Tensorflow library path")
find_library(Tensorflow_LIBRARY tensorflow_framework HINTS ${Tensorflow_LIB_DIR} REQUIRED DOC "Tensorflow framework library; for tensorflow < 1.4, you can set this to blank")

# in the following, we need ../external/tensorflow for cuda_config.h in tf versions with #16959 unfixed
include_directories(SYSTEM ../external/tensorflow ${NSYNC_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} ${EGL_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})
include_directories(${Tensorflow_INCLUDE_DIRS} ${Tensorflow_INCLUDE_DIRS}/external/nsync/public)

# Override g++ ABI version if needed. Official Tensorflow packages used to be built with gcc 4.x, hence use the C++03 ABI; most
# users have a recent version of gcc that defaults to the C++11 ABI, and therefore requires -D_GLIBCXX_USE_CXX11_ABI=0
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.COMPILER_VERSION)" OUTPUT_VARIABLE tensorflow_compiler_version OUTPUT_STRIP_TRAILING_WHITESPACE)
string(SUBSTRING "${tensorflow_compiler_version}" 0 1 tensorflow_compiler_major_version)
if(${tensorflow_compiler_major_version} LESS 5)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --expt-relaxed-constexpr")

add_library(
    rasterise SHARED
    rasterise_egl.cpp rasterise_egl.cu
    rasterise_grad_egl.cpp rasterise_grad_egl.cu rasterise_grad_common.h
    shaders.cpp shaders.h
    gl_dispatcher.h concurrentqueue.h blockingconcurrentqueue.h gl_common.h hwc.h
)

target_compile_features(rasterise PUBLIC cxx_std_11)
target_link_libraries(rasterise ${EGL_LIBRARIES} ${OPENGL_LIBRARIES} ${Tensorflow_LIBRARY})

# Put the compiled library in the python package folder, rather than whatever build folder is being used
set_target_properties(
    rasterise PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../dirt
)

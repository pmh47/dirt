
#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glext.h>

#undef Status
#undef Success
#undef None

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/stream_executor.h"

// This form of include path matches what the TensorFlow headers use
#include <cuda/include/cuda.h>
#include <cuda/include/cuda_runtime_api.h>
#include <cuda/include/cuda_gl_interop.h>

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"

#include <fstream>

#include "hwc.h"
#include "gl_dispatcher.h"
#include "gl_common.h"
#include "rasterise_grad_common.h"

using namespace tensorflow;

REGISTER_OP("RasteriseGrad")
    .Attr("height: int")
    .Attr("width: int")
    .Attr("channels: int = 3")
    .Input("vertices: float32")
    .Input("faces: int32")
    .Input("pixels: float32")
    .Input("grad_pixels: float32")
    .Output("grad_background: float32")
    .Output("grad_vertices: float32")
    .Output("grad_vertex_colors: float32")
    .Output("debug_thingy: float32")
    .SetShapeFn( [] (::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(2));
        c->set_output(1, c->input(0));
        ::tensorflow::shape_inference::ShapeHandle grad_vertex_colours_shape;
        c->ReplaceDim(c->input(0), 2, c->Dim(c->input(2), 3), &grad_vertex_colours_shape);  // i.e. vertex-colors has same shape as vertices, but with final dimension replaced by channel count
        c->set_output(2, grad_vertex_colours_shape);
        c->set_output(3, c->input(3));
        return Status::OK();
    } );

struct VertexBuffer
{
    GLuint gl_index;

private:
    std::size_t buffer_size;
    cudaGraphicsResource_t cuda_resource;
    static int constexpr vertex_stride = (4 + 2) * sizeof(float) + 3 * sizeof(int);

public:

    VertexBuffer() :
        gl_index(0), buffer_size(0), cuda_resource(nullptr)
    {
    }

    void initialise()
    {
        assert(gl_index == 0);
        glGenBuffers(1, &gl_index);
        glBindBuffer(GL_ARRAY_BUFFER, gl_index);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertex_stride, (void *) 0);  // position
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertex_stride, (void *) (4 * sizeof(GLfloat)));  // barycentric
        glEnableVertexAttribArray(2);
        glVertexAttribIPointer(2, 3, GL_INT, vertex_stride, (void *) (6 * sizeof(GLfloat)));  // indices
    }

    void ensure_size(std::size_t const &required_size)
    {
        assert(gl_index != 0);
        if (buffer_size >= required_size)
            return;
        if (buffer_size != 0) {
            // The buffer has previously been initialised and registered, so we should unregister it before reallocating
            // ** is this re-registration actually necessary? is just re-mapping sufficient?
            cudaGraphicsUnregisterResource(cuda_resource);
        }
        glNamedBufferData(gl_index, required_size, nullptr, GL_DYNAMIC_COPY);
        if (auto const err = glGetError())
            LOG(FATAL) << "glNamedBufferData failed: " << err;
        buffer_size = required_size;
        if (auto const err = cudaGraphicsGLRegisterBuffer(&cuda_resource, gl_index, cudaGraphicsRegisterFlagsWriteDiscard))
            LOG(FATAL) << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorName(err);
    }

    void fill_from(
        Tensor const &vertices_tensor, Tensor const &faces_tensor,
        Eigen::GpuDevice const &device
    ) {
        auto const total_bytes = static_cast<std::size_t>(sizeof(float) * vertex_stride * faces_tensor.NumElements());
        ensure_size(total_bytes);
        if (auto const err = cudaGraphicsMapResources(1, &cuda_resource, device.stream()))
            LOG(FATAL) << "cuGraphicsMapResources failed: " << cudaGetErrorName(err);
        Vertex *buffer_ptr;
        std::size_t mapped_size;
        if (auto const err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&buffer_ptr), &mapped_size, cuda_resource))
            LOG(FATAL) << "cuGraphicsResourceGetMappedPointer failed: " << cudaGetErrorName(err);
        assert(mapped_size >= total_bytes);
        TTypes<Vertex, 2>::Tensor buffer_tensormap(buffer_ptr, faces_tensor.dim_size(0), faces_tensor.dim_size(1) * 3);
        launch_vertex_upload(
            buffer_tensormap,
            vertices_tensor, faces_tensor,
            device
        );
        if (auto const err = cudaGraphicsUnmapResources(1, &cuda_resource, device.stream()))  // this synchronises the stream with future graphics calls, so the copying is guaranteed to happen before any rendering
            LOG(FATAL) << "cuGraphicsUnmapResources failed: " << cudaGetErrorName(err);
    }
};

struct RegisteredTexture
{
    // This is used for render buffers (pixels, barycentrics, indices) that are owned and written to by opengl, but must also be read by cuda

    class MappedArray final {

        cudaArray_t array;
        cudaGraphicsResource_t cuda_resource;
        cudaStream_t stream;

    public:

        MappedArray(cudaArray_t const &array_, cudaGraphicsResource_t const &cuda_resource_, cudaStream_t const &stream_) :
            array(array_), cuda_resource(cuda_resource_), stream(stream_)
        {
        }

        MappedArray(MappedArray const &) = delete;
        MappedArray &operator =(MappedArray const &) = delete;

        MappedArray(MappedArray &&other) {
            array = other.array;
            cuda_resource = other.cuda_resource;
            other.cuda_resource = nullptr;
            stream = other.stream;
        }

        ~MappedArray() {
            if (cuda_resource) {
                if (auto const err = cudaGraphicsUnmapResources(1, &cuda_resource, stream))
                    LOG(FATAL) << "cudaGraphicsUnmapResources failed" << cudaGetErrorName(err);
            }
        }

        operator cudaArray_t const &() const {
            return array;
        }
    };

private:
    GLuint gl_index;
    cudaGraphicsResource_t cuda_resource;

public:

    RegisteredTexture() :
        gl_index(0), cuda_resource(nullptr)
    {
    }

    RegisteredTexture(RegisteredTexture const &) = delete;
    RegisteredTexture &operator =(RegisteredTexture const &) = delete;

    void allocate_and_register(GLenum const format, GLenum const attachment, int const width, int const height)
    {
        if (gl_index) {
            glDeleteTextures(1, &gl_index);
            assert(cuda_resource);
            cudaGraphicsUnregisterResource(cuda_resource);  // ** it's unclear whether we actually need to re-register when the size is changed
        }

        glCreateTextures(GL_TEXTURE_2D, 1, &gl_index);
        glTextureStorage2D(gl_index, 1, format, width, height);
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, gl_index, 0);
        if (auto const err = glGetError())
            LOG(FATAL) << "framebuffer texture initialisation failed: " << err;

        if (auto const err = cudaGraphicsGLRegisterImage(&cuda_resource, gl_index, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly | cudaGraphicsRegisterFlagsSurfaceLoadStore))
            LOG(FATAL) << "cudaGraphicsGLRegisterImage failed: " << cudaGetErrorName(err);
    }

    MappedArray get_array(cudaStream_t const &stream)
    {
        if (auto const err = cudaGraphicsMapResources(1, &cuda_resource, stream))  // this also synchronises pending graphics calls with our stream
            LOG(FATAL) << "cudaGraphicsMapResources failed: " << cudaGetErrorName(err);
        cudaArray_t array;
        if (auto const err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0))
            LOG(FATAL) << "cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorName(err);
        return MappedArray(array, cuda_resource, stream);
    }
};

class RasteriseGradOpGpu : public OpKernel
{
    struct PerThreadObjects
    {
        PerThreadObjects() : cuda_context(nullptr)
        {
        }

        PerThreadObjects(PerThreadObjects const &) = delete;
        PerThreadObjects &operator =(PerThreadObjects const &) = delete;

        int frame_height, frame_width, channels;
        int buffer_height, buffer_width;  // these are some multiple of frame_*, depending on batch size
        GLuint framebuffer, depth_buffer;
        VertexBuffer vertex_buffer;
        GLuint program;
        RegisteredTexture barycentrics_and_depth_texture, indices_texture;
        CUcontext cuda_context;  // this is nullptr iff the thread-objects have not yet been initialised
    };

    HWC hwc;

    static GlDispatcher<PerThreadObjects> &get_gl_dispatcher()
    {
        static GlDispatcher<PerThreadObjects> gl_dispatcher;
        return gl_dispatcher;
    }

    static void initialise_per_thread_objects(PerThreadObjects &objects, HWC const &hwc, CUcontext const &cuda_context)
    {
        // This is called directly during the first kernel-invocation when each thread is used. cuda_context refers
        // to the context active in the dispatching thread (but not yet active in our thread)

        if (objects.cuda_context != nullptr)
            LOG(FATAL) << "attempted to reinitialise PerThreadObjects already bound to cuda context " << objects.cuda_context;

        if (auto const err = cuCtxSetCurrent(cuda_context))
            LOG(FATAL) << "cuCtxSetCurrent failed: " << err;
        objects.cuda_context = cuda_context;

        objects.frame_height = hwc.height;
        objects.frame_width = hwc.width;
        objects.channels = hwc.channels;

        CUdevice active_cuda_device;
        if (auto const err = cuCtxGetDevice(&active_cuda_device))
            LOG(FATAL) << "cudaGetDevice failed: " << err;
        gl_common::initialise_context(active_cuda_device);

        objects.buffer_height = objects.buffer_width = 0;
        objects.framebuffer = objects.depth_buffer = 0;

        // ** The shaders we initialise here would preferably be global, not per-thread. However, that
        // ** requires fancier synchronisation to ensure everything is set up (and stored somewhere) before use

        // Load and compile the vertex and fragment shaders
        GLuint const vertex_shader = gl_common::create_shader(shaders::backward_vertex);
        GLuint const fragment_shader = gl_common::create_shader(shaders::backward_fragment);

        // Link the vertex & fragment shaders
        objects.program = glCreateProgram();
        glAttachShader(objects.program, vertex_shader);
        glAttachShader(objects.program, fragment_shader);
        glLinkProgram(objects.program);
        gl_common::print_log(glGetProgramInfoLog, glGetProgramiv, GL_LINK_STATUS, objects.program, "program");
        glUseProgram(objects.program);

        // Set up and bind the vertex buffer and attributes
        objects.vertex_buffer.initialise();

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_SCISSOR_TEST);
    }

    static void reinitialise_framebuffer(PerThreadObjects &objects)
    {
        assert(objects.buffer_width != 0 && objects.buffer_height != 0);

        // Unregister and delete previous buffers
        // ** for depth, we really just need to reallocate storage...
        glDeleteFramebuffers(1, &objects.framebuffer);
        glDeleteRenderbuffers(1, &objects.depth_buffer);

        // Create the framebuffer, and map the color-attachments to fragment shader outputs
        glGenFramebuffers(1, &objects.framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, objects.framebuffer);
        GLenum all_draw_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, all_draw_buffers);  // map fragment-shader output locations to framebuffer attachments
        if (auto const err = glGetError())
            LOG(FATAL) << "framebuffer creation failed: " << err;

        // Set up float-valued rgb(a) textures for barycentrics/depth and indices
        // ** for indices, should be integer-valued! However, doesn't seem to work...
        objects.barycentrics_and_depth_texture.allocate_and_register(GL_RGBA32F, GL_COLOR_ATTACHMENT0, objects.buffer_width, objects.buffer_height);
        objects.indices_texture.allocate_and_register(GL_RGBA32F, GL_COLOR_ATTACHMENT1, objects.buffer_width, objects.buffer_height);

        // Set up a non-texture renderbuffer for depth. This is additional to the copy of the depth values rendered into
        // the barycentrics_and_depth texture, as cuda interop can't read depth-valued textures, and opengl can't use
        // colour-valued textures as a depth buffer
        glGenRenderbuffers(1, &objects.depth_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, objects.depth_buffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, objects.buffer_width, objects.buffer_height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, objects.depth_buffer);
        if (auto const err = glGetError())
            LOG(FATAL) << "depth buffer initialisation failed: " << err;

        LOG(INFO) << "reinitialised framebuffer with size " << objects.buffer_width << " x " << objects.buffer_height;
    }

public:

    explicit RasteriseGradOpGpu(OpKernelConstruction* context) :
        OpKernel(context), hwc(get_hwc(context))
    {
    }

    void Compute(OpKernelContext* context) override
    {
        CUcontext current_context;
        if (auto const err = cuCtxGetCurrent(&current_context))
            LOG(FATAL) << "cuCtxGetCurrent failed: " << err;

        get_gl_dispatcher().dispatch(hwc, [&](PerThreadObjects &objects) {

#ifdef TIME_SECTIONS
            auto start = std::chrono::high_resolution_clock::now();
#endif

            // If this is the first call on this GL thread, activate the cuda context and initialise opengl
            if (!objects.cuda_context) {
                // We haven't yet bound this thread to a cuda context nor initialised the gl context
                initialise_per_thread_objects(objects, hwc, current_context);
            }

            // Sanity-check that the cuda context is set as expected
            CUcontext thread_context;
            cuCtxGetCurrent(&thread_context);
            assert(thread_context == objects.cuda_context);  // this should always be true, as nothing apart from us affects either of these

            // Check input tensors and get references to data

            Tensor const &vertices_tensor = context->input(0);
            OP_REQUIRES(context, vertices_tensor.shape().dims() == 3 && vertices_tensor.shape().dim_size(2) == 4, errors::InvalidArgument("RasteriseGrad expects vertices to be 3D, and vertices.shape[2] == 4"));

            Tensor const &faces_tensor = context->input(1);
            OP_REQUIRES(context, faces_tensor.shape().dims() == 3 && faces_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("RasteriseGrad expects faces to be 3D, and faces.shape[2] == 3"));

            Tensor const &pixels_tensor = context->input(2);
            OP_REQUIRES(
                context,
                pixels_tensor.shape().dims() == 4 && pixels_tensor.shape().dim_size(1) == objects.frame_height &&
                pixels_tensor.shape().dim_size(2) == objects.frame_width && pixels_tensor.shape().dim_size(3) == objects.channels,
                errors::InvalidArgument("RasteriseGrad expects pixels to be 4D, and pixels.shape == [None, height, width, channels]")
            );

            Tensor const &grad_pixels_tensor = context->input(3);
            OP_REQUIRES(
                context,
                grad_pixels_tensor.shape().dims() == 4 && grad_pixels_tensor.shape().dim_size(1) == objects.frame_height &&
                    grad_pixels_tensor.shape().dim_size(2) == objects.frame_width && grad_pixels_tensor.shape().dim_size(3) == objects.channels,
                errors::InvalidArgument("RasteriseGrad expects grad_pixels to be 4D, and grad_pixels.shape == [None, height, width, channels]")
            );

            auto const batch_size = vertices_tensor.shape().dim_size(0);
            OP_REQUIRES(
                context,
                faces_tensor.shape().dim_size(0) == batch_size &&
                    pixels_tensor.shape().dim_size(0) == batch_size && grad_pixels_tensor.shape().dim_size(0) == batch_size,
                errors::InvalidArgument("RasteriseGrad expects all arguments to have same leading (batch) dimension")
            );

            // Allocate output tensors

            Tensor *grad_background_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, pixels_tensor.shape(), &grad_background_tensor));

            Tensor *grad_vertices_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, vertices_tensor.shape(), &grad_vertices_tensor));

            Tensor *grad_vertex_colors_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{batch_size, vertices_tensor.shape().dim_size(1), pixels_tensor.shape().dim_size(3)}, &grad_vertex_colors_tensor));

            Tensor *debug_thingy_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{batch_size, objects.frame_height, objects.frame_width, 3}, &debug_thingy_tensor));

#ifdef TIME_SECTIONS
            std::map<std::string, double> timings;
            auto const time_A = std::chrono::high_resolution_clock::now();
            timings["allocations"] += std::chrono::duration<double>(time_A - start).count();
#endif

            // Check that there are not too many vertices -- in which case our use of float32 for indices will yield incorrect results
            int constexpr max_vertex_count = (1 << std::numeric_limits<float>::digits);
            OP_REQUIRES(
                context,
                vertices_tensor.dim_size(1) <= max_vertex_count,
                errors::InvalidArgument("RasteriseGrad supports a maximum of ", max_vertex_count, " vertices, vs. ", vertices_tensor.dim_size(1), " passed")
            );

            // Check the framebuffer is large enough for our batch; expand it if not
            int const maximum_batch_size = (objects.buffer_width / objects.frame_width) * (objects.buffer_height / objects.frame_height);
            if (batch_size > maximum_batch_size) {
                int const horizontal_count = static_cast<int>(std::sqrt(static_cast<float>(batch_size)) + .1f);
                int const vertical_count = batch_size / horizontal_count + (batch_size % horizontal_count == 0 ? 0 : 1);
                objects.buffer_width = objects.frame_width * horizontal_count;
                objects.buffer_height = objects.frame_height * vertical_count;
                reinitialise_framebuffer(objects);
            }
            int const frames_per_row = objects.buffer_width / objects.frame_width;

            // Get handles to the stream we're running on, to synchronise opengl and cuda
            // ** we assume that the cuda stream underlying exec_stream is the same as that named by device.stream()
            auto const &device = context->eigen_device<Eigen::GpuDevice>();
            auto const exec_stream = context->op_device_context()->stream();  // this is the StreamExecutor stream, not the raw cuda stream device.stream()
            OP_REQUIRES(context, exec_stream, errors::Internal("No GPU stream available"));

            // Expand and register the gl vertex-buffer if required, and fill it with expanded, interleaved vertex data
            objects.vertex_buffer.fill_from(vertices_tensor, faces_tensor, device);

#ifdef TIME_SECTIONS
            auto const time_B = std::chrono::high_resolution_clock::now();
            timings["expand-vertices"] = std::chrono::duration<double>(time_B - time_A).count();
#endif

            for (int index_in_batch = 0; index_in_batch < batch_size; ++index_in_batch) {

                // Find where this frame in the batch should be rendered; set the viewport transform (so geometry
                // ends up in the right place) and scissor region (so clears/etc. don't escape)
                int const frame_x = (index_in_batch % frames_per_row) * objects.frame_width;
                int const frame_y = (index_in_batch / frames_per_row) * objects.frame_height;
                glViewport(frame_x, frame_y, objects.frame_width, objects.frame_height);
                glScissor(frame_x, frame_y, objects.frame_width, objects.frame_height);

                // Clear barycentric_and_depth, indices, and depth buffers
                float const clear_values_barycentric_and_depth[4] = {-1.f, -1.f, -1.f, std::numeric_limits<float>::infinity()};  // for the alpha (clip-w) component, could use the far-plane depth, if we knew it!
                glClearBufferfv(GL_COLOR, 0, clear_values_barycentric_and_depth);
                float const clear_values_indices[4] = {-1.f, -1.f, -1.f, -1.f};
                glClearBufferfv(GL_COLOR, 1, clear_values_indices);
                glClear(GL_DEPTH_BUFFER_BIT);

                // Render the barycentrics and indices for this iib
                // ** It may or may not more efficient to render these in the forward pass and return them in separate outputs
                auto const face_count = static_cast<int>(faces_tensor.dim_size(1));
                glDrawArrays(
                    GL_TRIANGLES,
                    index_in_batch * face_count * 3,
                    face_count * 3
                );
            }

#ifdef TIME_SECTIONS
                glTextureBarrier();  // ensure the rendering is actually done!
                auto const time_D2 = std::chrono::high_resolution_clock::now();
                timings["clear-and-render"] = std::chrono::duration<double>(time_D2 - time_B).count();
#endif

            // Map the barycentric and index buffers to cuda arrays, and dispatch cuda kernels to zero then accumulate the gradients
            launch_grad_assembly(
                *grad_vertices_tensor, *grad_vertex_colors_tensor, *grad_background_tensor, *debug_thingy_tensor,
                objects.barycentrics_and_depth_texture.get_array(device.stream()),  // as these go out of scope, they are unmapped and we synchronise with opengl, ensuring the buffers aren't gl-cleared before we finish using them
                objects.indices_texture.get_array(device.stream()),
                pixels_tensor, grad_pixels_tensor, vertices_tensor,
                objects.buffer_width, objects.buffer_height,
                device
            );

#ifdef TIME_SECTIONS
            auto const time_F = std::chrono::high_resolution_clock::now();
            timings["grad-assembly"] += std::chrono::duration<double>(time_F - time_D2).count();
#endif

#ifdef TIME_SECTIONS
            for (auto &label_and_time : timings) {
                std::cout << label_and_time.first << " : " << label_and_time.second << "s" << std::endl;
            }
#endif
        } );
    }
};

REGISTER_KERNEL_BUILDER(Name("RasteriseGrad").Device(DEVICE_GPU), RasteriseGradOpGpu);



#ifndef HWC_H
#define HWC_H

#include "tensorflow/core/framework/op_kernel.h"

struct HWC
{
    // This captures the attributes (apart from cuda-context) that must be the same for two kernel-instances to share the same gl-context / threads
    int height, width, channels;

    bool operator <(HWC const &other) const {
        return height < other.height || (
            height == other.height && (width < other.width || (
                width == other.width && channels < other.channels)
            )
        );
    }
};

inline HWC get_hwc(tensorflow::OpKernelConstruction *context)
{
    int height, width, channels;
    context->GetAttr("height", &height);
    context->GetAttr("width", &width);
    context->GetAttr("channels", &channels);
    CHECK(channels == 1 || channels == 3);
    CHECK(width > 0 && height > 0);
    return {height, width, channels};
}

#endif //HWC_H

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_DEFORMPSROIPOOLING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_DEFORMPSROIPOOLING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
template <typename Device, typename DType>
struct DeformPSROIPoolForwardLauncher {
    bool operator()(
        const DType* bottom_data, const float spatial_scale, const int num_rois, const int channels, const int height,
        const int width, const int pooled_height, const int pooled_width, const DType* bottom_rois, const DType* bottom_trans,
        const bool no_trans, const float trans_std, const int sample_per_part, const int output_dim, const int num_classes,
        const int group_size, const int part_size, DType* top_data, DType* mapping_channel, const Eigen::GpuDevice& d);
};

template<typename Device, typename DType>
struct DeformPSROIPoolBackwardLauncher {    
    bool operator() (const DType* top_diff, const DType* mapping_channel, const int num_rois, const float spatial_scale,
                     const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
                     const int output_dim, DType* bottom_data_diff, DType* bottom_trans_diff, const DType* bottom_data,
                     const DType* bottom_rois, const DType* bottom_trans, const bool no_trans, const float trans_std,
                     const int sample_per_part, const int group_size, const int part_size,
                     const int num_classes, const int channels_each_class, const Eigen::GpuDevice& d);
};

template <typename Device, typename DType>
struct setZero {
    void operator() (const Device& d, const int n, DType* result_data);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_USER_OPS_DEFORMPSROIPOOLING_OP_GPU_H_
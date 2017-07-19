/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include <stdio.h>
#include <iostream>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "deform_psroi_pooling_op_gpu.h"

namespace tensorflow{
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
REGISTER_OP("DeformPSROIPool")
.Attr("T: {float, double}")
.Attr("output_dim: int")
.Attr("group_size: int")
.Attr("pooled_size: int")
.Attr("part_size: int = 0")
.Attr("sample_per_part: int = 1")
.Attr("spatial_scale: float")
.Attr("trans_std: float = 0.0")
.Attr("no_trans: bool = false")
.Attr("data_format: { 'NHWC', 'NCHW' } = 'NCHW' ")
.Input("bottom_data: T")
.Input("bottom_rois: T")
.Input("trans: T")
.Output("top_data: T")
.Output("mapping_channel: T")
.SetShapeFn([](InferenceContext* c) {
    string data_format;
    TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));
    if(data_format != "NCHW"){
        return errors::InvalidArgument(
            "currently only support NCHW");

    }
    ShapeHandle dims;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &dims));
    // DimensionHandle channels;
    // channels = c->Dim(dims, 1);
    int64 output_dim;
    TF_RETURN_IF_ERROR(c->GetAttr("output_dim", &output_dim));
    ShapeHandle dims_rois;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &dims_rois));
    DimensionHandle num_rois;
    num_rois = c->Dim(dims_rois, 0);

    int64 pooled_size;
    TF_RETURN_IF_ERROR(c->GetAttr("pooled_size", &pooled_size));
    ShapeHandle output_shape =\
                              c->MakeShape({num_rois, output_dim, pooled_size, pooled_size});
    c->set_output(0, output_shape);
    c->set_output(1, output_shape);
    return Status::OK();
});

REGISTER_OP("DeformPSROIPoolGrad")
.Attr("T: {float, double}")
.Attr("spatial_scale: float")
.Attr("output_dim: int")
.Attr("group_size: int")
.Attr("pooled_size: int")
.Attr("part_size: int = 0")
.Attr("sample_per_part: int = 1")
.Attr("trans_std: float = 0.0")
.Attr("no_trans: bool = false")
.Input("bottom_data: T")
.Input("bottom_rois: T")
.Input("trans: T")
.Input("mapping_channel: T")
.Input("grad: T")
.Output("data_grad: T")
.Output("trans_grad: T");

template <typename Device, typename T>
class DeformPSROIPoolOp : public OpKernel {
public:
    explicit DeformPSROIPoolOp(OpKernelConstruction* context) : OpKernel(context) {}
    void Compute(OpKernelContext* context) override {}
private:
    int output_dim_;
    int group_size_;
    int spatial_scale_;
};



// template <class T>
// bool DeformPSROIPoolForwardLauncher(
//     const float* bottom_data, const float spatial_scale, const int num_rois, const int channels, const int height,
//     const int width, const int pooled_height, const int pooled_width, const float* bottom_rois,
//     const int output_dim, const int group_size, float* top_data, int* mapping_channel, const Eigen::GpuDevice& d);

template <class T>
static void DeformPSROIPoolingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_trans,
    const bool no_trans, const float spatial_scale, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height, const int pooled_width, const Tensor* bottom_rois,
    const float trans_std, const int sample_per_part,
    const int output_dim, const int group_size, const int part_size, const TensorShape& tensor_output_shape)
{
    const GPUDevice& device = context->eigen_device<Eigen::GpuDevice>();
    Tensor* output = nullptr;
    Tensor* mapping_channel = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
    OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &mapping_channel));
    setZero<GPUDevice, T>()(device, mapping_channel->NumElements(), mapping_channel->flat<T>().data());
    // setZero<GPUDevice, T>()(device, output->NumElements(), output->flat<T>().data());    
    if (!context->status().ok()) {
        return;
    }
    int num_class =  no_trans ? 1 : bottom_trans->dim_size(1)/2;
    // auto temp = bottom_trans->flat<T>().data();
    // std::cout <<"\n" << 
    // printf("%d", (static_cast<T>(temp[0])));
    //  << static_cast<T>(temp[1]) << static_cast<T>(temp[2]);
    DeformPSROIPoolForwardLauncher<GPUDevice,T>()(
        bottom_data->flat<T>().data(), spatial_scale, num_rois, channels,
        height, width, pooled_height, pooled_width, bottom_rois->flat<T>().data(), bottom_trans->flat<T>().data(), no_trans,
        trans_std, sample_per_part, output_dim, num_class, group_size, part_size,
        output->flat<T>().data(), mapping_channel->flat<T>().data(), device);
}

template <typename T>
class DeformPSROIPoolOp<GPUDevice, T> : public OpKernel {
public:
    typedef Eigen::GpuDevice Device;

    explicit DeformPSROIPoolOp(OpKernelConstruction* context) : OpKernel(context) {

        OP_REQUIRES_OK(context,
                       context->GetAttr("output_dim", &output_dim_));

        OP_REQUIRES_OK(context,
                       context->GetAttr("group_size", &group_size_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("pooled_size", &pooled_size_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("spatial_scale", &spatial_scale_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("part_size", &part_size_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("sample_per_part", &sample_per_part_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("trans_std", &trans_std_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("no_trans", &no_trans_));

    }

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& bottom_data = context->input(0);
        const Tensor& bottom_rois = context->input(1);
        const Tensor& bottom_trans = context->input(2);

        // data should have 4 dimensions.
        OP_REQUIRES(context, bottom_data.dims() == 4,
                    errors::InvalidArgument("data must be 4-dimensional"));

        // rois should have 2 dimensions.
        OP_REQUIRES(context, bottom_rois.dims() == 2,
                    errors::InvalidArgument("rois must be 2-dimensional"));
        // trans should have 4 dimensions.
        OP_REQUIRES(context, bottom_trans.dims() == 4,
                    errors::InvalidArgument("trans must be 4-dimensional"));
        // Number of ROIs
        int num_rois = bottom_rois.dim_size(0);
        // batch size
        int batch_size = bottom_data.dim_size(0);
        // data height
        int data_height = bottom_data.dim_size(2);
        // data width
        int data_width = bottom_data.dim_size(3);
        // Number of channels
        int num_channels = bottom_data.dim_size(1);

        int pooled_height = pooled_size_;

        int pooled_width = pooled_size_;

        // construct the output shape
        int dims[4];
        dims[0] = num_rois;
        dims[1] = output_dim_;
        dims[2] = pooled_height;
        dims[3] = pooled_width;
        TensorShape output_shape;
        TensorShapeUtils::MakeShape(dims, 4, &output_shape);
        DeformPSROIPoolingKernel<T>(context, &bottom_data, &bottom_trans, no_trans_, spatial_scale_, num_rois, num_channels, data_height, data_width,
                                    pooled_height, pooled_width, &bottom_rois, trans_std_, sample_per_part_, output_dim_, group_size_, part_size_, output_shape);

    }
private:
    int output_dim_;
    int group_size_;
    int part_size_;
    int sample_per_part_;
    int pooled_size_;
    float spatial_scale_;
    float trans_std_;
    bool no_trans_;
};

// REGISTER_KERNEL_BUILDER(Name("DeformPSROIPool").Device(DEVICE_GPU).TypeConstraint<float>("T"), DeformPSROIPoolOp<Eigen::GpuDevice, float>);

// template <class T>
// bool DeformPSROIPoolBackwardLauncher(const float* top_diff, const int* mapping_channel, const int num_rois, const float spatial_scale,
//                                const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
//                                const int output_dim, float* bottom_diff, const float* bottom_rois, const Eigen::GpuDevice& d);

template <class T>
static void DeformPSROIPoolingGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* out_backprop, const Tensor* mapping_channel, const int num_rois,
    const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
    const int output_dim, const Tensor* bottom_rois, const Tensor* bottom_trans, const TensorShape& tensor_data_shape,
    const bool no_trans, const float trans_std, const int sample_per_part, const int group_size, const int part_size)
{
    Tensor* grad_data = nullptr;
    Tensor* grad_trans = nullptr;
    const TensorShape& tensor_trans_shape = bottom_trans->shape();
    OP_REQUIRES_OK(context, context->allocate_output(0, tensor_data_shape, &grad_data));
    OP_REQUIRES_OK(context, context->allocate_output(1, tensor_trans_shape, &grad_trans));
    int num_class =  no_trans ? 1 : bottom_trans->dim_size(1)/2;
    const int channels_each_class = output_dim / num_class;
    if (!context->status().ok()) {
        return;
    }
    const GPUDevice& device = context->eigen_device<Eigen::GpuDevice>();
    setZero<GPUDevice, T>()(device, grad_data->NumElements(), grad_data->flat<T>().data());    
    setZero<GPUDevice, T>()(device, grad_trans->NumElements(), grad_trans->flat<T>().data());
    DeformPSROIPoolBackwardLauncher<GPUDevice,T>()(
        out_backprop->flat<T>().data(), mapping_channel->flat<T>().data(), num_rois, spatial_scale, channels, height, width,
        pooled_height, pooled_width, output_dim, grad_data->flat<T>().data(), grad_trans->flat<T>().data(),
        bottom_data->flat<T>().data(), bottom_rois->flat<T>().data(), bottom_trans->flat<T>().data(), no_trans, trans_std,
        sample_per_part, group_size, part_size, num_class, channels_each_class, device);
}


// compute gradient
template <class Device, class T>
class DeformPSROIPoolGradOp : public OpKernel {
public:
    explicit DeformPSROIPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {

        OP_REQUIRES_OK(context,
                       context->GetAttr("output_dim", &output_dim_));

        OP_REQUIRES_OK(context,
                       context->GetAttr("group_size", &group_size_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("pooled_size", &pooled_size_));

        // Get the spatial scale
        OP_REQUIRES_OK(context,
                       context->GetAttr("spatial_scale", &spatial_scale_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("part_size", &part_size_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("sample_per_part", &sample_per_part_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("trans_std", &trans_std_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("no_trans", &no_trans_));
    }

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& bottom_data = context->input(0);
        const Tensor& bottom_rois = context->input(1);
        const Tensor& bottom_trans = context->input(2);
        const Tensor& mapping_channel = context->input(3);
        const Tensor& out_backprop = context->input(4);

        // data should have 4 dimensions.
        OP_REQUIRES(context, bottom_data.dims() == 4,
                    errors::InvalidArgument("data must be 4-dimensional"));

        // rois should have 2 dimensions.
        OP_REQUIRES(context, bottom_rois.dims() == 2,
                    errors::InvalidArgument("rois must be 2-dimensional"));

        OP_REQUIRES(context, mapping_channel.dims() == 4,
                    errors::InvalidArgument("mapping_channel must be 4-dimensional"));

        OP_REQUIRES(context, out_backprop.dims() == 4,
                    errors::InvalidArgument("out_backprop must be 4-dimensional"));

        // Number of ROIs
        int num_rois = bottom_rois.dim_size(0);
        // batch size
        int batch_size = bottom_data.dim_size(0);
        // data height
        int data_height = bottom_data.dim_size(2);
        // data width
        int data_width = bottom_data.dim_size(3);
        // Number of channels
        int channels = bottom_data.dim_size(1);

        int pooled_height = out_backprop.dim_size(2);

        int pooled_width = out_backprop.dim_size(3);

        int output_dim = out_backprop.dim_size(1);
        std::cout<<pooled_height<<" "<<pooled_width<<" "<<output_dim;

        // construct the output shape
        TensorShape output_shape = bottom_data.shape();

        DeformPSROIPoolingGradKernel<T>(
            context, &bottom_data, &out_backprop, &mapping_channel, num_rois, spatial_scale_, channels,
            data_height, data_width, pooled_height, pooled_width, output_dim, &bottom_rois, &bottom_trans,
            output_shape, no_trans_, trans_std_, sample_per_part_, group_size_, part_size_
        );

    }
private:
    int output_dim_;
    int group_size_;
    int part_size_;
    int sample_per_part_;
    int pooled_size_;
    float spatial_scale_;
    float trans_std_;
    bool no_trans_;
};

#if GOOGLE_CUDA

#define REGISTER(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("DeformPSROIPool").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DeformPSROIPoolOp<GPUDevice, T>);                                    \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("DeformPSROIPoolGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DeformPSROIPoolGradOp<GPUDevice, T>);

// TF_CALL_GPU_NUMBER_TYPES(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA


// REGISTER_KERNEL_BUILDER(Name("DeformPSROIPoolGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), DeformPSROIPoolGradOp<Eigen::GpuDevice, float>);
}
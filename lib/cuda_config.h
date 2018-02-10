/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

/*
# If coming across: cudaCheckError() failed : invalid device function. change -arch=sm_xx accordingly.

# Which CUDA capabilities do we want to pre-build for?
# https://developer.nvidia.com/cuda-gpus
#   Compute/shader model   Cards
#   6.1		      P4, P40, Titan X so CUDA_MODEL = 61
#   6.0                    P100 so CUDA_MODEL = 60
#   5.2                    M40
#   3.7                    K80
#   3.5                    K40, K20
#   3.0                    K10, Grid K520 (AWS G2)
#   Other Nvidia shader models should work, but they will require extra startup
#   time as the code is pre-optimized for them.
# CUDA_MODELS=30 35 37 52 60 61
*/

// DO NOT EDIT: automatically generated file
#ifndef CUDA_CUDA_CONFIG_H_
#define CUDA_CUDA_CONFIG_H_
// please modify the TF_CUDA_CAPABILITIES according to the above list and
// your gpu model.

#define TF_CUDA_CAPABILITIES CudaVersion("3.7")

#define TF_CUDA_VERSION "8.0"
#define TF_CUDNN_VERSION "6"

#define TF_CUDA_TOOLKIT_PATH "/usr/local/cuda-8.0"

#endif  // CUDA_CUDA_CONFIG_H_
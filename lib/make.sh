#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC


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



CUDA_HOME=/usr/local/cuda/

if [ ! -f $TF_INC/tensorflow/stream_executor/cuda/cuda_config.h ]; then
    cp ./cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/
fi

cd roi_pooling_layer

nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D GOOGLE_CUDA -arch=sm_37

## if you install tf using already-built binary, or gcc version 4.x, uncomment the two lines below
g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
	roi_pooling_op.cu.o -I $TF_INC -fPIC -D GOOGLE_CUDA -lcudart -L $CUDA_HOME/lib64

# for gcc5-built tf
# g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
# 	roi_pooling_op.cu.o -I $TF_INC -fPIC -D GOOGLE_CUDA -lcudart -L $CUDA_HOME/lib64 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..


# add building psroi_pooling layer
cd psroi_pooling_layer
nvcc -std=c++11 -c -o psroi_pooling_op.cu.o psroi_pooling_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D GOOGLE_CUDA -arch=sm_37


## if you install tf using already-built binary, or gcc version 4.x, uncomment the two lines below
g++ -std=c++11 -shared -o psroi_pooling.so psroi_pooling_op.cc \
	psroi_pooling_op.cu.o -I $TF_INC -fPIC -D GOOGLE_CUDA -lcudart -L $CUDA_HOME/lib64
# for gcc5-built tf
# g++ -std=c++11 -shared -o psroi_pooling.so psroi_pooling_op.cc \
# 	psroi_pooling_op.cu.o -I $TF_INC -fPIC -D GOOGLE_CUDA -lcudart -L $CUDA_HOME/lib64 -D_GLIBCXX_USE_CXX11_ABI=0

cd ..

cd deform_psroi_pooling_layer
nvcc -std=c++11 -c -o deform_psroi_pooling_op.cu.o deform_psroi_pooling_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D GOOGLE_CUDA -arch=sm_37

## if you install tf using already-built binary, or gcc version 4.x, uncomment the three lines below
g++ -std=c++11 -shared -o deform_psroi_pooling.so deform_psroi_pooling_op.cc deform_psroi_pooling_op.cu.o -I \
    $TF_INC -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I \
    $CUDA_HOME/include
# for gcc5-built tf
# g++ -std=c++11 -shared -o deform_psroi_pooling.so deform_psroi_pooling_op.cc deform_psroi_pooling_op.cu.o \
#   -I $TF_INC -fPIC -D GOOGLE_CUDA -lcudart -L $CUDA_HOME/lib64 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

cd deform_conv_layer
nvcc -std=c++11 -c -o deform_conv.cu.o deform_conv.cu.cc -I $TF_INC -D\
          GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-8.0/lib64/ --expt-relaxed-constexpr -arch=sm_37
## if you install tf using already-built binary, or gcc version 4.x, uncomment the three lines below
g++ -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o -I\
      $TF_INC -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I\
      $CUDA_HOME/include
# for gcc5-built tf
# g++ -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o \
#   -I $TF_INC -fPIC -D GOOGLE_CUDA -lcudart -L $CUDA_HOME/lib64 -D_GLIBCXX_USE_CXX11_ABI=0

cd ..

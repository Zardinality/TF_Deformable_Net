#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
typedef std::vector<int32> TShape;

inline int ProdShape(const TensorShape &shape, int start) {
    int64 res = 1;
    for(int i=start; i<shape.dims(); i++) {
        res*=shape.dim_size(i);
    }
    return res;
}

inline std::vector<int> ToVector(const TensorShape &shape) {
    // int64 res = 1;
    std::vector<int> res;
    for(int i=0; i<shape.dims(); i++) {
        res.push_back(shape.dim_size(i));
    }
    return res;
}

inline TShape ToVector(const TShape &shape) {
    // int64 res = 1;
    return shape;
}

inline TensorShape Slice(const TensorShape &shape, int start, int end) {
    TensorShape temp = shape;
    for(int i=0; i<start; i++) {
        temp.RemoveDim(0);
    }
    for(int i=0; i<shape.dims()-end; i++) {
        temp.RemoveDim(temp.dims()-1);
    }
    return temp;
}

struct DeformConvParam {
    DeformConvParam(TShape kernel, TShape stride,
                    TShape pad, TShape rates, int num_group, int num_filter,
                    bool no_bias): kernel(kernel), stride(stride), pad(pad), num_group(num_group), num_filter(num_filter), no_bias(no_bias), rates(rates) {};
    TShape kernel, stride, pad, rates;
    int num_group;
    int num_filter;
    bool no_bias;
};


}